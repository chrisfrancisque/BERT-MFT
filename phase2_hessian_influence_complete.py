"""
Phase 2 Complete: Ground Truth + Hessian Influence + Visualization + Collective Removal
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# Set up TPU
os.environ["PJRT_DEVICE"] = "TPU"
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
print(f"Using TPU: {device}")

from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from evaluation import ModelEvaluator

# Configure
config.train_samples = 1000
config.batch_size = 16
config.device = str(device)
config.use_tpu = True

class HessianInfluenceCalculator:
    """
    Implements proper influence functions with Hessian inverse using Conjugate Gradient
    """
    
    def __init__(self, model, device, damping=0.01, cg_iterations=10):
        self.model = model
        self.device = device
        self.damping = damping
        self.cg_iterations = cg_iterations
        self.exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
    def compute_hessian_vector_product(self, vector_dict, dataloader, num_batches=5):
        """
        Compute Hessian-vector product H*v using finite differences
        H*v ≈ ∇L(θ + εv) - ∇L(θ) / ε
        """
        epsilon = 1e-3
        
        # Save original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if not any(pat in name for pat in self.exclude_patterns):
                original_params[name] = param.data.clone()
        
        # Perturb parameters: θ = θ + ε*v
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in vector_dict:
                    param.data += epsilon * vector_dict[name].to(self.device)
        xm.mark_step()
        
        # Compute gradient at perturbed point
        grad_plus = self._compute_gradient(dataloader, num_batches)
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.data = original_params[name]
        xm.mark_step()
        
        # Compute gradient at original point
        grad_orig = self._compute_gradient(dataloader, num_batches)
        
        # Compute HVP = (grad_plus - grad_orig) / epsilon
        hvp = {}
        for name in vector_dict:
            if name in grad_plus and name in grad_orig:
                hvp[name] = (grad_plus[name] - grad_orig[name]) / epsilon
        
        return hvp
    
    def _compute_gradient(self, dataloader, num_batches):
        """Compute gradient over specified number of batches"""
        self.model.eval()
        self.model.zero_grad()
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if not any(pat in name for pat in self.exclude_patterns):
                gradients[name] = torch.zeros_like(param, device=self.device)
        
        batch_count = 0
        for batch in dataloader:
            if batch_count >= num_batches:
                break
                
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            if not torch.isnan(loss):
                loss.backward()
                
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in gradients and param.grad is not None:
                            gradients[name] += param.grad
                            param.grad.zero_()
                
                batch_count += 1
        
        xm.mark_step()
        
        # Average and move to CPU
        for name in gradients:
            gradients[name] = (gradients[name] / batch_count).cpu()
        
        return gradients
    
    def conjugate_gradient_solve(self, b_dict, train_dataloader):
        """
        Solve H*x = b using Conjugate Gradient
        Returns x = H^(-1)*b
        """
        print(f"Starting CG solve with {self.cg_iterations} iterations...")
        
        # Initialize x = 0
        x = {name: torch.zeros_like(b_dict[name]) for name in b_dict}
        
        # r = b - H*x = b (since x=0)
        r = {name: b_dict[name].clone() for name in b_dict}
        
        # p = r
        p = {name: r[name].clone() for name in r}
        
        # rsold = r^T * r
        rsold = sum((r[name] * r[name]).sum().item() for name in r)
        
        for iteration in range(self.cg_iterations):
            # Compute H*p
            Hp = self.compute_hessian_vector_product(p, train_dataloader)
            
            # Add damping: Hp = Hp + damping * p
            for name in Hp:
                Hp[name] = Hp[name] + self.damping * p[name]
            
            # alpha = rsold / (p^T * H * p)
            pHp = sum((p[name] * Hp[name]).sum().item() for name in p)
            
            if abs(pHp) < 1e-10:
                print(f"CG: Early termination at iteration {iteration}, pHp too small")
                break
            
            alpha = rsold / pHp
            
            # x = x + alpha * p
            for name in x:
                x[name] = x[name] + alpha * p[name]
            
            # r = r - alpha * H * p
            for name in r:
                r[name] = r[name] - alpha * Hp[name]
            
            # Check convergence
            rsnew = sum((r[name] * r[name]).sum().item() for name in r)
            
            if np.sqrt(rsnew) < 1e-6:
                print(f"CG converged in {iteration + 1} iterations")
                break
            
            # p = r + (rsnew/rsold) * p
            beta = rsnew / rsold
            for name in p:
                p[name] = r[name] + beta * p[name]
            
            rsold = rsnew
            
            print(f"  CG iteration {iteration + 1}: residual = {np.sqrt(rsnew):.6f}")
        
        return x
    
    def compute_influence_for_parameters(self, param_indices, val_gradient, train_dataloader):
        """
        Compute influence for a list of parameter indices
        I(θ_i) = -∇L_val^T * H^(-1) * e_i
        """
        influences = {}
        
        # First compute H^(-1) * ∇L_val
        print("Computing H^(-1) * ∇L_val...")
        h_inv_grad = self.conjugate_gradient_solve(val_gradient, train_dataloader)
        
        # Now for each parameter, compute influence
        print("Computing influences for each parameter...")
        for param_info in tqdm(param_indices, desc="Computing influences"):
            param_name = param_info['param_name']
            local_idx = param_info['local_idx']
            global_idx = param_info['global_idx']
            
            if param_name in h_inv_grad:
                # Get the influence value for this specific parameter
                h_inv_grad_flat = h_inv_grad[param_name].flatten()
                influence = -h_inv_grad_flat[local_idx].item()
            else:
                influence = 0.0
            
            influences[global_idx] = influence
        
        return influences

class Phase2HessianValidator:
    """
    Enhanced validator with proper Hessian-based influence functions
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
        
    def get_all_parameter_indices(self):
        """Get flattened indices of all parameters"""
        exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
        param_indices = []
        current_idx = 0
        
        for name, param in self.model.named_parameters():
            if any(pat in name for pat in exclude_patterns):
                current_idx += param.numel()
                continue
                
            for local_idx in range(param.numel()):
                param_indices.append({
                    'global_idx': current_idx + local_idx,
                    'param_name': name,
                    'local_idx': local_idx,
                    'shape': param.shape
                })
            current_idx += param.numel()
            
        return param_indices
    
    def sample_parameters(self, n_samples=10000):
        """Randomly sample parameters"""
        print(f"\nSampling {n_samples} parameters...")
        
        all_indices = self.get_all_parameter_indices()
        print(f"Total available parameters: {len(all_indices):,}")
        
        if len(all_indices) <= n_samples:
            sampled = all_indices
        else:
            sampled = random.sample(all_indices, n_samples)
        
        print(f"Sampled {len(sampled)} parameters")
        return sampled
    
    def compute_baseline_metrics(self, dataloader):
        """Compute baseline loss and accuracy"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing baseline"):
                outputs = self.model(**batch)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
        
        xm.mark_step()
        return total_loss / num_batches, correct / total
    
    def compute_ground_truth(self, sampled_params, dataloader, baseline_loss):
        """Compute ground truth by actually zeroing each parameter"""
        print("\n" + "="*70)
        print("COMPUTING GROUND TRUTH")
        print("="*70)
        
        ground_truth = {}
        original_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        for i, param_info in enumerate(tqdm(sampled_params, desc="Ground truth")):
            param_name = param_info['param_name']
            local_idx = param_info['local_idx']
            
            # Zero the parameter
            param = dict(self.model.named_parameters())[param_name]
            original_value = param.data.flatten()[local_idx].item()
            
            with torch.no_grad():
                param.data.flatten()[local_idx] = 0
            xm.mark_step()
            
            # Compute new loss
            self.model.eval()
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    outputs = self.model(**batch)
                    loss = F.cross_entropy(outputs.logits, batch['labels'])
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
            
            xm.mark_step()
            zeroed_loss = total_loss / num_batches
            delta_loss = zeroed_loss - baseline_loss
            
            ground_truth[param_info['global_idx']] = {
                'param_name': param_name,
                'local_idx': local_idx,
                'original_value': original_value,
                'delta_loss': delta_loss,
                'label': 'detrimental' if delta_loss < 0 else 'beneficial'
            }
            
            # Restore parameter
            with torch.no_grad():
                param.data.flatten()[local_idx] = original_value
            xm.mark_step()
            
            # Progress update
            if (i + 1) % 100 == 0:
                det_count = sum(1 for v in ground_truth.values() if v['label'] == 'detrimental')
                print(f"Progress: {i+1}/{len(sampled_params)}, Detrimental: {det_count}/{i+1}")
        
        return ground_truth
    
    def compute_validation_gradient(self, dataloader):
        """Compute gradient on validation set"""
        print("Computing validation gradient...")
        self.model.eval()
        self.model.zero_grad()
        
        exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
        num_batches = 0
        for batch in tqdm(dataloader, desc="Validation gradient"):
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            if not torch.isnan(loss):
                loss.backward()
                num_batches += 1
        
        xm.mark_step()
        
        # Extract and average gradients
        val_gradient = {}
        for name, param in self.model.named_parameters():
            if not any(pat in name for pat in exclude_patterns) and param.grad is not None:
                val_gradient[name] = (param.grad / num_batches).cpu()
        
        return val_gradient
    
    def test_collective_removal(self, ground_truth, dataloader, baseline_loss, baseline_acc):
        """Test removing all ground-truth detrimental parameters together"""
        print("\n" + "="*70)
        print("TESTING COLLECTIVE REMOVAL")
        print("="*70)
        
        # Get all detrimental parameters
        detrimental_params = [(v['param_name'], v['local_idx']) 
                             for v in ground_truth.values() 
                             if v['label'] == 'detrimental']
        
        print(f"Removing {len(detrimental_params)} detrimental parameters collectively...")
        
        # Save original state
        original_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Zero all detrimental parameters
        with torch.no_grad():
            for param_name, local_idx in detrimental_params:
                param = dict(self.model.named_parameters())[param_name]
                param.data.flatten()[local_idx] = 0
        xm.mark_step()
        
        # Evaluate
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating collective removal"):
                outputs = self.model(**batch)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
        
        xm.mark_step()
        
        collective_loss = total_loss / num_batches
        collective_acc = correct / total
        
        # Restore model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = original_state[name]
        xm.mark_step()
        
        results = {
            'num_removed': len(detrimental_params),
            'baseline_loss': baseline_loss,
            'collective_loss': collective_loss,
            'delta_loss': collective_loss - baseline_loss,
            'baseline_accuracy': baseline_acc,
            'collective_accuracy': collective_acc,
            'delta_accuracy': collective_acc - baseline_acc
        }
        
        print(f"\nCollective Removal Results:")
        print(f"  Parameters removed: {len(detrimental_params)}")
        print(f"  Baseline Loss: {baseline_loss:.6f}")
        print(f"  Collective Loss: {collective_loss:.6f}")
        print(f"  ΔLoss: {results['delta_loss']:.6f}")
        print(f"  Baseline Accuracy: {baseline_acc:.4f}")
        print(f"  Collective Accuracy: {collective_acc:.4f}")
        print(f"  ΔAccuracy: {results['delta_accuracy']:.4f} ({results['delta_accuracy']*100:.2f}%)")
        
        return results

def create_visualizations(ground_truth, predictions, output_dir):
    """Create and save visualization plots including confusion matrix"""
    print("\nCreating visualizations...")
    
    # Prepare data
    true_labels = []
    pred_labels = []
    influences = []
    delta_losses = []
    
    for idx in ground_truth:
        true_labels.append(1 if ground_truth[idx]['label'] == 'detrimental' else 0)
        pred_labels.append(1 if predictions[idx]['prediction'] == 'detrimental' else 0)
        influences.append(predictions[idx]['influence'])
        delta_losses.append(ground_truth[idx]['delta_loss'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
               xticklabels=['Beneficial', 'Detrimental'],
               yticklabels=['Beneficial', 'Detrimental'])
    axes[0, 0].set_title('Confusion Matrix (Hessian-based Influence)')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Add text with percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = 100 * cm[i, j] / total
            text = axes[0, 0].texts[i * 2 + j]
            text.set_text(f'{cm[i, j]}\n({percentage:.1f}%)')
    
    # 2. Influence vs Delta Loss Scatter
    axes[0, 1].scatter(influences, delta_losses, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Hessian Influence Score')
    axes[0, 1].set_ylabel('Actual ΔLoss')
    
    # Calculate correlation
    correlation = np.corrcoef(influences, delta_losses)[0, 1]
    axes[0, 1].set_title(f'Hessian Influence vs Actual Loss Change (r={correlation:.3f})')
    
    # 3. Distribution of Influence Scores
    axes[1, 0].hist(influences, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', label='Zero influence')
    axes[1, 0].set_xlabel('Hessian Influence Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Hessian Influence Scores')
    axes[1, 0].legend()
    
    # 4. Accuracy by Influence Quantiles
    influences_arr = np.array(influences)
    delta_losses_arr = np.array(delta_losses)
    
    quantiles = [0, 25, 50, 75, 100]
    accuracies = []
    
    for i in range(len(quantiles) - 1):
        lower = np.percentile(influences_arr, quantiles[i])
        upper = np.percentile(influences_arr, quantiles[i + 1])
        mask = (influences_arr >= lower) & (influences_arr < upper)
        
        # Accuracy: correctly predicting detrimental
        correct = np.sum((influences_arr[mask] > 0) == (delta_losses_arr[mask] < 0))
        total = np.sum(mask)
        acc = correct / total if total > 0 else 0
        accuracies.append(acc * 100)
    
    axes[1, 1].bar(range(4), accuracies, tick_label=['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1, 1].axhline(y=50, color='r', linestyle='--', alpha=0.3, label='Random chance')
    axes[1, 1].set_xlabel('Influence Quartile')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Prediction Accuracy by Influence Quartile')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hessian_analysis_plots.png'), dpi=150)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/hessian_analysis_plots.png")

def main():
    print("\n" + "="*70)
    print("PHASE 2 WITH PROPER HESSIAN-BASED INFLUENCE")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'phase2_hessian_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")
    
    # Load model
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump({
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForSequenceClassification"],
                "num_labels": 2,
                "model_type": "bert"
            }, f)
    
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    # Load data
    print("Loading datasets...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Initialize components
    validator = Phase2HessianValidator(model, device)
    hessian_calc = HessianInfluenceCalculator(model, device, damping=0.01, cg_iterations=10)
    
    # Sample parameters
    sampled_params = validator.sample_parameters(n_samples=10000)
    
    # Compute baseline
    baseline_loss, baseline_acc = validator.compute_baseline_metrics(eval_dataloader)
    print(f"\n" + "="*50)
    print(f"BASELINE METRICS")
    print(f"  Loss: {baseline_loss:.6f}")
    print(f"  Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print("="*50)
    
    # STEP 1: Ground Truth
    ground_truth = validator.compute_ground_truth(sampled_params, eval_dataloader, baseline_loss)
    
    with open(os.path.join(output_dir, 'ground_truth.json'), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    det_count = sum(1 for v in ground_truth.values() if v['label'] == 'detrimental')
    print(f"\nGround Truth Summary:")
    print(f"  Detrimental: {det_count} ({det_count*100/len(ground_truth):.1f}%)")
    print(f"  Beneficial: {len(ground_truth) - det_count} ({(len(ground_truth)-det_count)*100/len(ground_truth):.1f}%)")
    
    # STEP 2: Compute Hessian-based Influence
    print("\n" + "="*70)
    print("COMPUTING HESSIAN-BASED INFLUENCE")
    print("="*70)
    
    val_gradient = validator.compute_validation_gradient(eval_dataloader)
    hessian_influences = hessian_calc.compute_influence_for_parameters(
        sampled_params, val_gradient, train_dataloader
    )
    
    # Create predictions based on Hessian influence
    hessian_predictions = {}
    for idx in hessian_influences:
        hessian_predictions[idx] = {
            'influence': hessian_influences[idx],
            'prediction': 'detrimental' if hessian_influences[idx] > 0 else 'beneficial'
        }
    
    with open(os.path.join(output_dir, 'hessian_predictions.json'), 'w') as f:
        json.dump(hessian_predictions, f, indent=2)
    
    # STEP 3: Evaluate Hessian Predictions
    print("\n" + "="*70)
    print("EVALUATING HESSIAN PREDICTIONS")
    print("="*70)
    
    tp = fp = tn = fn = 0
    for idx in ground_truth:
        true_label = ground_truth[idx]['label']
        pred_label = hessian_predictions[idx]['prediction']
        
        if true_label == 'detrimental' and pred_label == 'detrimental':
            tp += 1
        elif true_label == 'beneficial' and pred_label == 'detrimental':
            fp += 1
        elif true_label == 'beneficial' and pred_label == 'beneficial':
            tn += 1
        else:
            fn += 1
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Correlation
    influences = [hessian_predictions[idx]['influence'] for idx in sorted(ground_truth.keys())]
    delta_losses = [ground_truth[idx]['delta_loss'] for idx in sorted(ground_truth.keys())]
    correlation = np.corrcoef(influences, delta_losses)[0, 1] if len(influences) > 1 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Detrimental  Predicted Beneficial")
    print(f"True Detrimental         {tp:6d}              {fn:6d}")
    print(f"True Beneficial          {fp:6d}              {tn:6d}")
    
    print(f"\nHessian Influence Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    # STEP 4: Create Visualizations
    create_visualizations(ground_truth, hessian_predictions, output_dir)
    
    # STEP 5: Test Collective Removal
    collective_results = validator.test_collective_removal(
        ground_truth, eval_dataloader, baseline_loss, baseline_acc
    )
    
    # Save all results
    final_results = {
        'timestamp': timestamp,
        'baseline': {
            'loss': float(baseline_loss),
            'accuracy': float(baseline_acc)
        },
        'ground_truth_stats': {
            'total': len(ground_truth),
            'detrimental': det_count,
            'beneficial': len(ground_truth) - det_count
        },
        'hessian_performance': {
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'correlation': float(correlation)
        },
        'collective_removal': collective_results
    }
    
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print(f"\nKey Results Summary:")
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")
    print(f"  Hessian Prediction Accuracy: {accuracy:.4f}")
    print(f"  Collective Removal ΔAccuracy: {collective_results['delta_accuracy']:.4f}")
    print(f"  Visualization: {output_dir}/hessian_analysis_plots.png")

if __name__ == "__main__":
    main()

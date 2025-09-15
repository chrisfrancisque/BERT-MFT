"""
Phase 2: Ground Truth Validation with Enhanced Metrics and Visualization
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

class Phase2GroundTruthValidator:
    """
    Enhanced validator with accuracy tracking and visualization
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {
            'ground_truth': {},
            'predictions': {},
            'metrics': {},
            'accuracy_tracking': []
        }
        
    def get_all_parameter_indices(self):
        """Get flattened indices of all parameters (excluding critical layers)"""
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
        """Randomly sample n parameters from all available"""
        print(f"\nSampling {n_samples} parameters...")
        
        all_indices = self.get_all_parameter_indices()
        print(f"Total available parameters: {len(all_indices):,}")
        
        # Random sampling
        if len(all_indices) <= n_samples:
            sampled = all_indices
        else:
            sampled = random.sample(all_indices, n_samples)
        
        print(f"Sampled {len(sampled)} parameters")
        return sampled
    
    def compute_baseline_metrics(self, dataloader):
        """Compute baseline validation loss and accuracy"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing baseline metrics"):
                outputs = self.model(**batch)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Calculate accuracy
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
                    
        xm.mark_step()
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        print(f"Baseline loss: {avg_loss:.6f}")
        print(f"Baseline accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
        
        return avg_loss, accuracy
    
    def compute_metrics_after_zeroing(self, dataloader):
        """Compute validation loss and accuracy after zeroing a parameter"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
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
    
    def compute_ground_truth(self, sampled_params, dataloader, baseline_loss, baseline_acc):
        """
        Ground truth with enhanced tracking
        """
        print("\n" + "="*70)
        print("COMPUTING GROUND TRUTH")
        print("="*70)
        
        ground_truth = {}
        accuracy_changes = []
        
        # Save original model state
        original_state = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        for i, param_info in enumerate(tqdm(sampled_params, desc="Ground truth collection")):
            param_name = param_info['param_name']
            local_idx = param_info['local_idx']
            
            # Zero the specific parameter
            param = dict(self.model.named_parameters())[param_name]
            original_value = param.data.flatten()[local_idx].item()
            
            with torch.no_grad():
                param.data.flatten()[local_idx] = 0
            xm.mark_step()
            
            # Compute new metrics
            zeroed_loss, zeroed_acc = self.compute_metrics_after_zeroing(dataloader)
            
            # Calculate changes
            delta_loss = zeroed_loss - baseline_loss
            delta_acc = zeroed_acc - baseline_acc
            
            # Store result
            ground_truth[param_info['global_idx']] = {
                'param_name': param_name,
                'local_idx': local_idx,
                'original_value': original_value,
                'baseline_loss': baseline_loss,
                'zeroed_loss': zeroed_loss,
                'delta_loss': delta_loss,
                'baseline_accuracy': baseline_acc,
                'zeroed_accuracy': zeroed_acc,
                'delta_accuracy': delta_acc,
                'label': 'detrimental' if delta_loss < 0 else 'beneficial'
            }
            
            accuracy_changes.append(delta_acc)
            
            # Restore parameter
            with torch.no_grad():
                param.data.flatten()[local_idx] = original_value
            xm.mark_step()
            
            # Progress update every 100 parameters (or every parameter if less than 100)
            update_freq = min(100, max(1, len(sampled_params) // 10))
            if (i + 1) % update_freq == 0:
                detrimental_count = sum(1 for v in ground_truth.values() if v['label'] == 'detrimental')
                print(f"\nProgress: {i+1}/{len(sampled_params)}")
                print(f"  Detrimental so far: {detrimental_count}/{i+1} ({100*detrimental_count/(i+1):.1f}%)")
                print(f"  Latest ΔL: {delta_loss:.6f}, ΔAcc: {delta_acc:.4f}")
                print(f"  Avg ΔAcc so far: {np.mean(accuracy_changes):.6f}")
        
        # Final statistics
        detrimental = sum(1 for v in ground_truth.values() if v['label'] == 'detrimental')
        beneficial = len(ground_truth) - detrimental
        
        print(f"\nGround Truth Complete:")
        print(f"  Detrimental (ΔL < 0): {detrimental} ({100*detrimental/len(ground_truth):.1f}%)")
        print(f"  Beneficial (ΔL > 0): {beneficial} ({100*beneficial/len(ground_truth):.1f}%)")
        print(f"  Mean ΔAccuracy: {np.mean(accuracy_changes):.6f}")
        print(f"  Std ΔAccuracy: {np.std(accuracy_changes):.6f}")
        
        return ground_truth
    
    def compute_influence_predictions(self, sampled_params, dataloader):
        """
        Compute influence predictions using simplified influence function
        """
        print("\n" + "="*70)
        print("COMPUTING INFLUENCE PREDICTIONS")
        print("="*70)
        
        # Compute validation gradient
        print("Computing validation gradient...")
        self.model.eval()
        self.model.zero_grad()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Computing gradient"):
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            if not torch.isnan(loss):
                loss.backward()
                total_loss += loss.item()
                num_batches += 1
                
        xm.mark_step()
        
        # Extract predictions for sampled parameters
        predictions = {}
        
        for param_info in sampled_params:
            param_name = param_info['param_name']
            local_idx = param_info['local_idx']
            global_idx = param_info['global_idx']
            
            param = dict(self.model.named_parameters())[param_name]
            
            if param.grad is not None:
                # Get gradient value
                grad_value = param.grad.flatten()[local_idx].item() / num_batches
                
                # Influence = gradient (simplified, assuming H ≈ I)
                influence = grad_value
                
                predictions[global_idx] = {
                    'param_name': param_name,
                    'local_idx': local_idx,
                    'influence': influence,
                    'prediction': 'detrimental' if influence > 0 else 'beneficial'
                }
            else:
                predictions[global_idx] = {
                    'param_name': param_name,
                    'local_idx': local_idx,
                    'influence': 0,
                    'prediction': 'beneficial'
                }
        
        print(f"Predictions complete for {len(predictions)} parameters")
        return predictions
    
    def evaluate_and_visualize(self, ground_truth, predictions, output_dir):
        """
        Enhanced evaluation with visualization
        """
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        # Build confusion matrix
        true_labels = []
        pred_labels = []
        
        for idx in ground_truth:
            true_labels.append(1 if ground_truth[idx]['label'] == 'detrimental' else 0)
            pred_labels.append(1 if predictions[idx]['prediction'] == 'detrimental' else 0)
        
        # Calculate metrics
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        # Calculate correlation
        influences = [predictions[idx]['influence'] for idx in sorted(ground_truth.keys())]
        delta_losses = [ground_truth[idx]['delta_loss'] for idx in sorted(ground_truth.keys())]
        
        if len(influences) > 1 and np.std(influences) > 0 and np.std(delta_losses) > 0:
            correlation = np.corrcoef(influences, delta_losses)[0, 1]
        else:
            correlation = 0.0
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Beneficial', 'Detrimental'],
                   yticklabels=['Beneficial', 'Detrimental'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Influence vs Delta Loss Scatter
        axes[0, 1].scatter(influences, delta_losses, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Influence Score')
        axes[0, 1].set_ylabel('Actual ΔLoss')
        axes[0, 1].set_title(f'Influence vs Actual Loss Change (r={correlation:.3f})')
        
        # 3. Distribution of Delta Losses
        axes[1, 0].hist(delta_losses, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', label='Zero change')
        axes[1, 0].set_xlabel('ΔLoss')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Loss Changes')
        axes[1, 0].legend()
        
        # 4. Accuracy Changes Distribution
        delta_accs = [ground_truth[idx]['delta_accuracy'] for idx in ground_truth.keys()]
        axes[1, 1].hist(delta_accs, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Zero change')
        axes[1, 1].set_xlabel('ΔAccuracy')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Accuracy Changes')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analysis_plots.png'), dpi=150)
        plt.close()
        
        # Print results
        print("\nConfusion Matrix:")
        print(f"                 Predicted Detrimental  Predicted Beneficial")
        print(f"True Detrimental         {tp:6d}              {fn:6d}")
        print(f"True Beneficial          {fp:6d}              {tn:6d}")
        
        print(f"\nMetrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        
        metrics = {
            'confusion_matrix': {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            },
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'correlation': float(correlation),
            'mean_delta_accuracy': float(np.mean(delta_accs)),
            'std_delta_accuracy': float(np.std(delta_accs))
        }
        
        return metrics

def main():
    print("\n" + "="*70)
    print("PHASE 2: GROUND TRUTH VALIDATION (ENHANCED)")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'phase2_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Load model
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    
    # Ensure config exists
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
    
    print(f"\nLoading model from {checkpoint_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    # Load data
    print("\nLoading datasets...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    # Wrap for TPU
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Initialize validator
    validator = Phase2GroundTruthValidator(model, device)
    
    # Sample parameters
    sampled_params = validator.sample_parameters(n_samples=10000)
    
    # Save sampled parameters
    with open(os.path.join(output_dir, 'sampled_params.json'), 'w') as f:
        json.dump(sampled_params, f)
    
    # Compute baseline metrics
    baseline_loss, baseline_accuracy = validator.compute_baseline_metrics(eval_dataloader)
    
    # Task 1: Ground Truth Collection
    ground_truth = validator.compute_ground_truth(
        sampled_params, eval_dataloader, baseline_loss, baseline_accuracy
    )
    
    # Save ground truth
    with open(os.path.join(output_dir, 'ground_truth.json'), 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    # Task 2: Influence Predictions
    predictions = validator.compute_influence_predictions(sampled_params, eval_dataloader)
    
    # Save predictions
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Task 3: Evaluation with Visualization
    metrics = validator.evaluate_and_visualize(ground_truth, predictions, output_dir)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save complete results
    complete_results = {
        'timestamp': timestamp,
        'model_checkpoint': checkpoint_path,
        'num_samples': len(sampled_params),
        'baseline_loss': float(baseline_loss),
        'baseline_accuracy': float(baseline_accuracy),
        'metrics': metrics,
        'summary': {
            'total_parameters_sampled': len(sampled_params),
            'ground_truth_detrimental': sum(1 for v in ground_truth.values() if v['label'] == 'detrimental'),
            'predicted_detrimental': sum(1 for v in predictions.values() if v['prediction'] == 'detrimental'),
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'correlation': metrics['correlation']
        }
    }
    
    with open(os.path.join(output_dir, 'complete_results.json'), 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print(f"\nKey Findings:")
    print(f"  Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"  Influence Prediction Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Correlation with actual ΔL: {metrics['correlation']:.4f}")
    print(f"  Mean ΔAccuracy from zeroing: {metrics['mean_delta_accuracy']:.6f}")
    print(f"\nVisualization saved as: {output_dir}/analysis_plots.png")

if __name__ == "__main__":
    main()

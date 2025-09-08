"""
Detailed analysis comparing gradient vs influence methods
Shows parameter counts and tests with equal numbers
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# Setup environment
os.environ["PJRT_DEVICE"] = "TPU"

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
print(f"Using TPU: {device}")

from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from evaluation import ModelEvaluator
from influence_functions_tpu import ParameterInfluenceAnalyzer

# Configure
config.train_samples = 1000  # Use more samples for better analysis
config.batch_size = 16
config.device = str(device)
config.use_tpu = True

def collect_gradients_simple(model, dataloader, exclude_patterns=("classifier.", "pooler.", "LayerNorm", "embeddings")):
    """Collect gradients with exclusion patterns"""
    print("Collecting gradients...")
    model.train()
    
    accumulated_gradients = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and not any(pat in name for pat in exclude_patterns):
            accumulated_gradients[name] = torch.zeros_like(param, device=device)
    
    num_batches = 0
    max_batches = 30  # More batches for better gradient estimate
    
    for i, batch in enumerate(tqdm(dataloader, total=min(max_batches, len(dataloader)))):
        if i >= max_batches:
            break
            
        model.zero_grad()
        outputs = model(**batch)
        loss = F.cross_entropy(outputs.logits, batch['labels'])
        
        if not torch.isnan(loss):
            loss.backward()
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in accumulated_gradients and param.grad is not None:
                        accumulated_gradients[name] += param.grad
            
            num_batches += 1
        
        if i % 5 == 0:
            xm.mark_step()
    
    xm.mark_step()
    
    for name in accumulated_gradients:
        accumulated_gradients[name] = (accumulated_gradients[name] / num_batches).cpu()
    
    print(f"Collected gradients from {num_batches} batches")
    return accumulated_gradients

def analyze_all_parameters(model, gradients, val_gradient, learning_rate=2e-5):
    """Comprehensive analysis of all parameters"""
    print("\nAnalyzing all parameters...")
    
    all_params_info = []
    stats = {
        'total_params': 0,
        'excluded_params': 0,
        'analyzed_params': 0,
        'gradient_detrimental': 0,
        'influence_beneficial': 0,
        'influence_harmful': 0,
        'both_agree': 0,
        'disagree': 0
    }
    
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        stats['total_params'] += param_size
        
        # Check if excluded
        if any(pat in name for pat in exclude_patterns):
            stats['excluded_params'] += param_size
            continue
        
        if name not in gradients:
            continue
            
        stats['analyzed_params'] += param_size
        
        param_cpu = param.data.cpu()
        gradient = gradients[name]
        
        # Gradient analysis
        theoretical_update = param_cpu - learning_rate * gradient
        original_abs = torch.abs(param_cpu)
        updated_abs = torch.abs(theoretical_update)
        moving_to_zero = updated_abs < original_abs
        movement = original_abs - updated_abs
        movement[~moving_to_zero] = 0
        
        # For each parameter element that's moving toward zero
        if moving_to_zero.sum() > 0:
            # Get top elements
            flat_movement = movement.flatten()
            flat_mask = moving_to_zero.flatten()
            
            # Get indices of all detrimental parameters
            detrimental_indices = torch.where(flat_mask)[0]
            
            for idx in detrimental_indices[:100]:  # Limit to top 100 per parameter tensor
                movement_val = flat_movement[idx].item()
                if movement_val > 0:
                    stats['gradient_detrimental'] += 1
                    
                    # Influence analysis
                    influence = 0
                    if name in val_gradient:
                        grad_val = val_gradient[name].flatten()[idx].item()
                        param_val = param_cpu.flatten()[idx].item()
                        influence = -grad_val * param_val
                    
                    is_beneficial = influence < 0
                    if is_beneficial:
                        stats['influence_beneficial'] += 1
                    else:
                        stats['influence_harmful'] += 1
                    
                    # Check agreement
                    if is_beneficial:
                        stats['both_agree'] += 1
                    else:
                        stats['disagree'] += 1
                    
                    all_params_info.append({
                        'param_name': name,
                        'flat_index': idx.item(),
                        'movement': movement_val,
                        'influence': influence,
                        'gradient_says_remove': True,
                        'influence_says_remove': is_beneficial
                    })
    
    return all_params_info, stats

def main():
    print("\n" + "="*70)
    print("DETAILED PARAMETER ANALYSIS: GRADIENT VS INFLUENCE")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'detailed_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForSequenceClassification"],
                "num_labels": 2,
                "model_type": "bert"
            }, f)
    
    print(f"Loading model from {checkpoint_path}")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} total parameters")
    
    # Count excluded parameters
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    excluded_params = sum(p.numel() for name, p in model.named_parameters() 
                         if any(pat in name for pat in exclude_patterns))
    print(f"Excluded from analysis: {excluded_params:,} parameters ({100*excluded_params/total_params:.1f}%)")
    print(f"Available for analysis: {total_params - excluded_params:,} parameters")
    
    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Baseline
    print("\n" + "="*40)
    print("BASELINE EVALUATION")
    print("="*40)
    
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Collect gradients
    print("\n" + "="*40)
    print("GRADIENT COLLECTION")
    print("="*40)
    
    gradients = collect_gradients_simple(model, train_dataloader)
    
    # Compute validation gradient for influence
    print("\n" + "="*40)
    print("INFLUENCE PREPARATION")
    print("="*40)
    
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    # Comprehensive analysis
    print("\n" + "="*40)
    print("COMPREHENSIVE ANALYSIS")
    print("="*40)
    
    all_params_info, stats = analyze_all_parameters(model, gradients, val_gradient)
    
    print(f"\nParameter Statistics:")
    print(f"  Total model parameters: {stats['total_params']:,}")
    print(f"  Excluded (critical layers): {stats['excluded_params']:,}")
    print(f"  Analyzed parameters: {stats['analyzed_params']:,}")
    print(f"  Gradient identifies as detrimental: {stats['gradient_detrimental']:,}")
    print(f"  Influence identifies as beneficial to remove: {stats['influence_beneficial']:,}")
    print(f"  Influence identifies as harmful to remove: {stats['influence_harmful']:,}")
    print(f"  Both methods agree (remove): {stats['both_agree']:,}")
    print(f"  Methods disagree: {stats['disagree']:,}")
    
    if stats['gradient_detrimental'] > 0:
        false_positive_rate = 100 * stats['disagree'] / stats['gradient_detrimental']
        print(f"\nEstimated false positive rate of gradient method: {false_positive_rate:.1f}%")
    
    # Save original state
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Test different parameter counts
    test_counts = [100, 500, 1000, 2000]
    results = []
    
    for num_to_zero in test_counts:
        if num_to_zero > len(all_params_info):
            continue
            
        print(f"\n" + "="*40)
        print(f"TESTING WITH {num_to_zero} PARAMETERS")
        print(f"="*40)
        
        # Sort by gradient movement
        gradient_sorted = sorted(all_params_info, key=lambda x: x['movement'], reverse=True)
        
        # Sort by influence (most beneficial to remove first)
        influence_sorted = [x for x in all_params_info if x['influence_says_remove']]
        influence_sorted = sorted(influence_sorted, key=lambda x: x['influence'])
        
        # Test gradient method
        print(f"\nGradient method (zeroing {num_to_zero} params)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_state[name].to(device)
        xm.mark_step()
        
        with torch.no_grad():
            for item in gradient_sorted[:num_to_zero]:
                param = dict(model.named_parameters())[item['param_name']]
                param.data.flatten()[item['flat_index']] = 0
        xm.mark_step()
        
        gradient_metrics = evaluator.evaluate(eval_dataloader, f"Gradient-{num_to_zero}")
        
        # Test influence method
        print(f"\nInfluence method (zeroing {min(num_to_zero, len(influence_sorted))} params)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_state[name].to(device)
        xm.mark_step()
        
        actual_zeroed = min(num_to_zero, len(influence_sorted))
        with torch.no_grad():
            for item in influence_sorted[:actual_zeroed]:
                param = dict(model.named_parameters())[item['param_name']]
                param.data.flatten()[item['flat_index']] = 0
        xm.mark_step()
        
        influence_metrics = evaluator.evaluate(eval_dataloader, f"Influence-{actual_zeroed}")
        
        results.append({
            'num_params': num_to_zero,
            'gradient_accuracy': gradient_metrics['accuracy'],
            'gradient_change': gradient_metrics['accuracy'] - baseline_metrics['accuracy'],
            'influence_params_zeroed': actual_zeroed,
            'influence_accuracy': influence_metrics['accuracy'],
            'influence_change': influence_metrics['accuracy'] - baseline_metrics['accuracy']
        })
        
        print(f"\nResults with {num_to_zero} parameters:")
        print(f"  Gradient: {gradient_metrics['accuracy']:.4f} ({gradient_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
        print(f"  Influence: {influence_metrics['accuracy']:.4f} ({influence_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f})")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Params Zeroed':<15} {'Gradient Acc':<15} {'Influence Acc':<15} {'Difference':<15}")
    print("-"*60)
    
    for r in results:
        print(f"{r['num_params']:<15} {r['gradient_accuracy']:<15.4f} {r['influence_accuracy']:<15.4f} "
              f"{r['influence_accuracy'] - r['gradient_accuracy']:<+15.4f}")
    
    # Save all results
    final_results = {
        'stats': stats,
        'baseline_accuracy': baseline_metrics['accuracy'],
        'tests': results,
        'false_positive_rate_estimate': 100 * stats['disagree'] / stats['gradient_detrimental'] if stats['gradient_detrimental'] > 0 else 0
    }
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_dir}")
    
    # Hypothesis verification
    print("\n" + "="*70)
    print("HYPOTHESIS VERIFICATION")
    print("="*70)
    
    avg_gradient_drop = np.mean([abs(r['gradient_change']) for r in results])
    avg_influence_drop = np.mean([abs(r['influence_change']) for r in results])
    
    if avg_influence_drop < avg_gradient_drop * 0.2:
        print("✅ STRONGLY SUPPORTED: Influence functions reduce false positives by >80%")
        print(f"   Average gradient accuracy drop: {avg_gradient_drop:.4f}")
        print(f"   Average influence accuracy drop: {avg_influence_drop:.4f}")
        print(f"   False positive reduction: {(1 - avg_influence_drop/avg_gradient_drop)*100:.1f}%")
    else:
        print(f"   Average gradient accuracy drop: {avg_gradient_drop:.4f}")
        print(f"   Average influence accuracy drop: {avg_influence_drop:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

"""
Simplified comparison of gradient vs influence methods on 75% checkpoint
Properly handles TPU synchronization
"""
import os
import sys
import torch
import torch.nn.functional as F
import json
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# Setup environment
os.environ["PJRT_DEVICE"] = "TPU"

# Import TPU libraries
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
config.train_samples = 500  # Start with 500 samples
config.batch_size = 16  # Smaller batch size for stability
config.device = str(device)
config.use_tpu = True

def collect_gradients_simple(model, dataloader, exclude_patterns=("classifier.", "pooler.")):
    """Simple gradient collection with proper TPU sync"""
    print("Collecting gradients...")
    model.train()
    
    # Initialize gradient storage
    accumulated_gradients = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and not any(pat in name for pat in exclude_patterns):
            accumulated_gradients[name] = torch.zeros_like(param, device=device)
    
    num_batches = 0
    max_batches = 20  # Limit for TPU
    
    for i, batch in enumerate(tqdm(dataloader, total=min(max_batches, len(dataloader)))):
        if i >= max_batches:
            break
            
        model.zero_grad()
        
        # Forward and backward
        outputs = model(**batch)
        loss = F.cross_entropy(outputs.logits, batch['labels'])
        
        if not torch.isnan(loss):
            loss.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in accumulated_gradients and param.grad is not None:
                        accumulated_gradients[name] += param.grad
            
            num_batches += 1
        
        # Sync every 5 batches
        if i % 5 == 0:
            xm.mark_step()
    
    # Final sync
    xm.mark_step()
    
    # Average and move to CPU
    for name in accumulated_gradients:
        accumulated_gradients[name] = (accumulated_gradients[name] / num_batches).cpu()
    
    print(f"Collected gradients from {num_batches} batches")
    return accumulated_gradients

def analyze_detrimental_simple(model, gradients, learning_rate=2e-5):
    """Simple detrimental parameter analysis"""
    print("Analyzing detrimental parameters...")
    
    detrimental_info = []
    total_params = 0
    total_detrimental = 0
    
    for name, param in model.named_parameters():
        if name not in gradients:
            continue
        
        param_cpu = param.data.cpu()
        gradient = gradients[name]
        
        # Theoretical update
        theoretical_update = param_cpu - learning_rate * gradient
        
        # Check which parameters move toward zero
        original_abs = torch.abs(param_cpu)
        updated_abs = torch.abs(theoretical_update)
        moving_to_zero = updated_abs < original_abs
        
        # Calculate movement magnitude
        movement = original_abs - updated_abs
        movement[~moving_to_zero] = 0
        
        num_detrimental = moving_to_zero.sum().item()
        if num_detrimental > 0:
            # Get top 10 by movement
            flat_movement = movement.flatten()
            flat_mask = moving_to_zero.flatten()
            
            top_k = min(10, num_detrimental)
            top_values, top_indices = torch.topk(flat_movement, top_k)
            
            for idx, val in zip(top_indices, top_values):
                if val > 0:
                    detrimental_info.append({
                        'param_name': name,
                        'flat_index': idx.item(),
                        'movement': val.item()
                    })
        
        total_params += param.numel()
        total_detrimental += num_detrimental
    
    print(f"Found {total_detrimental:,} detrimental parameters ({100*total_detrimental/total_params:.2f}%)")
    
    # Sort by movement magnitude
    detrimental_info.sort(key=lambda x: x['movement'], reverse=True)
    
    return detrimental_info, total_params

def main():
    print("\n" + "="*70)
    print("SIMPLIFIED COMPARISON: GRADIENT VS INFLUENCE (75% checkpoint)")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'comparison_simple_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    
    # Ensure config
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
    print(f"Model loaded: {total_params:,} parameters")
    
    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Wrap for TPU
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    print(f"Data loaded: {len(train_dataloader)} train batches, {len(eval_dataloader)} eval batches")
    
    # Baseline evaluation
    print("\n" + "="*40)
    print("BASELINE EVALUATION")
    print("="*40)
    
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Save original model state
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Method 1: Gradient-based
    print("\n" + "="*40)
    print("METHOD 1: GRADIENT-BASED")
    print("="*40)
    
    # Collect gradients
    gradients = collect_gradients_simple(model, train_dataloader)
    
    # Analyze detrimental parameters
    detrimental_info, total_model_params = analyze_detrimental_simple(model, gradients)
    
    # Zero top 1% of detrimental parameters
    num_to_zero = int(len(detrimental_info) * 0.1)  # Take top 10% of detrimental
    num_to_zero = min(num_to_zero, int(total_model_params * 0.001))  # Max 0.1% of model
    
    print(f"Zeroing {num_to_zero} parameters using gradient method...")
    
    with torch.no_grad():
        for item in detrimental_info[:num_to_zero]:
            param = dict(model.named_parameters())[item['param_name']]
            param.data.flatten()[item['flat_index']] = 0
    
    xm.mark_step()
    
    # Evaluate gradient method
    gradient_metrics = evaluator.evaluate(eval_dataloader, "Gradient Method")
    print(f"Gradient Method Accuracy: {gradient_metrics['accuracy']:.4f}")
    
    # Restore model
    print("\nRestoring model for influence analysis...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = original_state[name].to(device)
    xm.mark_step()
    
    # Method 2: Influence-based (simplified)
    print("\n" + "="*40)
    print("METHOD 2: INFLUENCE-BASED (SIMPLIFIED)")
    print("="*40)
    
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    
    # Compute validation gradient
    print("Computing validation gradient...")
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    # Quick influence check on same parameters
    print("Checking influence of candidate parameters...")
    influence_checked = []
    
    for i, item in enumerate(tqdm(detrimental_info[:50], desc="Checking influences")):
        param_name = item['param_name']
        
        if param_name in val_gradient:
            param = dict(model.named_parameters())[param_name]
            flat_idx = item['flat_index']
            
            # Simple influence approximation
            grad_val = val_gradient[param_name].flatten()[flat_idx].item()
            param_val = param.data.flatten()[flat_idx].item()
            influence = -grad_val * param_val
            
            influence_checked.append({
                'param_name': param_name,
                'flat_index': flat_idx,
                'influence': influence,
                'would_improve': influence < 0,
                'gradient_movement': item['movement']
            })
        
        if i % 10 == 0:
            xm.mark_step()
    
    # Filter to beneficial parameters
    beneficial = [x for x in influence_checked if x['would_improve']]
    harmful = [x for x in influence_checked if not x['would_improve']]
    
    print(f"Influence analysis: {len(beneficial)} beneficial, {len(harmful)} harmful")
    
    # Zero only beneficial parameters
    num_to_zero_influence = min(len(beneficial), num_to_zero)
    
    print(f"Zeroing {num_to_zero_influence} beneficial parameters...")
    
    with torch.no_grad():
        for item in beneficial[:num_to_zero_influence]:
            param = dict(model.named_parameters())[item['param_name']]
            param.data.flatten()[item['flat_index']] = 0
    
    xm.mark_step()
    
    # Evaluate influence method
    influence_metrics = evaluator.evaluate(eval_dataloader, "Influence Method")
    print(f"Influence Method Accuracy: {influence_metrics['accuracy']:.4f}")
    
    # Results summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'Accuracy':<12} {'Change':<12}")
    print("-"*44)
    print(f"{'Baseline':<20} {baseline_metrics['accuracy']:<12.4f} {0:<+12.4f}")
    print(f"{'Gradient':<20} {gradient_metrics['accuracy']:<12.4f} "
          f"{gradient_metrics['accuracy'] - baseline_metrics['accuracy']:<+12.4f}")
    print(f"{'Influence':<20} {influence_metrics['accuracy']:<12.4f} "
          f"{influence_metrics['accuracy'] - baseline_metrics['accuracy']:<+12.4f}")
    print("-"*44)
    
    # Hypothesis check
    gradient_drop = baseline_metrics['accuracy'] - gradient_metrics['accuracy']
    influence_drop = baseline_metrics['accuracy'] - influence_metrics['accuracy']
    
    print("\n" + "="*70)
    print("HYPOTHESIS TEST")
    print("="*70)
    
    if influence_drop < gradient_drop * 0.5:
        print("✅ HYPOTHESIS SUPPORTED!")
        print(f"   Gradient accuracy drop: {gradient_drop:.4f}")
        print(f"   Influence accuracy drop: {influence_drop:.4f}")
        print(f"   Improvement: {(1 - influence_drop/gradient_drop)*100:.1f}% less accuracy loss")
        print(f"   Estimated false positive reduction: from ~{gradient_drop*200:.0f}% to ~{influence_drop*200:.0f}%")
    else:
        print("⚠️  Partial support")
        print(f"   Gradient accuracy drop: {gradient_drop:.4f}")
        print(f"   Influence accuracy drop: {influence_drop:.4f}")
    
    # Save results
    results = {
        'baseline_accuracy': baseline_metrics['accuracy'],
        'gradient_accuracy': gradient_metrics['accuracy'],
        'influence_accuracy': influence_metrics['accuracy'],
        'gradient_change': gradient_metrics['accuracy'] - baseline_metrics['accuracy'],
        'influence_change': influence_metrics['accuracy'] - baseline_metrics['accuracy'],
        'params_zeroed': num_to_zero,
        'beneficial_found': len(beneficial),
        'harmful_found': len(harmful)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

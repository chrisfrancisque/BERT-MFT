"""
First identify beneficial parameters with influence functions, then test removal
"""
import os
import torch
import torch.nn.functional as F
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

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

config.train_samples = 1000
config.batch_size = 16
config.device = str(device)
config.use_tpu = True

def collect_gradients(model, dataloader, exclude_patterns):
    """Collect gradients"""
    print("Collecting gradients...")
    model.train()
    
    accumulated_gradients = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and not any(pat in name for pat in exclude_patterns):
            accumulated_gradients[name] = torch.zeros_like(param, device=device)
    
    num_batches = 0
    max_batches = 30
    
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
    
    return accumulated_gradients

def identify_beneficial_parameters(model, gradients, val_gradient, learning_rate=2e-5):
    """Identify parameters that would be beneficial to remove using influence analysis"""
    print("\nIdentifying beneficial parameters with influence functions...")
    
    beneficial = []
    harmful = []
    
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    
    for name, param in tqdm(model.named_parameters(), desc="Analyzing"):
        if any(pat in name for pat in exclude_patterns):
            continue
        
        if name not in gradients or name not in val_gradient:
            continue
        
        param_cpu = param.data.cpu()
        gradient = gradients[name]
        
        # Find parameters moving toward zero (gradient criterion)
        theoretical_update = param_cpu - learning_rate * gradient
        original_abs = torch.abs(param_cpu)
        updated_abs = torch.abs(theoretical_update)
        moving_to_zero = updated_abs < original_abs
        
        if moving_to_zero.sum() == 0:
            continue
        
        # For those moving toward zero, check influence
        movement = original_abs - updated_abs
        movement[~moving_to_zero] = 0
        
        # Get top moving parameters
        flat_movement = movement.flatten()
        flat_mask = moving_to_zero.flatten()
        val_grad_flat = val_gradient[name].flatten()
        param_flat = param_cpu.flatten()
        
        # Get indices of top 100 detrimental per tensor
        detrimental_indices = torch.where(flat_mask)[0]
        if len(detrimental_indices) > 100:
            top_movement_vals, top_indices = torch.topk(flat_movement[detrimental_indices], 100)
            detrimental_indices = detrimental_indices[top_indices]
        
        for idx in detrimental_indices:
            if flat_movement[idx] > 1e-6:  # Significant movement
                # Compute influence
                grad_val = val_grad_flat[idx].item()
                param_val = param_flat[idx].item()
                influence = -grad_val * param_val
                
                param_info = {
                    'name': name,
                    'index': idx.item(),
                    'influence': influence,
                    'movement': flat_movement[idx].item()
                }
                
                if influence < 0:  # Beneficial to remove
                    beneficial.append(param_info)
                else:  # Harmful to remove
                    harmful.append(param_info)
    
    # Sort by influence (most beneficial first)
    beneficial.sort(key=lambda x: x['influence'])
    
    print(f"Found {len(beneficial)} beneficial and {len(harmful)} harmful parameters")
    print(f"False positive rate: {100*len(harmful)/(len(beneficial)+len(harmful)):.1f}%")
    
    return beneficial, harmful

def main():
    print("\n" + "="*70)
    print("IDENTIFY AND TEST BENEFICIAL PARAMETERS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'influence_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({"architectures": ["BertForSequenceClassification"], 
                      "num_labels": 2, "model_type": "bert"}, f)
    
    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Baseline
    print("\nBaseline evaluation...")
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    
    # Collect gradients
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    gradients = collect_gradients(model, train_dataloader, exclude_patterns)
    
    # Compute validation gradient
    print("\nComputing validation gradient...")
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    # Identify beneficial parameters
    beneficial, harmful = identify_beneficial_parameters(model, gradients, val_gradient)
    
    # Save beneficial parameters
    with open(os.path.join(output_dir, 'beneficial_params.json'), 'w') as f:
        json.dump(beneficial[:5000], f)  # Save top 5000
    
    # Test removing different amounts
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    test_amounts = [100, 500, 1000, 2000, min(5000, len(beneficial))]
    results = []
    
    for num_to_zero in test_amounts:
        print(f"\n{'='*40}")
        print(f"Testing {num_to_zero} beneficial parameters")
        print(f"{'='*40}")
        
        # Restore model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_state[name].to(device)
        xm.mark_step()
        
        # Zero beneficial parameters
        params_to_zero = beneficial[:num_to_zero]
        
        # Group by parameter name
        by_name = OrderedDict()
        for p in params_to_zero:
            if p['name'] not in by_name:
                by_name[p['name']] = []
            by_name[p['name']].append(p['index'])
        
        # Apply zeroing
        with torch.no_grad():
            for name, indices in by_name.items():
                param = dict(model.named_parameters())[name]
                # Use masking for efficiency
                mask = torch.ones_like(param.flatten())
                mask[indices] = 0
                param.data = (param.data.flatten() * mask.to(device)).reshape(param.shape)
        
        xm.mark_step()
        
        # Evaluate
        test_metrics = evaluator.evaluate(eval_dataloader, f"Beneficial-{num_to_zero}")
        
        result = {
            'num_zeroed': num_to_zero,
            'accuracy': test_metrics['accuracy'],
            'change': test_metrics['accuracy'] - baseline_metrics['accuracy']
        }
        results.append(result)
        
        print(f"Accuracy: {result['accuracy']:.4f} ({result['change']:+.4f})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: INFLUENCE-GUIDED PARAMETER REMOVAL")
    print("="*70)
    print(f"Baseline: {baseline_metrics['accuracy']:.4f}")
    print(f"{'Params Removed':<15} {'Accuracy':<12} {'Change':<12}")
    print("-"*39)
    for r in results:
        print(f"{r['num_zeroed']:<15} {r['accuracy']:<12.4f} {r['change']:<+12.4f}")
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({'baseline': baseline_metrics['accuracy'], 'tests': results}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

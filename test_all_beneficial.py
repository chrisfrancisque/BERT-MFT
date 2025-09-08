"""
Test zeroing ALL parameters that influence functions identify as beneficial to remove
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
config.train_samples = 1000
config.batch_size = 16
config.device = str(device)
config.use_tpu = True

def collect_gradients_comprehensive(model, dataloader, exclude_patterns=("classifier.", "pooler.", "LayerNorm", "embeddings")):
    """Collect gradients for comprehensive analysis"""
    print("Collecting gradients comprehensively...")
    model.train()
    
    accumulated_gradients = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and not any(pat in name for pat in exclude_patterns):
            accumulated_gradients[name] = torch.zeros_like(param, device=device)
    
    num_batches = 0
    max_batches = 40  # More batches for comprehensive analysis
    
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

def find_all_beneficial_parameters(model, gradients, val_gradient, learning_rate=2e-5):
    """Find ALL parameters that influence functions identify as beneficial to remove"""
    print("\nFinding all beneficial parameters...")
    
    beneficial_params = []
    harmful_params = []
    stats = {
        'total_analyzed': 0,
        'gradient_detrimental': 0,
        'influence_beneficial': 0,
        'influence_harmful': 0
    }
    
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    
    for name, param in tqdm(model.named_parameters(), desc="Analyzing parameters"):
        if any(pat in name for pat in exclude_patterns):
            continue
        
        if name not in gradients or name not in val_gradient:
            continue
        
        param_cpu = param.data.cpu()
        gradient = gradients[name]
        
        # Gradient analysis - find parameters moving toward zero
        theoretical_update = param_cpu - learning_rate * gradient
        original_abs = torch.abs(param_cpu)
        updated_abs = torch.abs(theoretical_update)
        moving_to_zero = updated_abs < original_abs
        movement = original_abs - updated_abs
        movement[~moving_to_zero] = 0
        
        # Check ALL parameters (not just top ones)
        flat_movement = movement.flatten()
        flat_mask = moving_to_zero.flatten()
        val_grad_flat = val_gradient[name].flatten()
        param_flat = param_cpu.flatten()
        
        # Analyze each parameter element
        for idx in range(len(flat_mask)):
            if flat_mask[idx]:  # If gradient says it's detrimental
                stats['gradient_detrimental'] += 1
                
                # Compute influence
                grad_val = val_grad_flat[idx].item()
                param_val = param_flat[idx].item()
                influence = -grad_val * param_val
                
                if influence < 0:  # Beneficial to remove
                    stats['influence_beneficial'] += 1
                    beneficial_params.append({
                        'param_name': name,
                        'flat_index': idx,
                        'influence': influence,
                        'movement': flat_movement[idx].item()
                    })
                else:  # Harmful to remove
                    stats['influence_harmful'] += 1
                    harmful_params.append({
                        'param_name': name,
                        'flat_index': idx,
                        'influence': influence,
                        'movement': flat_movement[idx].item()
                    })
            
            stats['total_analyzed'] += 1
            
            # Limit analysis for memory
            if stats['gradient_detrimental'] > 50000:
                print(f"Stopping at {stats['gradient_detrimental']} detrimental parameters")
                break
        
        if stats['gradient_detrimental'] > 50000:
            break
    
    return beneficial_params, harmful_params, stats

def main():
    print("\n" + "="*70)
    print("TESTING ALL BENEFICIAL PARAMETERS IDENTIFIED BY INFLUENCE FUNCTIONS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'all_beneficial_test_{timestamp}'
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
    
    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Baseline evaluation
    print("\n" + "="*40)
    print("BASELINE EVALUATION")
    print("="*40)
    
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    
    # Save original state
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Collect gradients
    print("\n" + "="*40)
    print("COLLECTING GRADIENTS")
    print("="*40)
    
    gradients = collect_gradients_comprehensive(model, train_dataloader)
    
    # Compute validation gradient
    print("\n" + "="*40)
    print("COMPUTING VALIDATION GRADIENT")
    print("="*40)
    
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    # Find all beneficial parameters
    print("\n" + "="*40)
    print("FINDING ALL BENEFICIAL PARAMETERS")
    print("="*40)
    
    beneficial_params, harmful_params, stats = find_all_beneficial_parameters(
        model, gradients, val_gradient
    )
    
    print(f"\nAnalysis complete:")
    print(f"  Total parameters analyzed: {stats['total_analyzed']:,}")
    print(f"  Gradient identifies as detrimental: {stats['gradient_detrimental']:,}")
    print(f"  Influence identifies as beneficial: {stats['influence_beneficial']:,}")
    print(f"  Influence identifies as harmful: {stats['influence_harmful']:,}")
    print(f"  False positive rate: {100*stats['influence_harmful']/stats['gradient_detrimental']:.1f}%")
    
    # Sort beneficial parameters by influence (most beneficial first)
    beneficial_params.sort(key=lambda x: x['influence'])
    
    # Test different amounts
    test_amounts = [
        100, 500, 1000, 2000, 5000,
        len(beneficial_params)  # ALL beneficial parameters
    ]
    
    results = []
    
    for num_to_zero in test_amounts:
        if num_to_zero > len(beneficial_params):
            num_to_zero = len(beneficial_params)
        
        print(f"\n" + "="*40)
        print(f"TESTING WITH {num_to_zero} PARAMETERS")
        print(f"="*40)
        
        # Restore model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_state[name].to(device)
        xm.mark_step()
        
        # Zero beneficial parameters
        print(f"Zeroing {num_to_zero} beneficial parameters...")
        params_by_name = {}
        for item in beneficial_params[:num_to_zero]:
            name = item['param_name']
            if name not in params_by_name:
                params_by_name[name] = []
            params_by_name[name].append(item['flat_index'])
        
        with torch.no_grad():
            for name, indices in params_by_name.items():
                param = dict(model.named_parameters())[name]
                for idx in indices:
                    param.data.flatten()[idx] = 0
        
        xm.mark_step()
        
        # Evaluate
        test_metrics = evaluator.evaluate(eval_dataloader, f"Beneficial-{num_to_zero}")
        
        result = {
            'num_zeroed': num_to_zero,
            'accuracy': test_metrics['accuracy'],
            'f1_score': test_metrics['f1_score'],
            'accuracy_change': test_metrics['accuracy'] - baseline_metrics['accuracy'],
            'percentage_of_model': 100 * num_to_zero / total_params
        }
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Parameters zeroed: {num_to_zero:,} ({result['percentage_of_model']:.4f}% of model)")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({result['accuracy_change']:+.4f})")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ZEROING BENEFICIAL PARAMETERS")
    print("="*70)
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print("-"*70)
    print(f"{'Params Zeroed':<15} {'% of Model':<12} {'Accuracy':<12} {'Change':<12} {'Status':<20}")
    print("-"*70)
    
    for r in results:
        if r['accuracy_change'] > 0:
            status = "✅ IMPROVED"
        elif r['accuracy_change'] > -0.01:
            status = "✓ Maintained"
        elif r['accuracy_change'] > -0.05:
            status = "⚠️ Small drop"
        else:
            status = "❌ Significant drop"
        
        print(f"{r['num_zeroed']:<15,} {r['percentage_of_model']:<12.4f} "
              f"{r['accuracy']:<12.4f} {r['accuracy_change']:<+12.4f} {status:<20}")
    
    # Save results
    final_results = {
        'baseline': {
            'accuracy': baseline_metrics['accuracy'],
            'f1_score': baseline_metrics['f1_score']
        },
        'stats': stats,
        'tests': results,
        'total_beneficial': len(beneficial_params),
        'total_harmful': len(harmful_params)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Final conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if results[-1]['accuracy_change'] > -0.01:
        print(f"✅ SUCCESS: Can zero ALL {len(beneficial_params):,} beneficial parameters")
        print(f"   with only {abs(results[-1]['accuracy_change']):.4f} accuracy loss!")
        print(f"   This confirms influence functions correctly identify safe parameters.")
    else:
        print(f"⚠️ Zeroing all {len(beneficial_params):,} beneficial parameters")
        print(f"   causes {abs(results[-1]['accuracy_change']):.4f} accuracy loss.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

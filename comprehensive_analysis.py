"""
Comprehensive analysis: Gradient method → Influence verification → Progressive zeroing
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

def comprehensive_parameter_analysis(model, train_dataloader, eval_dataloader, learning_rate=2e-5):
    """
    Complete pipeline: gradient analysis → influence verification → classification
    """
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    excluded_params = sum(p.numel() for name, p in model.named_parameters() 
                         if any(pat in name for pat in exclude_patterns))
    analyzed_params = total_params - excluded_params
    
    print("\n" + "="*70)
    print("PARAMETER STATISTICS")
    print("="*70)
    print(f"Total model parameters: {total_params:,}")
    print(f"Excluded from analysis: {excluded_params:,} ({100*excluded_params/total_params:.1f}%)")
    print(f"Parameters to analyze: {analyzed_params:,} ({100*analyzed_params/total_params:.1f}%)")
    
    # Step 1: Collect gradients
    print("\n" + "="*70)
    print("STEP 1: GRADIENT COLLECTION")
    print("="*70)
    
    model.train()
    accumulated_gradients = OrderedDict()
    
    for name, param in model.named_parameters():
        if param.requires_grad and not any(pat in name for pat in exclude_patterns):
            accumulated_gradients[name] = torch.zeros_like(param, device=device)
    
    num_batches = 0
    max_batches = 40  # More batches for better gradient estimate
    
    for i, batch in enumerate(tqdm(train_dataloader, total=min(max_batches, len(train_dataloader)))):
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
    
    # Average gradients
    for name in accumulated_gradients:
        accumulated_gradients[name] = (accumulated_gradients[name] / num_batches).cpu()
    
    # Step 2: Identify detrimental parameters (gradient criterion)
    print("\n" + "="*70)
    print("STEP 2: GRADIENT-BASED IDENTIFICATION")
    print("="*70)
    
    gradient_detrimental = []
    total_moving_to_zero = 0
    
    for name, param in model.named_parameters():
        if name not in accumulated_gradients:
            continue
        
        param_cpu = param.data.cpu()
        gradient = accumulated_gradients[name]
        
        # Parameters moving toward zero
        theoretical_update = param_cpu - learning_rate * gradient
        original_abs = torch.abs(param_cpu)
        updated_abs = torch.abs(theoretical_update)
        moving_to_zero = updated_abs < original_abs
        
        num_moving = moving_to_zero.sum().item()
        total_moving_to_zero += num_moving
        
        if num_moving > 0:
            movement = original_abs - updated_abs
            movement[~moving_to_zero] = 0
            
            # Get all parameters with significant movement
            flat_movement = movement.flatten()
            flat_mask = moving_to_zero.flatten()
            
            for idx in torch.where(flat_mask)[0]:
                if flat_movement[idx] > 1e-8:  # Very small threshold to catch more
                    gradient_detrimental.append({
                        'name': name,
                        'index': idx.item(),
                        'movement': flat_movement[idx].item()
                    })
    
    print(f"Parameters moving toward zero: {total_moving_to_zero:,} ({100*total_moving_to_zero/analyzed_params:.2f}% of analyzed)")
    print(f"Gradient method identified: {len(gradient_detrimental):,} candidates")
    
    # Step 3: Compute validation gradient for influence analysis
    print("\n" + "="*70)
    print("STEP 3: INFLUENCE FUNCTION ANALYSIS")
    print("="*70)
    
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    # Step 4: Classify parameters using influence
    influence_beneficial = []
    influence_harmful = []
    influence_neutral = []
    
    for item in tqdm(gradient_detrimental, desc="Computing influences"):
        name = item['name']
        idx = item['index']
        
        if name in val_gradient:
            param = dict(model.named_parameters())[name]
            param_flat = param.data.cpu().flatten()
            val_grad_flat = val_gradient[name].flatten()
            
            # Compute influence
            grad_val = val_grad_flat[idx].item()
            param_val = param_flat[idx].item()
            influence = -grad_val * param_val
            
            item['influence'] = influence
            
            if influence < -1e-8:  # Beneficial to remove
                influence_beneficial.append(item)
            elif influence > 1e-8:  # Harmful to remove
                influence_harmful.append(item)
            else:  # Neutral
                influence_neutral.append(item)
    
    # Sort beneficial by influence (most beneficial first)
    influence_beneficial.sort(key=lambda x: x['influence'])
    
    print(f"\nInfluence function classification of gradient-identified parameters:")
    print(f"  Beneficial to remove: {len(influence_beneficial):,} ({100*len(influence_beneficial)/len(gradient_detrimental):.1f}%)")
    print(f"  Harmful to remove: {len(influence_harmful):,} ({100*len(influence_harmful)/len(gradient_detrimental):.1f}%)")
    print(f"  Neutral: {len(influence_neutral):,} ({100*len(influence_neutral)/len(gradient_detrimental):.1f}%)")
    
    false_positive_rate = 100 * len(influence_harmful) / len(gradient_detrimental) if gradient_detrimental else 0
    print(f"\nGradient method false positive rate: {false_positive_rate:.1f}%")
    
    return {
        'total_params': total_params,
        'excluded_params': excluded_params,
        'analyzed_params': analyzed_params,
        'gradient_detrimental': gradient_detrimental,
        'influence_beneficial': influence_beneficial,
        'influence_harmful': influence_harmful,
        'influence_neutral': influence_neutral
    }

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE GRADIENT + INFLUENCE ANALYSIS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'comprehensive_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({"architectures": ["BertForSequenceClassification"], 
                      "num_labels": 2, "model_type": "bert"}, f)
    
    print(f"Loading model from {checkpoint_path}")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    # Load data
    print("Loading datasets...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Baseline evaluation
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # Run comprehensive analysis
    analysis = comprehensive_parameter_analysis(model, train_dataloader, eval_dataloader)
    
    # Save analysis
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as f:
        # Convert to serializable format
        save_data = {
            'stats': {
                'total_params': analysis['total_params'],
                'excluded_params': analysis['excluded_params'],
                'analyzed_params': analysis['analyzed_params'],
                'gradient_identified': len(analysis['gradient_detrimental']),
                'influence_beneficial': len(analysis['influence_beneficial']),
                'influence_harmful': len(analysis['influence_harmful']),
                'influence_neutral': len(analysis['influence_neutral'])
            }
        }
        json.dump(save_data, f, indent=2)
    
    # Save original state
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Test progressive zeroing of beneficial parameters
    print("\n" + "="*70)
    print("PROGRESSIVE PARAMETER REMOVAL TEST")
    print("="*70)
    
    beneficial = analysis['influence_beneficial']
    
    if len(beneficial) == 0:
        print("No beneficial parameters found! Analysis may be too restrictive.")
        return
    
    # Test amounts based on available parameters
    test_amounts = []
    for amt in [100, 500, 1000, 2000, 5000, 10000]:
        if amt <= len(beneficial):
            test_amounts.append(amt)
    test_amounts.append(len(beneficial))  # All beneficial
    
    results = []
    
    for num_to_zero in test_amounts:
        print(f"\nTesting with {num_to_zero:,} parameters removed...")
        
        # Restore model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_state[name].to(device)
        xm.mark_step()
        
        # Group parameters by name for efficient zeroing
        params_to_zero = beneficial[:num_to_zero]
        by_name = OrderedDict()
        for p in params_to_zero:
            if p['name'] not in by_name:
                by_name[p['name']] = []
            by_name[p['name']].append(p['index'])
        
        # Apply masking
        with torch.no_grad():
            for name, indices in by_name.items():
                param = dict(model.named_parameters())[name]
                mask = torch.ones_like(param.flatten())
                mask[indices] = 0
                param.data = (param.data.flatten() * mask.to(device)).reshape(param.shape)
        
        xm.mark_step()
        
        # Evaluate
        test_metrics = evaluator.evaluate(eval_dataloader, f"Zero-{num_to_zero}")
        
        result = {
            'num_zeroed': num_to_zero,
            'percentage_of_model': 100 * num_to_zero / analysis['total_params'],
            'percentage_of_analyzed': 100 * num_to_zero / analysis['analyzed_params'],
            'accuracy': test_metrics['accuracy'],
            'change': test_metrics['accuracy'] - baseline_metrics['accuracy']
        }
        results.append(result)
        
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['change']:+.4f})")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Model: 75% accuracy checkpoint")
    print(f"Total parameters: {analysis['total_params']:,}")
    print(f"Parameters analyzed: {analysis['analyzed_params']:,}")
    print(f"Gradient identified as detrimental: {len(analysis['gradient_detrimental']):,}")
    print(f"Influence confirmed as beneficial: {len(analysis['influence_beneficial']):,}")
    print(f"False positive rate: {100*len(analysis['influence_harmful'])/len(analysis['gradient_detrimental']):.1f}%")
    
    print("\n" + "-"*70)
    print(f"{'Params Removed':<15} {'% of Model':<12} {'% of Analyzed':<15} {'Accuracy':<12} {'Change':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['num_zeroed']:<15,} {r['percentage_of_model']:<12.3f} "
              f"{r['percentage_of_analyzed']:<15.3f} {r['accuracy']:<12.4f} {r['change']:<+12.4f}")
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({'baseline': baseline_metrics['accuracy'], 'tests': results}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

"""
Analyze multiple checkpoints to test the hypothesis:
Higher accuracy models have more parameters that can be masked
"""
import os
import torch
import json
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Force CPU usage
device = torch.device('cpu')
print(f"Using device: {device}")

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
from evaluation import ModelEvaluator

def analyze_checkpoint(checkpoint_path, name, train_dataloader, eval_dataloader):
    """Analyze a single checkpoint"""
    
    print(f"\\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    # Ensure config.json exists
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        bert_config = {
            "_name_or_path": "bert-base-uncased",
            "architectures": ["BertForSequenceClassification"],
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "vocab_size": 30522,
            "num_labels": 2,
            "model_type": "bert"
        }
        with open(config_path, 'w') as f:
            json.dump(bert_config, f, indent=2)
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    
    # Load metadata
    metadata = {}
    for meta_file in ['metadata.json', 'checkpoint_info.json']:
        meta_path = os.path.join(checkpoint_path, meta_file)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata.update(json.load(f))
    
    # Baseline evaluation
    print("Evaluating baseline...")
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    
    # Collect gradients
    print("Collecting gradients...")
    analyzer = GradientAnalyzer(model, config)
    gradient_results = analyzer.collect_gradients(train_dataloader)
    
    # Analyze detrimental parameters
    print("Analyzing detrimental parameters...")
    analysis_results = analyzer.analyze_detrimental_parameters()
    
    # Zero parameters
    print("Applying MFT...")
    handler = DetrimentalParameterHandler(model, config)
    handler.save_original_state()
    
    zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
    zeroing_stats = handler.zero_parameters(zeroing_info)
    
    # Evaluate after MFT
    print("Evaluating after MFT...")
    mft_metrics = evaluator.evaluate(eval_dataloader, "After MFT")
    
    results = {
        'name': name,
        'path': checkpoint_path,
        'metadata': metadata,
        'baseline_accuracy': baseline_metrics['accuracy'],
        'mft_accuracy': mft_metrics['accuracy'],
        'accuracy_change': mft_metrics['accuracy'] - baseline_metrics['accuracy'],
        'detrimental_params': analysis_results['total_detrimental'],
        'detrimental_percentage': 100.0 * analysis_results['total_detrimental'] / analysis_results['total_params'],
        'params_zeroed': zeroing_info['num_zeroed'],
        'zeroed_percentage': zeroing_info['percentage_of_total']
    }
    
    print(f"\\nResults for {name}:")
    print(f"  Baseline → MFT: {results['baseline_accuracy']:.4f} → {results['mft_accuracy']:.4f} ({results['accuracy_change']:+.4f})")
    print(f"  Detrimental: {results['detrimental_percentage']:.2f}% | Zeroed: {results['zeroed_percentage']:.2f}%")
    
    return results

def main():
    # Configure
    config.device = 'cpu'
    config.train_samples = 1000
    config.batch_size = 32
    config.use_tpu = False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'mft_comparison_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load data once
    print("Loading dataset...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Select checkpoints to analyze
    checkpoints = [
        # Baseline
        ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct'),
        
        # FFT at different accuracy levels
        ('models/fft_checkpoints/checkpoint_60pct', 'fft_60pct'),
        ('models/fft_checkpoints/checkpoint_75pct', 'fft_75pct'),
        ('models/fft_checkpoints/checkpoint_85pct', 'fft_85pct'),
        ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct'),
        
        # LoRA checkpoints
        ('models/lora_checkpoints/checkpoint_epoch_1', 'lora_epoch1'),
        ('models/lora_checkpoints/checkpoint_epoch_3', 'lora_epoch3'),
    ]
    
    all_results = []
    
    for checkpoint_path, name in checkpoints:
        if os.path.exists(checkpoint_path):
            try:
                results = analyze_checkpoint(
                    checkpoint_path, 
                    name,
                    train_dataloader,
                    eval_dataloader
                )
                all_results.append(results)
                
                # Save after each checkpoint
                with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                continue
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Create summary
    print("\\n" + "="*70)
    print("MFT HYPOTHESIS TEST RESULTS")
    print("="*70)
    print("Testing: Do higher accuracy models have more maskable parameters?")
    print("-"*70)
    print(f"{'Checkpoint':<20} {'Baseline':<10} {'After MFT':<10} {'Change':<10} {'Detrimental':<12} {'Zeroed':<10}")
    print("-"*70)
    
    for r in sorted(all_results, key=lambda x: x['baseline_accuracy']):
        print(f"{r['name']:<20} {r['baseline_accuracy']:<10.4f} {r['mft_accuracy']:<10.4f} "
              f"{r['accuracy_change']:<+10.4f} {r['detrimental_percentage']:<12.2f}% {r['zeroed_percentage']:<10.2f}%")
    
    # Save complete results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create CSV for analysis
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    print("-"*70)
    print(f"Results saved to: {output_dir}")
    
    # Analyze correlation
    if len(all_results) > 2:
        accuracies = [r['baseline_accuracy'] for r in all_results]
        detrimental_pcts = [r['detrimental_percentage'] for r in all_results]
        
        from scipy.stats import pearsonr
        correlation, p_value = pearsonr(accuracies, detrimental_pcts)
        
        print(f"\\nCorrelation Analysis:")
        print(f"  Accuracy vs Detrimental%: r={correlation:.3f}, p={p_value:.3f}")
        
        if correlation > 0:
            print("  → HYPOTHESIS SUPPORTED: Higher accuracy correlates with more detrimental parameters")
        else:
            print("  → HYPOTHESIS NOT SUPPORTED: No positive correlation found")

if __name__ == "__main__":
    main()

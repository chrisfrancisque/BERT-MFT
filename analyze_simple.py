"""
Simplified MFT analysis - CPU optimized
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

def quick_test():
    """Test MFT on just the baseline checkpoint"""
    
    # Configure for CPU
    config.device = 'cpu'
    config.train_samples = 1000
    config.batch_size = 32
    config.use_tpu = False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'mft_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Config: {config.train_samples} samples, batch_size={config.batch_size}")
    
    # Test with baseline checkpoint
    checkpoint_path = 'models/baseline/checkpoint_50pct_baseline'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\\n{'='*60}")
    print(f"Testing MFT on baseline checkpoint")
    print(f"{'='*60}")
    
    # Ensure config.json exists
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        print("Creating config.json...")
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
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except:
        print("Using default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load metadata
    metadata = {}
    metadata_path = os.path.join(checkpoint_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Baseline accuracy from metadata: {metadata.get('val_accuracy', 'N/A')}")
    
    # Load data
    print("Loading dataset...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Eval batches: {len(eval_dataloader)}")
    
    # Baseline evaluation
    print("\\nEvaluating baseline performance...")
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    
    # Collect gradients
    print("\\nCollecting gradients (this should take 2-5 minutes on CPU)...")
    analyzer = GradientAnalyzer(model, config)
    gradient_results = analyzer.collect_gradients(train_dataloader)
    
    # Analyze detrimental parameters
    print("\\nAnalyzing detrimental parameters...")
    analysis_results = analyzer.analyze_detrimental_parameters()
    
    # Zero parameters
    print("\\nApplying MFT...")
    handler = DetrimentalParameterHandler(model, config)
    handler.save_original_state()
    
    zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
    print(f"  Will zero {zeroing_info['num_zeroed']:,} parameters ({zeroing_info['percentage_of_total']:.2f}% of model)")
    
    zeroing_stats = handler.zero_parameters(zeroing_info)
    
    # Evaluate after MFT
    print("\\nEvaluating after MFT...")
    mft_metrics = evaluator.evaluate(eval_dataloader, "After MFT")
    
    # Results
    print(f"\\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"MFT Accuracy: {mft_metrics['accuracy']:.4f}")
    print(f"Change: {mft_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}")
    print(f"Detrimental params: {analysis_results['total_detrimental']:,} ({100.0 * analysis_results['total_detrimental'] / analysis_results['total_params']:.2f}%)")
    print(f"Params zeroed: {zeroing_info['num_zeroed']:,} ({zeroing_info['percentage_of_total']:.2f}%)")
    
    # Save results
    results = {
        'checkpoint': 'baseline_50pct',
        'baseline_metrics': baseline_metrics,
        'mft_metrics': mft_metrics,
        'improvement': mft_metrics['accuracy'] - baseline_metrics['accuracy'],
        'analysis': {
            'total_params': analysis_results['total_params'],
            'detrimental_params': analysis_results['total_detrimental'],
            'params_zeroed': zeroing_info['num_zeroed']
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to: {output_dir}")

if __name__ == "__main__":
    quick_test()

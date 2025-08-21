"""
Apply MFT analysis to FFT and LoRA checkpoints - TPU optimized version
"""
import os
import sys

# Set environment variables for TPU
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import json
import numpy as np
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Try to import TPU libraries
try:
    import torch_xla.core.xla_model as xm
    USE_TPU = True
    device = xm.xla_device()
    print(f"Using TPU: {device}")
except:
    USE_TPU = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# Import MFT modules
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
from evaluation import ModelEvaluator

def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint model"""
    print(f"\\nLoading checkpoint from: {checkpoint_path}")
    
    # Check what files exist
    files = os.listdir(checkpoint_path)
    print(f"  Files found: {files[:5]}...")  # Show first 5 files
    
    # Check if it's a LoRA checkpoint
    is_lora = 'adapter_model.bin' in files
    if is_lora:
        print("  Detected LoRA checkpoint")
    
    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Attempting to create config.json...")
        
        # Create config if missing
        config_path = os.path.join(checkpoint_path, 'config.json')
        if not os.path.exists(config_path):
            bert_config = {
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForSequenceClassification"],
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "num_labels": 2,
                "torch_dtype": "float32",
                "vocab_size": 30522
            }
            with open(config_path, 'w') as f:
                json.dump(bert_config, f, indent=2)
            print("  Created config.json")
        
        # Try loading again
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    except:
        print("  Using default bert-base-uncased tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model.to(device)
    
    # Load metadata
    metadata = {}
    for meta_file in ['metadata.json', 'checkpoint_info.json']:
        meta_path = os.path.join(checkpoint_path, meta_file)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata.update(json.load(f))
    
    if metadata:
        if 'accuracy' in metadata:
            print(f"  Loaded checkpoint with accuracy: {metadata['accuracy']:.4f}")
        elif 'val_accuracy' in metadata:
            print(f"  Loaded checkpoint with accuracy: {metadata['val_accuracy']:.4f}")
    
    return model, tokenizer, metadata

def analyze_checkpoint(checkpoint_path, checkpoint_name, output_dir, device):
    """Apply MFT analysis to a single checkpoint"""
    
    print(f"\\n{'='*60}")
    print(f"Analyzing: {checkpoint_name}")
    print(f"{'='*60}")
    
    try:
        # Load checkpoint
        model, tokenizer, metadata = load_checkpoint(checkpoint_path, device)
        
        # Load data
        print("Loading dataset...")
        train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
        train_dataloader, eval_dataloader = create_dataloaders(
            train_dataset, eval_dataset, config
        )
        
        # Baseline evaluation
        print("Evaluating baseline performance...")
        evaluator = ModelEvaluator(model, config)
        baseline_metrics = evaluator.evaluate(eval_dataloader, f"{checkpoint_name} Baseline")
        
        # Collect gradients
        print("Collecting gradients...")
        analyzer = GradientAnalyzer(model, config)
        gradient_results = analyzer.collect_gradients(train_dataloader)
        
        # Analyze detrimental parameters
        print("Analyzing detrimental parameters...")
        analysis_results = analyzer.analyze_detrimental_parameters()
        
        # Zero parameters
        print("Applying MFT (zeroing top detrimental parameters)...")
        handler = DetrimentalParameterHandler(model, config)
        handler.save_original_state()
        
        zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
        zeroing_stats = handler.zero_parameters(zeroing_info)
        
        # Evaluate after MFT
        print("Evaluating after MFT...")
        mft_metrics = evaluator.evaluate(eval_dataloader, f"{checkpoint_name} After MFT")
        
        # Mark step for TPU
        if USE_TPU:
            xm.mark_step()
        
        # Calculate improvements
        results = {
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': checkpoint_path,
            'checkpoint_metadata': metadata,
            'baseline_evaluation': baseline_metrics,
            'mft_evaluation': mft_metrics,
            'improvement': {
                'accuracy': mft_metrics['accuracy'] - baseline_metrics['accuracy'],
                'f1_score': mft_metrics['f1_score'] - baseline_metrics['f1_score']
            },
            'detrimental_analysis': {
                'total_params': analysis_results['total_params'],
                'total_detrimental': analysis_results['total_detrimental'],
                'percentage_detrimental': 100.0 * analysis_results['total_detrimental'] / analysis_results['total_params']
            },
            'zeroing_stats': {
                'num_zeroed': zeroing_info['num_zeroed'],
                'percentage_of_total': zeroing_info['percentage_of_total']
            }
        }
        
        # Save individual results
        result_path = os.path.join(output_dir, f'mft_{checkpoint_name.replace("/", "_")}.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\nResults for {checkpoint_name}:")
        print(f"  Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  MFT Accuracy: {mft_metrics['accuracy']:.4f}")
        print(f"  Improvement: {results['improvement']['accuracy']:+.4f}")
        print(f"  Detrimental params: {results['detrimental_analysis']['percentage_detrimental']:.2f}%")
        print(f"  Params zeroed: {results['zeroing_stats']['percentage_of_total']:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {checkpoint_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Analyze all checkpoints"""
    
    # Update config
    config.device = str(device)
    config.train_samples = 1000
    config.batch_size = 32
    config.use_tpu = USE_TPU
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'mft_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    all_results = []
    
    # Test with just a few checkpoints first
    test_checkpoints = [
        ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct'),
        ('models/fft_checkpoints/checkpoint_75pct', 'fft_75pct'),
        ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct'),
    ]
    
    for checkpoint_path, checkpoint_name in test_checkpoints:
        if os.path.exists(checkpoint_path):
            results = analyze_checkpoint(
                checkpoint_path,
                checkpoint_name,
                output_dir,
                device
            )
            if results:
                all_results.append(results)
    
    # Create summary
    if all_results:
        print("\\n" + "="*60)
        print("MFT ANALYSIS SUMMARY")
        print("="*60)
        
        for result in all_results:
            print(f"\\n{result['checkpoint_name']}:")
            print(f"  Baseline → MFT: {result['baseline_evaluation']['accuracy']:.4f} → {result['mft_evaluation']['accuracy']:.4f}")
            print(f"  Change: {result['improvement']['accuracy']:+.4f}")
        
        # Save summary
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\\n{'='*60}")
        print(f"Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

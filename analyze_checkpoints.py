"""
Apply MFT to different training checkpoints
"""
import sys
import os
import torch
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
from evaluation import ModelEvaluator
from data_utils import load_and_prepare_dataset, create_dataloaders
from load_checkpoint import load_checkpoint, list_checkpoints

def analyze_checkpoint(checkpoint_path, output_dir):
    """Apply MFT analysis to a single checkpoint"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load checkpoint
    model, tokenizer, checkpoint_info = load_checkpoint(checkpoint_path, device)
    epoch = checkpoint_info.get('epoch', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint from epoch {epoch}")
    print(f"{'='*60}")
    
    # Load data
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Apply MFT analysis
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, f"Epoch {epoch} Baseline")
    
    # Collect gradients
    analyzer = GradientAnalyzer(model, config)
    gradient_results = analyzer.collect_gradients(train_dataloader)
    
    # Analyze detrimental parameters
    analysis_results = analyzer.analyze_detrimental_parameters()
    
    # Zero parameters
    handler = DetrimentalParameterHandler(model, config)
    handler.save_original_state()
    
    zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
    zeroing_stats = handler.zero_parameters(zeroing_info)
    
    # Evaluate after MFT
    mft_metrics = evaluator.evaluate(eval_dataloader, f"Epoch {epoch} After MFT")
    
    # Save results
    results = {
        'checkpoint_epoch': epoch,
        'checkpoint_metrics': checkpoint_info.get('metrics', {}),
        'baseline_evaluation': baseline_metrics,
        'mft_evaluation': mft_metrics,
        'improvement': {
            'accuracy': mft_metrics['accuracy'] - baseline_metrics['accuracy'],
            'f1_score': mft_metrics['f1_score'] - baseline_metrics['f1_score']
        },
        'zeroing_stats': {
            'num_zeroed': zeroing_info['num_zeroed'],
            'percentage_of_total': zeroing_info['percentage_of_total']
        }
    }
    
    # Save to file
    result_path = os.path.join(output_dir, f'mft_analysis_epoch_{epoch}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_all_checkpoints(results_dir):
    """Apply MFT to all checkpoints from a training run"""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'mft_checkpoint_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all checkpoints
    checkpoints = list_checkpoints(results_dir)
    
    print(f"Found {len(checkpoints)} checkpoints to analyze")
    
    all_results = []
    for checkpoint in checkpoints:
        result = analyze_checkpoint(checkpoint['path'], output_dir)
        all_results.append(result)
    
    # Create summary
    summary = {
        'num_checkpoints': len(checkpoints),
        'results': all_results
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MFT CHECKPOINT ANALYSIS SUMMARY")
    print("="*60)
    
    for result in all_results:
        epoch = result['checkpoint_epoch']
        baseline_acc = result['baseline_evaluation']['accuracy']
        mft_acc = result['mft_evaluation']['accuracy']
        improvement = result['improvement']['accuracy']
        
        print(f"\nEpoch {epoch}:")
        print(f"  Baseline Accuracy: {baseline_acc:.4f}")
        print(f"  MFT Accuracy: {mft_acc:.4f}")
        print(f"  Improvement: {improvement:+.4f}")
    
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to full fine-tuning results directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Analyze specific checkpoint only')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Analyze single checkpoint
        output_dir = f'mft_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        analyze_checkpoint(args.checkpoint, output_dir)
    else:
        # Analyze all checkpoints
        analyze_all_checkpoints(args.results_dir)
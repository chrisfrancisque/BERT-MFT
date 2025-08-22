"""
Run MFT on existing checkpoints to test the hypothesis
"""
import os
import json
from datetime import datetime
from mft_fixed import run_mft_on_checkpoint
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders

def main():
    # Configure
    config.device = 'cpu'  # Use CPU to avoid TPU compilation issues
    config.train_samples = 1000  # Use subset for faster testing
    config.batch_size = 32
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'mft_paper_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print("="*70)
    print("TESTING PAPER'S MFT APPROACH WITH LEARNABLE SCORES")
    print("="*70)
    
    # Load data once
    print("\nLoading dataset...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # List of checkpoints to test
    checkpoints = [
        ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct'),
        ('models/fft_checkpoints/checkpoint_60pct', 'fft_60pct'),
        ('models/fft_checkpoints/checkpoint_75pct', 'fft_75pct'),
        ('models/fft_checkpoints/checkpoint_85pct', 'fft_85pct'),
        ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct'),
    ]
   
    
    all_results = []
    
    for checkpoint_path, name in checkpoints:
        if os.path.exists(checkpoint_path):
            # Ensure config.json exists
            config_path = os.path.join(checkpoint_path, 'config.json')
            if not os.path.exists(config_path):
                bert_config = {
                    "architectures": ["BertForSequenceClassification"],
                    "num_labels": 2,
                    "model_type": "bert"
                }
                with open(config_path, 'w') as f:
                    json.dump(bert_config, f)
            
            try:
                # Apply MFT to this checkpoint
                results, masked_model = run_mft_on_checkpoint(
                    checkpoint_path, 
                    train_dataloader, 
                    eval_dataloader,
                    config
                )
                
                all_results.append(results)
                
                # Save the improved model
                save_path = os.path.join(output_dir, f'{name}_mft')
                masked_model.save_pretrained(save_path)
                print(f"Saved MFT model to: {save_path}")
                
                # Save individual results
                with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("MFT EXPERIMENT RESULTS (Paper's Approach)")
    print("="*70)
    print(f"{'Checkpoint':<25} {'Original':<12} {'After MFT':<12} {'Change':<12}")
    print("-"*70)
    
    for r in all_results:
        name = os.path.basename(r['checkpoint'])
        print(f"{name:<25} {r['original_accuracy']:<12.4f} "
              f"{r['mft_accuracy']:<12.4f} {r['improvement']:<+12.4f}")
    
    print("-"*70)
    
    # Save all results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Compare with previous approach
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS APPROACH")
    print("="*70)
    print("Previous (gradient-based): All models collapsed to ~49%")
    print("New (learnable scores):    Expected improvements of 1-5%")

if __name__ == "__main__":
    main()

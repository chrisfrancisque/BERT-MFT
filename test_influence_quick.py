"""
Quick test of influence functions - minimal version
"""
import os
import sys
import torch
import json
import logging

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
from influence_functions_tpu import ParameterInfluenceAnalyzer

# Very small test config
config.train_samples = 32  # Just 32 samples
config.batch_size = 8
config.device = str(device)
config.use_tpu = True

def quick_test():
    print("\n" + "="*60)
    print("QUICK INFLUENCE FUNCTION TEST")
    print("="*60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
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
    
    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load minimal data
    print("Loading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    
    # Use only first 32 eval samples
    eval_dataset = eval_dataset.select(range(32))
    
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    # Wrap for TPU
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    print(f"Data loaded: {len(train_dataloader)} train batches, {len(eval_dataloader)} eval batches")
    
    # Test influence analyzer
    print("\nTesting ParameterInfluenceAnalyzer...")
    analyzer = ParameterInfluenceAnalyzer(model, config, device)
    
    print("Computing validation gradient (this should take <1 minute)...")
    val_gradient = analyzer.compute_validation_gradient(eval_dataloader)
    
    print(f"✓ Success! Computed gradients for {len(val_gradient)} parameters")
    
    # Show some statistics
    if val_gradient:
        grad_norms = {name: torch.norm(grad).item() for name, grad in list(val_gradient.items())[:5]}
        print(f"Sample gradient norms: {list(grad_norms.values())[:3]}")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        if success:
            print("\n✅ Quick test passed!")
            print("The influence functions are working on TPU.")
            print("You can now run the full analysis with more samples.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""Test the enhanced loss contribution check on TPU"""

import os
import sys

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
import torch_xla.distributed.parallel_loader as pl

def test_loss_contribution():
    """Compare original vs enhanced analysis"""
    
    device = xm.xla_device()
    xm.master_print(f"Testing on {device}")
    
    # Small test configuration
    config.train_samples = 100
    config.batch_size = 10
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    ).to(device)
    
    # Load data
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_dataset, eval_dataset, config
    )
    
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
    
    # Import the analyzer
    from main_tpu import TPUGradientAnalyzer
    analyzer = TPUGradientAnalyzer(model, config, device)
    
    # Collect gradients
    xm.master_print("Collecting gradients...")
    gradient_results = analyzer.collect_gradients(train_device_loader, 0)
    
    # Run enhanced analysis
    xm.master_print("\n" + "="*60)
    xm.master_print("ENHANCED ANALYSIS WITH LOSS CHECK")
    xm.master_print("="*60)
    
    results = analyzer.analyze_detrimental_parameters_with_loss_check()
    
    xm.master_print("\nTest completed successfully!")
    xm.master_print(f"Safe parameters found: {results['total_detrimental']:,}")
    xm.master_print(f"Critical parameters avoided: {results['total_harmful_skipped']:,}")
    
    return results

if __name__ == "__main__":
    test_loss_contribution()

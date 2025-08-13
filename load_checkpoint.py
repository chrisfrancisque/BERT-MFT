import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def list_checkpoints(results_dir):
    """List all available checkpoints in a results directory"""
    checkpoints = []
    
    # Find all checkpoint directories
    for item in os.listdir(results_dir):
        if item.startswith('checkpoint_epoch_'):
            checkpoint_path = os.path.join(results_dir, item)
            info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
            
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                checkpoints.append({
                    'path': checkpoint_path,
                    'epoch': info['epoch'],
                    'metrics': info['metrics']
                })
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints

def load_checkpoint(checkpoint_path, device='cpu'):
    """Load a specific checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load checkpoint info
    info_path = os.path.join(checkpoint_path, 'checkpoint_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            checkpoint_info = json.load(f)
    else:
        checkpoint_info = {}
    
    model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint_info.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint_info:
        metrics = checkpoint_info['metrics']
        print(f"  Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}")
        print(f"  Validation Loss: {metrics.get('val_loss', 0):.4f}")
    
    return model, tokenizer, checkpoint_info

def print_checkpoint_summary(results_dir):
    """Print a summary of all checkpoints"""
    checkpoints = list_checkpoints(results_dir)
    
    print("\n" + "="*60)
    print("AVAILABLE CHECKPOINTS")
    print("="*60)
    
    for cp in checkpoints:
        metrics = cp['metrics']
        print(f"\nEpoch {cp['epoch']}:")
        print(f"  Path: {cp['path']}")
        print(f"  Accuracy: {metrics.get('val_accuracy', 0):.4f}")
        print(f"  F1 Score: {metrics.get('val_f1', 0):.4f}")
        print(f"  Loss: {metrics.get('val_loss', 0):.4f}")
    
    print("="*60)
    
    return checkpoints
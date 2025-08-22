"""
Debug version to verify MFT is working
"""
import torch
from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
import json
import os

# Configure
config.device = 'cpu'
config.train_samples = 1000
config.batch_size = 32

# Load one checkpoint
checkpoint_path = 'models/fft_checkpoints/checkpoint_90pct'

# Ensure config
config_path = os.path.join(checkpoint_path, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump({"architectures": ["BertForSequenceClassification"], "num_labels": 2, "model_type": "bert"}, f)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.to('cpu')

# Count zeros before
zeros_before = sum((param.data == 0).sum().item() for _, param in model.named_parameters())
total_params = sum(param.numel() for _, param in model.named_parameters())
print(f"Before MFT: {zeros_before:,} zero parameters out of {total_params:,}")

# Load data
print("Loading data...")
train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)

# Collect gradients
print("Collecting gradients...")
analyzer = GradientAnalyzer(model, config)
gradient_results = analyzer.collect_gradients(train_dataloader)

# Analyze
print("Analyzing detrimental parameters...")
analysis_results = analyzer.analyze_detrimental_parameters()
print(f"Found {analysis_results['total_detrimental']:,} detrimental parameters")

# Apply MFT
print("Applying MFT...")
handler = DetrimentalParameterHandler(model, config)
handler.save_original_state()

zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
print(f"Will zero {zeroing_info['num_zeroed']:,} parameters")

# Check model before zeroing
param_sample_before = next(model.parameters()).data[0, :10].clone()
print(f"Sample params before: {param_sample_before}")

# Zero parameters
zeroing_stats = handler.zero_parameters(zeroing_info)

# Check model after zeroing
param_sample_after = next(model.parameters()).data[0, :10]
print(f"Sample params after: {param_sample_after}")
print(f"Did params change? {not torch.equal(param_sample_before, param_sample_after)}")

# Count zeros after
zeros_after = sum((param.data == 0).sum().item() for _, param in model.named_parameters())
print(f"After MFT: {zeros_after:,} zero parameters")
print(f"Difference: {zeros_after - zeros_before:,} parameters were zeroed")

# Verify the model object is the same
print(f"Model ID: {id(model)}")
print(f"First param ID: {id(next(model.parameters()))}")

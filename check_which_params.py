"""
Check WHICH parameters are being zeroed - maybe we're hitting critical ones
"""
import torch
from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
import json
import os
from collections import Counter

config.device = 'cpu'
config.train_samples = 1000
config.batch_size = 32

checkpoint_path = 'models/fft_checkpoints/checkpoint_90pct'
config_path = os.path.join(checkpoint_path, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump({"architectures": ["BertForSequenceClassification"], "num_labels": 2, "model_type": "bert"}, f)

print("Loading 90% accuracy model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Test the model works
print("\nTesting model BEFORE any modifications:")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_input = tokenizer("This is great!", return_tensors='pt')
with torch.no_grad():
    output = model(**test_input)
    print(f"Test output shape: {output.logits.shape}")
    print(f"Test prediction: {torch.argmax(output.logits, dim=-1).item()}")

# Load data and collect gradients
train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)

print("\nCollecting gradients...")
analyzer = GradientAnalyzer(model, config)
gradient_results = analyzer.collect_gradients(train_dataloader)
analysis_results = analyzer.analyze_detrimental_parameters()

# Get the parameters that would be zeroed
handler = DetrimentalParameterHandler(model, config)
handler.save_original_state()
zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)

# Analyze WHICH layers are affected
layer_counts = Counter()
param_types = Counter()

print(f"\nWill zero {zeroing_info['num_zeroed']:,} parameters")
print("\nAnalyzing which parameters will be zeroed:")

for item in zeroing_info['params_to_zero'][:100]:  # Check first 100
    param_name = item['param_name']
    
    # Count by layer
    if 'layer.' in param_name:
        layer_num = param_name.split('layer.')[1].split('.')[0]
        layer_counts[f'layer_{layer_num}'] += 1
    elif 'embeddings' in param_name:
        layer_counts['embeddings'] += 1
    elif 'pooler' in param_name:
        layer_counts['pooler'] += 1
    elif 'classifier' in param_name:
        layer_counts['classifier'] += 1
    
    # Count by parameter type
    if 'weight' in param_name:
        param_types['weight'] += 1
    elif 'bias' in param_name:
        param_types['bias'] += 1
    
    # Check for critical parameters
    if 'LayerNorm' in param_name:
        param_types['LayerNorm'] += 1
    if 'attention' in param_name:
        param_types['attention'] += 1

print("\nParameters by layer:")
for layer, count in layer_counts.most_common():
    print(f"  {layer}: {count}")

print("\nParameters by type:")
for ptype, count in param_types.most_common():
    print(f"  {ptype}: {count}")

# Check if classifier is being zeroed
classifier_params_zeroed = sum(1 for item in zeroing_info['params_to_zero'] 
                              if 'classifier' in item['param_name'])
print(f"\nClassifier parameters to be zeroed: {classifier_params_zeroed}")

if classifier_params_zeroed > 0:
    print("WARNING: Classifier head is being modified!")
    
# Show some specific parameters that will be zeroed
print("\nFirst 10 parameters to be zeroed:")
for i, item in enumerate(zeroing_info['params_to_zero'][:10]):
    print(f"  {i+1}. {item['param_name']} (movement: {item['movement']:.6f})")

"""
Trace exactly what happens to the model during MFT
"""
import torch
from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
from evaluation import ModelEvaluator
import json
import os

config.device = 'cpu'
config.train_samples = 1000
config.batch_size = 32

# Load the 90% model
checkpoint_path = 'models/fft_checkpoints/checkpoint_90pct'
config_path = os.path.join(checkpoint_path, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump({"architectures": ["BertForSequenceClassification"], 
                  "num_labels": 2, "model_type": "bert"}, f)

print("1. Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model_id_original = id(model)
print(f"   Model ID: {model_id_original}")

# Get a parameter reference
first_param_original = next(model.parameters())
first_param_data_original = first_param_original.data.clone()
print(f"   First param sum: {first_param_data_original.sum().item():.6f}")

# Load data
print("\n2. Loading data...")
train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)

# Evaluate BEFORE
print("\n3. Evaluating BEFORE MFT...")
evaluator = ModelEvaluator(model, config)
baseline_metrics = evaluator.evaluate(eval_dataloader, "Before")
print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")

# Collect gradients
print("\n4. Collecting gradients...")
analyzer = GradientAnalyzer(model, config)
gradient_results = analyzer.collect_gradients(train_dataloader)
analysis_results = analyzer.analyze_detrimental_parameters()

# Apply MFT
print("\n5. Applying MFT...")
handler = DetrimentalParameterHandler(model, config)
print(f"   Handler model ID: {id(handler.model)}")
print(f"   Same as original? {id(handler.model) == model_id_original}")

handler.save_original_state()
zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
print(f"   Will zero {zeroing_info['num_zeroed']:,} parameters")

# Count zeros before
zeros_before = sum((p == 0).sum().item() for p in model.parameters())
print(f"   Zeros before: {zeros_before:,}")

# Apply zeroing
zeroing_stats = handler.zero_parameters(zeroing_info)

# Check model after zeroing
print("\n6. After zeroing:")
print(f"   Model ID: {id(model)}")
print(f"   Same as original? {id(model) == model_id_original}")

first_param_after = next(model.parameters())
print(f"   First param is same object? {first_param_after is first_param_original}")
print(f"   First param sum: {first_param_after.data.sum().item():.6f}")
print(f"   First param changed? {not torch.equal(first_param_after.data, first_param_data_original)}")

zeros_after = sum((p == 0).sum().item() for p in model.parameters())
print(f"   Zeros after: {zeros_after:,}")
print(f"   New zeros: {zeros_after - zeros_before:,}")

# Check if handler still has original state
if hasattr(handler, 'original_state_dict'):
    print(f"   Handler has original_state_dict: {handler.original_state_dict is not None}")

# Evaluate AFTER
print("\n7. Evaluating AFTER MFT...")
evaluator2 = ModelEvaluator(model, config)  # Create new evaluator to be sure
after_metrics = evaluator2.evaluate(eval_dataloader, "After")
print(f"   Accuracy: {after_metrics['accuracy']:.4f}")
print(f"   Change: {after_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}")

# Double check the model
print("\n8. Double-checking model state:")
print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"   Model has {sum((p == 0).sum().item() for p in model.parameters()):,} zeros")

# Test with fresh evaluator on same model
print("\n9. Testing with completely fresh evaluation:")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_sentences = ["This is amazing!", "This is terrible!"]
model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        print(f"   '{sentence}' → Class {pred}")

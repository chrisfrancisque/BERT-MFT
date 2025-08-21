"""
Debug: Verify checkpoints are actually different trained models
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json

checkpoints = [
    ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct'),
    ('models/fft_checkpoints/checkpoint_75pct', 'fft_75pct'),
    ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct'),
]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
test_text = "This movie is absolutely fantastic and amazing!"

print("Testing if checkpoints are actually different models:")
print("="*60)

for checkpoint_path, name in checkpoints:
    if not os.path.exists(checkpoint_path):
        print(f"{name}: NOT FOUND")
        continue
    
    # Ensure config exists
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({"architectures": ["BertForSequenceClassification"], 
                      "num_labels": 2, "model_type": "bert"}, f)
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    
    # Test prediction
    inputs = tokenizer(test_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(outputs.logits, dim=-1).item()
    
    print(f"\n{name}:")
    print(f"  Prediction: Class {pred}")
    print(f"  Probabilities: [Neg: {probs[0]:.4f}, Pos: {probs[1]:.4f}]")
    
    # Check a specific parameter to see if they differ
    first_param = list(model.parameters())[0]
    print(f"  First param sum: {first_param.sum().item():.6f}")
    print(f"  First param mean: {first_param.mean().item():.6f}")

print("\n" + "="*60)
print("DEBUGGING MFT COLLAPSE:")
print("="*60)

# Now let's see what happens during MFT
from mft_working import MFTHandler
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders

config.device = 'cpu'
config.train_samples = 100  # Just a few samples for debugging
config.batch_size = 10

# Load a model
model = AutoModelForSequenceClassification.from_pretrained('models/fft_checkpoints/checkpoint_90pct')
model.to(config.device)

# Create MFT handler
mft = MFTHandler(model, config)

print(f"\nMFT Setup:")
print(f"  Parameters to mask: {len(mft.mask_scores)}")
print(f"  Target layers: {mft.target_layers}")

# Check what's being masked
params_to_mask = []
for name, param in model.named_parameters():
    if mft._should_mask(name):
        params_to_mask.append(name)

print(f"\nParameters selected for masking (first 5):")
for name in params_to_mask[:5]:
    print(f"  - {name}")

# Get initial masks
initial_masks = mft.get_masks(masking_ratio=0.9)
print(f"\nInitial mask statistics:")
for i, (name, mask) in enumerate(list(initial_masks.items())[:3]):
    kept = (mask == 1).sum().item()
    total = mask.numel()
    print(f"  {name}: keeping {kept}/{total} ({100*kept/total:.1f}%)")

# Test forward pass
print("\nTesting forward pass with masks...")
train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)

# Get one batch
for batch in eval_dataloader:
    batch = {k: v.to(config.device) for k, v in batch.items()}
    break

# Test predictions before and after masking
print("\nPredictions on batch:")
model.eval()

# Before masking
with torch.no_grad():
    outputs_original = model(**batch)
    preds_original = torch.argmax(outputs_original.logits, dim=-1)
    print(f"  Original predictions: {preds_original[:10].tolist()}")
    print(f"  Original accuracy: {(preds_original == batch['labels']).float().mean():.4f}")

# With masking
with torch.no_grad():
    outputs_masked = mft(**batch)
    preds_masked = torch.argmax(outputs_masked.logits, dim=-1)
    print(f"  Masked predictions: {preds_masked[:10].tolist()}")
    print(f"  Masked accuracy: {(preds_masked == batch['labels']).float().mean():.4f}")

# Check if all predictions are the same
if len(set(preds_masked.tolist())) == 1:
    print("\n⚠️ WARNING: Model is predicting the same class for everything!")
    print(f"  All predictions are class: {preds_masked[0].item()}")

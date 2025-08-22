"""
Verify that MFT is actually modifying the model
"""
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
import os
import json

# Test with one checkpoint
checkpoint_path = 'models/fft_checkpoints/checkpoint_90pct'

# Ensure config exists
config_path = os.path.join(checkpoint_path, 'config.json')
if not os.path.exists(config_path):
    bert_config = {
        "_name_or_path": "bert-base-uncased",
        "architectures": ["BertForSequenceClassification"],
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "num_labels": 2,
        "model_type": "bert"
    }
    with open(config_path, 'w') as f:
        json.dump(bert_config, f, indent=2)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Count original non-zero parameters
original_nonzero = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    original_nonzero += (param.data != 0).sum().item()

print(f"Original model:")
print(f"  Total parameters: {total_params:,}")
print(f"  Non-zero parameters: {original_nonzero:,}")
print(f"  Zero parameters: {total_params - original_nonzero:,}")

# Now simulate MFT - zero some random parameters as a test
print("\nManually zeroing 1% of parameters as a test...")
params_to_zero = int(total_params * 0.01)
zeroed_count = 0

with torch.no_grad():
    for name, param in model.named_parameters():
        if 'classifier' not in name and zeroed_count < params_to_zero:
            # Zero some random parameters
            mask = torch.rand_like(param) < 0.01
            param.data[mask] = 0.0
            zeroed_count += mask.sum().item()

# Count again
new_nonzero = 0
for name, param in model.named_parameters():
    new_nonzero += (param.data != 0).sum().item()

print(f"\nAfter manual zeroing:")
print(f"  Non-zero parameters: {new_nonzero:,}")
print(f"  Zero parameters: {total_params - new_nonzero:,}")
print(f"  Parameters zeroed: {original_nonzero - new_nonzero:,}")

# Test predictions before and after
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

test_sentences = [
    "This movie is great!",
    "This movie is terrible.",
    "I love this film.",
    "I hate this film."
]

print("\nTest predictions:")
model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        print(f"  '{sentence}' -> Class {pred} (probs: {probs.tolist()})")

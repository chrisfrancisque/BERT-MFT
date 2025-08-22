"""
Test ALL checkpoints to see which ones are actually trained
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Simple test sentences
test_sentences = [
    ("This is absolutely wonderful and amazing!", 1),
    ("This is terrible and awful!", 0),
    ("I love this!", 1),
    ("I hate this!", 0),
]

checkpoints = [
    ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct', 0.50),
    ('models/fft_checkpoints/checkpoint_55pct', 'fft_55pct', 0.55),
    ('models/fft_checkpoints/checkpoint_60pct', 'fft_60pct', 0.60),
    ('models/fft_checkpoints/checkpoint_65pct', 'fft_65pct', 0.65),
    ('models/fft_checkpoints/checkpoint_70pct', 'fft_70pct', 0.70),
    ('models/fft_checkpoints/checkpoint_75pct', 'fft_75pct', 0.75),
    ('models/fft_checkpoints/checkpoint_80pct', 'fft_80pct', 0.80),
    ('models/fft_checkpoints/checkpoint_85pct', 'fft_85pct', 0.85),
    ('models/fft_checkpoints/checkpoint_88pct', 'fft_88pct', 0.88),
    ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct', 0.90),
]

print("Testing all checkpoints...")
print(f"{'Checkpoint':<30} {'Expected':<12} {'Actual':<12} {'Status':<10}")
print("-" * 70)

for checkpoint_path, name, expected_acc in checkpoints:
    if not os.path.exists(checkpoint_path):
        print(f"{name:<30} {expected_acc:<12.1%} {'N/A':<12} MISSING")
        continue
    
    # Ensure config
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({"architectures": ["BertForSequenceClassification"], 
                      "num_labels": 2, "model_type": "bert"}, f)
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    
    correct = 0
    with torch.no_grad():
        for sentence, expected_label in test_sentences:
            inputs = tokenizer(sentence, return_tensors='pt')
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            correct += (pred == expected_label)
    
    actual_acc = correct / len(test_sentences)
    
    # Check if it's working as expected
    if abs(actual_acc - expected_acc) > 0.2:
        status = "❌ BROKEN"
    elif actual_acc < 0.6 and expected_acc > 0.6:
        status = "⚠️ SUSPECT"
    else:
        status = "✅ OK"
    
    print(f"{name:<30} {expected_acc:<12.1%} {actual_acc:<12.1%} {status}")

print("-" * 70)
print("\nConclusion: Checkpoints marked with ❌ or ⚠️ need to be retrained!")

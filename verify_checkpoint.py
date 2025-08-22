"""
Verify the checkpoint is actually trained and not reset
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json

# Test sentences that should have clear sentiment
test_sentences = [
    ("This movie is absolutely fantastic!", 1),  # Clearly positive
    ("This movie is terrible and boring.", 0),   # Clearly negative
    ("I love this so much!", 1),                 # Clearly positive
    ("I hate everything about this.", 0),        # Clearly negative
    ("Perfect in every way!", 1),                # Clearly positive
    ("Worst thing I've ever seen.", 0),          # Clearly negative
]

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Test different checkpoints
checkpoints = [
    ('models/baseline/checkpoint_50pct_baseline', 'baseline_50pct', 0.50),
    ('models/fft_checkpoints/checkpoint_60pct', 'fft_60pct', 0.60),
    ('models/fft_checkpoints/checkpoint_90pct', 'fft_90pct', 0.90),
]

for checkpoint_path, name, expected_acc in checkpoints:
    print(f"\n{'='*60}")
    print(f"Testing {name} (expected ~{expected_acc*100:.0f}% accuracy)")
    print(f"{'='*60}")
    
    # Ensure config exists
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({"architectures": ["BertForSequenceClassification"], 
                      "num_labels": 2, "model_type": "bert"}, f)
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    
    correct = 0
    for sentence, expected_label in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
            
            is_correct = pred == expected_label
            correct += is_correct
            
            print(f"  '{sentence[:30]}...' → {pred} (expected {expected_label}) {'✓' if is_correct else '✗'}")
            print(f"    Confidence: {probs[pred]:.3f}")
    
    accuracy = correct / len(test_sentences)
    print(f"\nManual test accuracy: {accuracy:.2%}")
    
    if accuracy < 0.5 and expected_acc > 0.7:
        print("⚠️ WARNING: This model is NOT performing as expected!")
        print("  The checkpoint may be corrupted or not properly trained!")

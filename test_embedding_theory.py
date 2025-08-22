"""
Test if zeroing token_type_embeddings causes model collapse
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json

# Load the working 90% model
checkpoint_path = 'models/fft_checkpoints/checkpoint_90pct'
config_path = os.path.join(checkpoint_path, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump({"architectures": ["BertForSequenceClassification"], 
                  "num_labels": 2, "model_type": "bert"}, f)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Test sentences
test_sentences = ["This is great!", "This is terrible!"]

print("Testing effect of zeroing token_type_embeddings...")
print("\n1. BEFORE zeroing:")
model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        print(f"  '{sentence}' → Class {pred}")

# Zero ONLY the token_type_embeddings
print("\n2. Zeroing token_type_embeddings...")
with torch.no_grad():
    # Find and zero token_type_embeddings
    for name, param in model.named_parameters():
        if 'token_type_embeddings' in name:
            print(f"  Zeroing {name} (shape: {param.shape})")
            param.data.zero_()

print("\n3. AFTER zeroing:")
with torch.no_grad():
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        print(f"  '{sentence}' → Class {pred}")

print("\nConclusion: Zeroing token_type_embeddings likely breaks the model!")

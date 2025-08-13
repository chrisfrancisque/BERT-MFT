import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_baseline_model(baseline_path= 'baseline_model_seed42', device = 'cpu'):
    """
    Load the warmed baseline model that's shared across all methods.
    
    Args:
        baseline_path: Path to the baseline model folder
        device: Device to load model to
    
    Returns:
        model: The warmed BERT model
        tokenizer: The tokenizer
        baseline_info: Dictionary with baseline training info
    """
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline model not found at {baseline_path}!\n"
            f"Please copy the baseline_model_seed42 folder to this repository"
        )
    
    print(f"Loading baseline model from {baseline_path}")

    model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
    tokenizer = AutoTokenizer.from_pretrained(baseline_path)

    baseline_info_path = os.path.join(baseline_path, 'baseline_info.json')
    if os.path.exists(baseline_info_path):
        with open(baseline_info_path, 'r') as f:
            baseline_info = json.load(f)
        

        print("\nBaseline Model Info:")
        print(f"  - Seed: {baseline_info.get('seed', 'Unknown')}")
        print(f"  - Warm-up accuracy: {baseline_info.get('warm_up_accuracy', 0):.4f}")
        print(f"  - Classifier trained for: 1 epoch")
        print(f"  - Total params: {baseline_info.get('total_params', 0):,}")
    else:
        baseline_info = {}
        print("Warning: baseline_info.json not found")
    
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("All parameters are unfrozen and ready for training")

    return model, tokenizer, baseline_info

if __name__ == "__main__":
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer, info = load_baseline_model(device = device)

    print("\n Baseline Model loaded successfully")
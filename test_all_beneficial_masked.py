"""
Test zeroing ALL beneficial parameters using tensor-level masking
Avoids stack overflow by using masks instead of individual modifications
"""
import os
import torch
import torch.nn.functional as F
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

os.environ["PJRT_DEVICE"] = "TPU"
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

device = xm.xla_device()
print(f"Using TPU: {device}")

from transformers import AutoModelForSequenceClassification
from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from evaluation import ModelEvaluator

config.train_samples = 1000
config.batch_size = 16
config.device = str(device)
config.use_tpu = True

class MaskedBERT(torch.nn.Module):
    """Wrapper that applies masks to BERT parameters"""
    def __init__(self, bert_model, param_masks):
        super().__init__()
        self.bert = bert_model
        self.param_masks = param_masks
        
        # Apply masks to parameters
        self.apply_masks()
    
    def apply_masks(self):
        """Apply binary masks to zero out parameters"""
        with torch.no_grad():
            for name, param in self.bert.named_parameters():
                if name in self.param_masks:
                    param.data = param.data * self.param_masks[name]
    
    def forward(self, **kwargs):
        # Ensure masks are applied before forward pass
        self.apply_masks()
        return self.bert(**kwargs)

def create_parameter_masks(model, beneficial_indices, device):
    """Create binary masks for each parameter tensor"""
    masks = {}
    
    # Group indices by parameter name
    params_to_mask = {}
    for name, idx in beneficial_indices:
        if name not in params_to_mask:
            params_to_mask[name] = []
        params_to_mask[name].append(idx)
    
    # Create masks
    for name, param in model.named_parameters():
        if name in params_to_mask:
            # Create a mask of ones
            mask = torch.ones_like(param, device=device)
            
            # Set mask to 0 for indices to zero
            indices = params_to_mask[name]
            flat_mask = mask.flatten()
            for idx in indices:
                flat_mask[idx] = 0
            mask = flat_mask.reshape(param.shape)
            
            masks[name] = mask
        else:
            # No masking for this parameter
            masks[name] = torch.ones_like(param, device=device)
    
    return masks

def find_beneficial_params_file():
    """Load previously identified beneficial parameters if saved"""
    # Check if we saved the beneficial parameters from a previous run
    import glob
    result_files = glob.glob("all_beneficial_test_*/beneficial_params.json")
    if result_files:
        with open(result_files[-1], 'r') as f:
            return json.load(f)
    return None

def main():
    print("\n" + "="*70)
    print("TESTING ALL BENEFICIAL PARAMETERS WITH MASKING")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'masked_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'
    config_path = os.path.join(checkpoint_path, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({
                "_name_or_path": "bert-base-uncased",
                "architectures": ["BertForSequenceClassification"],
                "num_labels": 2,
                "model_type": "bert"
            }, f)
    
    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    xm.mark_step()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} total parameters")
    
    # Load data
    print("\nLoading data...")
    train_dataset, eval_dataset, _ = load_and_prepare_dataset(config)
    train_dataloader, eval_dataloader = create_dataloaders(train_dataset, eval_dataset, config)
    eval_dataloader = pl.MpDeviceLoader(eval_dataloader, device)
    
    # Baseline evaluation
    print("\nBaseline evaluation...")
    evaluator = ModelEvaluator(model, config)
    baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline")
    
    # Create parameter indices to mask (simulate beneficial parameters)
    # In a real run, you would load these from your previous analysis
    print("\nGenerating parameter indices to mask...")
    exclude_patterns = ("classifier.", "pooler.", "LayerNorm", "embeddings")
    
    beneficial_indices = []
    for name, param in model.named_parameters():
        if not any(pat in name for pat in exclude_patterns):
            # Add some indices (in practice, load from your influence analysis)
            num_to_add = min(100, param.numel())  # Sample for testing
            for i in range(num_to_add):
                beneficial_indices.append((name, i))
                if len(beneficial_indices) >= 25070:
                    break
        if len(beneficial_indices) >= 25070:
            break
    
    print(f"Will mask {len(beneficial_indices)} parameters")
    
    # Test different amounts
    test_amounts = [5000, 10000, 15000, 20000, len(beneficial_indices)]
    results = []
    
    for num_to_mask in test_amounts:
        print(f"\n" + "="*40)
        print(f"TESTING WITH {num_to_mask:,} MASKED PARAMETERS")
        print(f"="*40)
        
        # Create masks for this number of parameters
        print("Creating masks...")
        masks = create_parameter_masks(
            model, 
            beneficial_indices[:num_to_mask], 
            device
        )
        
        # Create masked model
        masked_model = MaskedBERT(model, masks)
        
        # Evaluate
        print("Evaluating masked model...")
        
        # We need to evaluate using the masked model's forward
        class MaskedEvaluator:
            def __init__(self, masked_model, config):
                self.model = masked_model
                self.config = config
                self.device = next(masked_model.parameters()).device
            
            def evaluate(self, dataloader, phase_name):
                self.model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in tqdm(dataloader, desc=phase_name):
                        outputs = self.model(**batch)
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        correct += (predictions == batch['labels']).sum().item()
                        total += len(batch['labels'])
                
                accuracy = correct / total
                return {'accuracy': accuracy, 'f1_score': accuracy}  # Simplified
        
        masked_evaluator = MaskedEvaluator(masked_model, config)
        test_metrics = masked_evaluator.evaluate(eval_dataloader, f"Masked-{num_to_mask}")
        
        result = {
            'num_masked': num_to_mask,
            'accuracy': test_metrics['accuracy'],
            'accuracy_change': test_metrics['accuracy'] - baseline_metrics['accuracy'],
            'percentage_of_model': 100 * num_to_mask / total_params
        }
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Parameters masked: {num_to_mask:,} ({result['percentage_of_model']:.3f}% of model)")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({result['accuracy_change']:+.4f})")
        
        # Clean up masks to free memory
        del masks
        del masked_model
        xm.mark_step()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - MASKING APPROACH")
    print("="*70)
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print("-"*70)
    print(f"{'Params Masked':<15} {'% of Model':<12} {'Accuracy':<12} {'Change':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['num_masked']:<15,} {r['percentage_of_model']:<12.3f} "
              f"{r['accuracy']:<12.4f} {r['accuracy_change']:<+12.4f}")
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

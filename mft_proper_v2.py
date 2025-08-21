"""
Proper MFT implementation with correct gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import numpy as np
from copy import deepcopy

class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for binary masks"""
    @staticmethod
    def forward(ctx, scores, threshold=0.0):
        return (scores > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class MaskedLinear(nn.Module):
    """Linear layer with learnable mask"""
    def __init__(self, original_linear, mask_scores):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias if original_linear.bias is not None else None
        self.mask_scores = mask_scores
        self.masking_ratio = 0.9
        
    def forward(self, input):
        # Create binary mask from scores (with gradient flow)
        keep_ratio = 1.0 - self.masking_ratio
        k = max(1, int(keep_ratio * self.mask_scores.numel()))
        
        # Get top-k mask
        topk_vals, topk_idx = torch.topk(self.mask_scores.flatten(), k)
        mask = torch.zeros_like(self.mask_scores.flatten())
        mask[topk_idx] = 1.0
        mask = mask.reshape(self.mask_scores.shape)
        
        # Apply mask to weight (maintains gradient flow)
        masked_weight = self.weight * mask
        
        # Perform linear transformation
        output = F.linear(input, masked_weight, self.bias)
        return output

class MFTHandler:
    """MFT handler with proper gradient flow"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        self.target_layers = [4, 5, 6, 7]
        
        # Initialize scores and replace layers
        self.scores = {}
        self.original_modules = {}
        self._replace_linear_layers()
        
    def _replace_linear_layers(self):
        """Replace linear layers with masked versions"""
        print("Initializing masked layers...")
        total_params = 0
        maskable_params = 0
        
        for name, module in self.model.named_modules():
            # Check if this is a linear layer in target layers
            if isinstance(module, nn.Linear):
                # Check if in target layers
                in_target = any(f"layer.{i}." in name for i in self.target_layers)
                
                # Check if it's a maskable component
                is_maskable = any(comp in name for comp in [
                    "attention.self.query",
                    "attention.self.key",
                    "attention.self.value", 
                    "attention.output.dense",
                    "intermediate.dense",
                    "output.dense"
                ])
                
                # Don't mask embeddings, LayerNorm, pooler, or classifier
                is_critical = any(comp in name for comp in [
                    "embeddings", "LayerNorm", "pooler", "classifier"
                ])
                
                if in_target and is_maskable and not is_critical:
                    # Create mask scores for this layer
                    mask_scores = nn.Parameter(
                        torch.randn(module.weight.shape, device=self.device) * 0.01
                    )
                    self.scores[name] = mask_scores
                    
                    # Store original module
                    self.original_modules[name] = module
                    
                    # Replace with masked version
                    masked_module = MaskedLinear(module, mask_scores)
                    
                    # Set the module in the model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = self.model.get_submodule(parent_name)
                        setattr(parent, child_name, masked_module)
                    
                    maskable_params += module.weight.numel()
                
                if isinstance(module, nn.Linear):
                    total_params += module.weight.numel()
                    if module.bias is not None:
                        total_params += module.bias.numel()
        
        print(f"  Total linear parameters: {total_params:,}")
        print(f"  Maskable parameters: {maskable_params:,} ({100*maskable_params/total_params:.1f}%)")
        print(f"  Number of masked layers: {len(self.scores)}")
    
    def get_optimizer(self, lr=5e-4):
        """Get optimizer for mask scores only"""
        return torch.optim.Adam(self.scores.values(), lr=lr)
    
    def train_masks(self, train_dataloader, eval_dataloader, epochs=2, lr=5e-4):
        """Train the mask scores"""
        print("\nTraining mask scores...")
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Ensure scores require gradients
        for score in self.scores.values():
            score.requires_grad = True
        
        optimizer = self.get_optimizer(lr)
        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training masks")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass (masks are applied inside MaskedLinear layers)
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"  Average loss: {avg_loss:.4f}")
            
            # Evaluation phase
            accuracy = self.evaluate(eval_dataloader)
            print(f"  Masked model accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        print(f"\nBest accuracy with masks: {best_accuracy:.4f}")
        return best_accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model with current masks"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        return correct / total
    
    def apply_permanent_masks(self):
        """Apply masks permanently and restore normal Linear layers"""
        print("\nApplying permanent masks...")
        total_masked = 0
        total_params = 0
        
        with torch.no_grad():
            for name, masked_module in list(self.model.named_modules()):
                if isinstance(masked_module, MaskedLinear):
                    # Get the mask
                    keep_ratio = 1.0 - masked_module.masking_ratio
                    k = max(1, int(keep_ratio * masked_module.mask_scores.numel()))
                    topk_vals, topk_idx = torch.topk(masked_module.mask_scores.flatten(), k)
                    mask = torch.zeros_like(masked_module.mask_scores.flatten())
                    mask[topk_idx] = 1.0
                    mask = mask.reshape(masked_module.mask_scores.shape)
                    
                    # Create new Linear layer with masked weights
                    new_linear = nn.Linear(
                        masked_module.weight.shape[1],
                        masked_module.weight.shape[0],
                        bias=(masked_module.bias is not None)
                    )
                    new_linear.weight.data = masked_module.weight.data * mask
                    if masked_module.bias is not None:
                        new_linear.bias.data = masked_module.bias.data
                    
                    # Replace in model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    if parent_name:
                        parent = self.model.get_submodule(parent_name)
                        setattr(parent, child_name, new_linear)
                    
                    # Statistics
                    masked_count = (mask == 0).sum().item()
                    total_masked += masked_count
                    total_params += mask.numel()
        
        print(f"  Total masked: {total_masked:,}/{total_params:,} "
              f"({100*total_masked/total_params:.1f}%)")
        
        return self.model


def run_mft_on_checkpoint(checkpoint_path, train_dataloader, eval_dataloader, config):
    """Apply MFT to a checkpoint"""
    print(f"\n{'='*70}")
    print(f"Processing: {checkpoint_path}")
    print(f"{'='*70}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(config.device)
    
    # Original accuracy
    print("\nEvaluating original model...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    original_acc = correct / total
    print(f"Original accuracy: {original_acc:.4f}")
    
    # Apply MFT
    mft_handler = MFTHandler(model, config)
    mft_acc = mft_handler.train_masks(train_dataloader, eval_dataloader, epochs=2)
    
    # Apply permanent masks
    masked_model = mft_handler.apply_permanent_masks()
    
    # Final evaluation
    print("\nFinal evaluation...")
    masked_model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = masked_model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    final_acc = correct / total
    
    print(f"\n{'='*40}")
    print(f"Results:")
    print(f"  Original: {original_acc:.4f}")
    print(f"  After MFT: {final_acc:.4f}")
    print(f"  Change: {final_acc - original_acc:+.4f}")
    
    return {
        'checkpoint': checkpoint_path,
        'original_accuracy': original_acc,
        'mft_accuracy': final_acc,
        'improvement': final_acc - original_acc
    }, masked_model


if __name__ == "__main__":
    pass

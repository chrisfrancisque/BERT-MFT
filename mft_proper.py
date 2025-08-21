"""
Proper MFT implementation based on the paper
Applies learnable masks to already-trained models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
        # Pass gradient straight through
        return grad_output, None

class MFTHandler:
    """
    Mask Fine-Tuning handler that learns which parameters to mask
    Applied AFTER normal fine-tuning is complete
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        
        # Task-specific layer targeting
        self.target_layers = self._get_target_layers()
        
        # Components to never mask
        self.never_mask = [
            "embeddings",  # All embeddings
            "LayerNorm",   # All layer norms
            "pooler",      # Pooler layer
            "classifier",  # Classification head
        ]
        
        # Components that can be masked (in target layers only)
        self.can_mask = [
            "attention.self.query",
            "attention.self.key", 
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
        
        # Initialize learnable scores
        self.scores = nn.ParameterDict()
        self._initialize_scores()
        
    def _get_target_layers(self):
        """Get target layers based on task"""
        # You can adjust these based on your task
        # Default to layers 4-7 (good for general tasks)
        return [4, 5, 6, 7]
    
    def _should_mask(self, param_name):
        """Determine if a parameter should be considered for masking"""
        # Never mask critical components
        if any(pattern in param_name for pattern in self.never_mask):
            return False
        
        # Must be in target layers
        in_target_layer = any(f"layer.{i}." in param_name for i in self.target_layers)
        if not in_target_layer:
            return False
        
        # Must be a maskable component type
        is_maskable = any(pattern in param_name for pattern in self.can_mask)
        if not is_maskable:
            return False
        
        return True
    
    def _initialize_scores(self):
        """Initialize learnable importance scores"""
        print("Initializing learnable scores for maskable parameters...")
        maskable_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            if self._should_mask(name):
                # Initialize scores with small random values
                self.scores[name] = nn.Parameter(
                    torch.randn(param.shape, device=self.device) * 0.01
                )
                maskable_params += param.numel()
                
        print(f"  Total parameters: {total_params:,}")
        print(f"  Maskable parameters: {maskable_params:,} ({100*maskable_params/total_params:.1f}%)")
        print(f"  Target layers: {self.target_layers}")
    
    def create_masks(self, masking_ratio=0.9):
        """Create binary masks from learned scores"""
        masks = {}
        
        for name, scores in self.scores.items():
            if masking_ratio == 1.0:
                # Use threshold-based masking
                masks[name] = StraightThroughEstimator.apply(scores)
            else:
                # Use ratio-based masking (keep top k%)
                keep_ratio = 1.0 - masking_ratio
                k = max(1, int(keep_ratio * scores.numel()))
                
                # Get top-k values
                topk_vals, topk_idx = torch.topk(
                    scores.flatten(), k, largest=True
                )
                
                # Create binary mask
                mask = torch.zeros_like(scores.flatten())
                mask[topk_idx] = 1.0
                masks[name] = mask.reshape(scores.shape)
        
        return masks
    
    def apply_masks_temporary(self, masks):
        """Apply masks temporarily for forward pass"""
        original_params = {}
        
        # Store original values and apply masks
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in masks:
                    original_params[name] = param.data.clone()
                    param.data = param.data * masks[name]
        
        return original_params
    
    def restore_params(self, original_params):
        """Restore original parameters after forward pass"""
        with torch.no_grad():
            for name, param_data in original_params.items():
                self.model.get_parameter(name).data.copy_(param_data)
    
    def train_masks(self, train_dataloader, eval_dataloader, epochs=2, lr=5e-4):
        """Train the importance scores"""
        print("\nTraining importance scores...")
        
        # Only train the scores, not model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Optimizer for scores only
        optimizer = torch.optim.Adam(self.scores.parameters(), lr=lr)
        
        best_accuracy = 0
        best_scores = {}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training masks")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Create and apply masks
                masks = self.create_masks(masking_ratio=0.9)
                original_params = self.apply_masks_temporary(masks)
                
                # Forward pass with masked model
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass (updates scores only)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Restore original parameters
                self.restore_params(original_params)
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"  Average training loss: {avg_loss:.4f}")
            
            # Evaluation
            accuracy = self.evaluate_with_masks(eval_dataloader)
            print(f"  Masked model accuracy: {accuracy:.4f}")
            
            # Save best scores
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_scores = {name: score.data.clone() 
                              for name, score in self.scores.items()}
        
        # Restore best scores
        for name, score in best_scores.items():
            self.scores[name].data.copy_(score)
        
        print(f"\nBest masked accuracy: {best_accuracy:.4f}")
        return best_accuracy
    
    def evaluate_with_masks(self, dataloader):
        """Evaluate model with current masks"""
        self.model.eval()
        correct = 0
        total = 0
        
        masks = self.create_masks(masking_ratio=0.9)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Apply masks
                original_params = self.apply_masks_temporary(masks)
                
                # Forward pass
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
                
                # Restore params
                self.restore_params(original_params)
        
        return correct / total
    
    def apply_final_masks(self):
        """Apply learned masks permanently to create final model"""
        print("\nApplying final masks...")
        
        masks = self.create_masks(masking_ratio=0.9)
        
        total_masked = 0
        total_params = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in masks:
                    mask = masks[name]
                    param.data = param.data * mask
                    
                    masked_count = (mask == 0).sum().item()
                    param_count = mask.numel()
                    
                    total_masked += masked_count
                    total_params += param_count
                    
                    print(f"  {name}: masked {masked_count}/{param_count} "
                          f"({100*masked_count/param_count:.1f}%)")
        
        print(f"\nTotal: masked {total_masked:,}/{total_params:,} "
              f"({100*total_masked/total_params:.1f}%) of maskable parameters")
        
        return self.model


def run_mft_on_checkpoint(checkpoint_path, train_dataloader, eval_dataloader, config):
    """Apply MFT to a single checkpoint"""
    print(f"\n{'='*70}")
    print(f"Applying MFT to: {checkpoint_path}")
    print(f"{'='*70}")
    
    # Load the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(config.device)
    
    # Evaluate original performance
    print("\nEvaluating original model...")
    original_acc = evaluate_model(model, eval_dataloader, config.device)
    print(f"Original accuracy: {original_acc:.4f}")
    
    # Apply MFT
    mft_handler = MFTHandler(model, config)
    mft_acc = mft_handler.train_masks(train_dataloader, eval_dataloader, epochs=2)
    
    # Apply final masks
    masked_model = mft_handler.apply_final_masks()
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_acc = evaluate_model(masked_model, eval_dataloader, config.device)
    
    results = {
        'checkpoint': checkpoint_path,
        'original_accuracy': original_acc,
        'mft_accuracy': final_acc,
        'improvement': final_acc - original_acc
    }
    
    print(f"\n{'='*40}")
    print(f"Results for {os.path.basename(checkpoint_path)}:")
    print(f"  Original: {original_acc:.4f}")
    print(f"  After MFT: {final_acc:.4f}")
    print(f"  Improvement: {final_acc - original_acc:+.4f}")
    print(f"{'='*40}")
    
    return results, masked_model


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    return correct / total


if __name__ == "__main__":
    # This will be called from run_mft_experiment.py
    pass

"""
Fixed MFT implementation with configurable masking ratio
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import json
import os
from copy import deepcopy

class MFTHandler(nn.Module):
    """MFT handler with configurable masking ratio"""
    def __init__(self, model, config, masking_ratio=0.02):  # Changed default to 0.1
        super().__init__()
        self.model = model
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        self.target_layers = [4, 5, 6, 7]
        self.masking_ratio = masking_ratio  # Percentage to MASK (not keep)
        
        # Create learnable mask scores
        self.mask_scores = nn.ParameterDict()
        self.param_shapes = {}
        self._initialize_scores()
        
    def _initialize_scores(self):
        """Initialize learnable scores for maskable parameters"""
        print(f"Initializing learnable mask scores (will mask {self.masking_ratio*100:.0f}% of params)...")
        total_params = 0
        maskable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Check if this parameter should be masked
            if self._should_mask(name):
                # Create sanitized name for ParameterDict
                safe_name = name.replace('.', '__')
                
                # Initialize scores
                self.mask_scores[safe_name] = nn.Parameter(
                    torch.randn(param.shape, device=self.device) * 0.01,
                    requires_grad=True
                )
                
                # Store shape info
                self.param_shapes[safe_name] = param.shape
                maskable_params += param.numel()
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Maskable parameters: {maskable_params:,} ({100*maskable_params/total_params:.1f}%)")
        print(f"  Number of parameter groups to mask: {len(self.mask_scores)}")
        print(f"  Will mask {self.masking_ratio*100:.0f}% of maskable params = {int(maskable_params*self.masking_ratio):,} params")
    
    def _should_mask(self, name):
        """Check if parameter should be masked"""
        # Must be in target layers
        in_target = any(f"layer.{i}." in name for i in self.target_layers)
        if not in_target:
            return False
        
        # Must be attention or FFN weight (not bias)
        is_maskable = any(comp in name for comp in [
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",
            "intermediate.dense.weight",
            "output.dense.weight"
        ])
        if not is_maskable:
            return False
        
        # Must not be critical component
        is_critical = any(comp in name for comp in [
            "embeddings", "LayerNorm", "pooler", "classifier"
        ])
        if is_critical:
            return False
        
        return True
    
    def get_masks(self):
        """Get binary masks from scores"""
        masks = {}
        
        for safe_name, scores in self.mask_scores.items():
            # Convert back to original name
            orig_name = safe_name.replace('__', '.')
            
            # Keep top (1 - masking_ratio) of parameters
            keep_ratio = 1.0 - self.masking_ratio
            k = max(1, int(keep_ratio * scores.numel()))
            
            # Get top k scores (these will be KEPT)
            topk_vals, topk_idx = torch.topk(scores.flatten(), k)
            
            # Create binary mask (1 = keep, 0 = mask)
            mask = torch.zeros_like(scores.flatten())
            mask[topk_idx] = 1.0
            masks[orig_name] = mask.reshape(scores.shape)
        
        return masks
    
    def forward(self, **inputs):
        """Forward pass with masked parameters"""
        # Get current masks
        masks = self.get_masks()
        
        # Store original parameters
        original_params = {}
        
        # Apply masks
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in masks:
                    original_params[name] = param.data.clone()
                    param.data = param.data * masks[name]
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Restore original parameters
        with torch.no_grad():
            for name, original in original_params.items():
                param = dict(self.model.named_parameters())[name]
                param.data = original
        
        return outputs


def train_mft(model, train_dataloader, eval_dataloader, config, 
              epochs=2, lr=5e-4, masking_ratio=0.2):
    """Train MFT masks"""
    # Create MFT handler with specified masking ratio
    mft_handler = MFTHandler(model, config, masking_ratio=masking_ratio)
    mft_handler.to(config.device)
    
    # Only optimize mask scores
    optimizer = torch.optim.Adam(mft_handler.mask_scores.parameters(), lr=lr)
    
    best_accuracy = 0
    best_masks = None
    
    print("\nTraining mask scores...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        mft_handler.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Forward with masks
            outputs = mft_handler(**batch)
            loss = outputs.loss
            
            # Add L1 regularization on mask scores to encourage sparsity
            l1_loss = 0
            for scores in mft_handler.mask_scores.values():
                l1_loss += 0.0001 * torch.abs(scores).sum()  # Reduced regularization
            loss = loss + l1_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"  Average loss: {avg_loss:.4f}")
        
        # Evaluation
        mft_handler.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(config.device) for k, v in batch.items()}
                outputs = mft_handler(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        accuracy = correct / total
        print(f"  Masked model accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_masks = mft_handler.get_masks()
    
    print(f"\nBest accuracy: {best_accuracy:.4f}")
    
    # Apply best masks permanently
    if best_masks:
        apply_masks_permanently(model, best_masks, masking_ratio)
    
    return best_accuracy


def apply_masks_permanently(model, masks, masking_ratio):
    """Apply masks permanently to model"""
    print(f"\nApplying masks permanently (masking {masking_ratio*100:.0f}% of params)...")
    total_masked = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                param.data = param.data * mask
                
                masked_count = (mask == 0).sum().item()
                param_count = mask.numel()
                
                total_masked += masked_count
                total_params += param_count
    
    print(f"  Masked {total_masked:,}/{total_params:,} parameters "
          f"({100*total_masked/total_params:.1f}%)")


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
    
    # Train MFT with 10% masking (keeping 90%)
    mft_acc = train_mft(model, train_dataloader, eval_dataloader, config, 
                        masking_ratio=0.02)  # Only mask 10%!
    
    # Final evaluation
    print("\nFinal evaluation...")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
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
    }, model

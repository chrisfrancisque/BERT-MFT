"""
Influence Functions for BERT-MFT - TPU Optimized Version
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from collections import OrderedDict

# Import XLA for TPU synchronization
try:
    import torch_xla.core.xla_model as xm
    USE_TPU = True
except:
    USE_TPU = False

logger = logging.getLogger(__name__)

class ParameterInfluenceAnalyzer:
    """
    TPU-optimized influence function analyzer.
    """
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.use_tpu = USE_TPU and 'xla' in str(device)
        
        # Critical layers to exclude
        self.EXCLUDE_PATTERNS = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
        # Numerical stability parameters
        self.damping = 0.01
        self.cg_tol = 1e-6
        self.cg_max_iter = 20
        self.hvp_epsilon = 1e-3
        
        logger.info(f"Initialized ParameterInfluenceAnalyzer on {device}")
        logger.info(f"TPU mode: {self.use_tpu}")
        logger.info(f"Exclusion patterns: {self.EXCLUDE_PATTERNS}")
    
    def compute_validation_gradient(self, val_dataloader):
        """
        Compute gradient of validation loss w.r.t. parameters.
        TPU-optimized version with proper synchronization.
        """
        logger.info("Computing validation gradient...")
        self.model.eval()
        
        # Initialize gradient storage
        val_gradients = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(pat in name for pat in self.EXCLUDE_PATTERNS):
                # Keep on device initially for TPU
                val_gradients[name] = torch.zeros_like(param, device=self.device)
        
        total_loss = 0
        num_batches = 0
        
        # Process in smaller chunks for TPU
        batch_limit = 10 if self.use_tpu else len(val_dataloader)
        
        # Accumulate gradients
        progress_bar = tqdm(val_dataloader, desc="Computing validation gradient", total=min(batch_limit, len(val_dataloader)))
        
        for i, batch in enumerate(progress_bar):
            if i >= batch_limit:
                break
                
            # Batch already on device from MpDeviceLoader
            self.model.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = self.model(**batch)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                if torch.isnan(loss):
                    logger.warning("NaN loss in validation gradient computation")
                    if self.use_tpu:
                        xm.mark_step()
                    continue
                
                loss.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in val_gradients and param.grad is not None:
                        val_gradients[name] += param.grad
            
            # Synchronize TPU every few batches
            if self.use_tpu and i % 5 == 0:
                xm.mark_step()
            
            total_loss += loss.item() if not self.use_tpu else 0
            num_batches += 1
        
        # Final synchronization
        if self.use_tpu:
            xm.mark_step()
        
        # Average gradients and move to CPU
        for name in val_gradients:
            val_gradients[name] = (val_gradients[name] / num_batches).cpu()
        
        # Get loss value after synchronization
        if self.use_tpu:
            total_loss = num_batches  # Placeholder for TPU
        
        logger.info(f"Validation gradient computed. Processed {num_batches} batches")
        return val_gradients
    
    def _compute_train_gradient(self, train_dataloader):
        """Helper to compute gradient over training data - TPU optimized."""
        self.model.train()
        
        gradients = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(pat in name for pat in self.EXCLUDE_PATTERNS):
                gradients[name] = torch.zeros_like(param, device=self.device)
        
        num_batches = 0
        batch_limit = 10 if self.use_tpu else len(train_dataloader)
        
        for i, batch in enumerate(train_dataloader):
            if i >= batch_limit:
                break
                
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            if not torch.isnan(loss):
                loss.backward()
                
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in gradients and param.grad is not None:
                            gradients[name] += param.grad
                
                num_batches += 1
            
            # Synchronize TPU periodically
            if self.use_tpu and i % 5 == 0:
                xm.mark_step()
        
        # Final synchronization
        if self.use_tpu:
            xm.mark_step()
        
        # Average and move to CPU
        for name in gradients:
            gradients[name] = (gradients[name] / num_batches).cpu()
        
        return gradients
    
    def compute_hvp(self, vector_dict, train_dataloader):
        """
        Compute Hessian-vector product - simplified for TPU.
        """
        logger.info("Computing Hessian-vector product...")
        
        # Move vectors to device
        device_vectors = OrderedDict()
        for name, vec in vector_dict.items():
            device_vectors[name] = vec.to(self.device)
        
        # Perturb parameters in positive direction
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in device_vectors:
                    param.data += self.hvp_epsilon * device_vectors[name]
        
        if self.use_tpu:
            xm.mark_step()
        
        # Compute gradient at θ + εv
        grad_plus = self._compute_train_gradient(train_dataloader)
        
        # Perturb parameters in negative direction
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in device_vectors:
                    param.data -= 2 * self.hvp_epsilon * device_vectors[name]
        
        if self.use_tpu:
            xm.mark_step()
        
        # Compute gradient at θ - εv
        grad_minus = self._compute_train_gradient(train_dataloader)
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in device_vectors:
                    param.data += self.hvp_epsilon * device_vectors[name]
        
        if self.use_tpu:
            xm.mark_step()
        
        # Compute HVP (already on CPU)
        hvp = OrderedDict()
        for name in vector_dict:
            if name in grad_plus and name in grad_minus:
                hvp[name] = (grad_plus[name] - grad_minus[name]) / (2 * self.hvp_epsilon)
        
        return hvp

"""
Influence Functions for BERT-MFT
Adapts "What Data Benefits My Classifier?" methodology to measure parameter influence
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ParameterInfluenceAnalyzer:
    """
    Computes influence of zeroing parameters using second-order approximations.
    Adapts influence functions to measure impact of parameter removal.
    """
    
    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # Critical layers to exclude
        self.EXCLUDE_PATTERNS = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
        # Numerical stability parameters
        self.damping = 0.01
        self.cg_tol = 1e-6
        self.cg_max_iter = 20
        self.hvp_epsilon = 1e-3
        
        logger.info(f"Initialized ParameterInfluenceAnalyzer on {device}")
        logger.info(f"Exclusion patterns: {self.EXCLUDE_PATTERNS}")
    
    def compute_validation_gradient(self, val_dataloader):
        """
        Compute gradient of validation loss w.r.t. parameters.
        ∇_θ L_validation
        """
        logger.info("Computing validation gradient...")
        self.model.eval()
        
        # Initialize gradient storage
        val_gradients = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(pat in name for pat in self.EXCLUDE_PATTERNS):
                val_gradients[name] = torch.zeros_like(param, device='cpu')
        
        total_loss = 0
        num_batches = 0
        
        # Accumulate gradients over validation set
        for batch in tqdm(val_dataloader, desc="Computing validation gradient"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.model.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = self.model(**batch)
                loss = F.cross_entropy(outputs.logits, batch['labels'])
                
                if torch.isnan(loss):
                    logger.warning("NaN loss in validation gradient computation")
                    continue
                
                loss.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in val_gradients and param.grad is not None:
                        val_gradients[name] += param.grad.cpu()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Average gradients
        for name in val_gradients:
            val_gradients[name] /= num_batches
        
        logger.info(f"Validation gradient computed. Avg loss: {total_loss/num_batches:.4f}")
        return val_gradients
    
    def compute_hvp(self, vector_dict, train_dataloader):
        """
        Compute Hessian-vector product using finite differences.
        H * v ≈ (∇L(θ + εv) - ∇L(θ - εv)) / (2ε)
        """
        logger.info("Computing Hessian-vector product...")
        
        # Perturb parameters in positive direction
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in vector_dict:
                    param.data += self.hvp_epsilon * vector_dict[name].to(self.device)
        
        # Compute gradient at θ + εv
        grad_plus = self._compute_train_gradient(train_dataloader)
        
        # Perturb parameters in negative direction (2ε total change)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in vector_dict:
                    param.data -= 2 * self.hvp_epsilon * vector_dict[name].to(self.device)
        
        # Compute gradient at θ - εv
        grad_minus = self._compute_train_gradient(train_dataloader)
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in vector_dict:
                    param.data += self.hvp_epsilon * vector_dict[name].to(self.device)
        
        # Compute HVP
        hvp = OrderedDict()
        for name in vector_dict:
            if name in grad_plus and name in grad_minus:
                hvp[name] = (grad_plus[name] - grad_minus[name]) / (2 * self.hvp_epsilon)
        
        return hvp
    
    def _compute_train_gradient(self, train_dataloader):
        """Helper to compute gradient over training data."""
        self.model.train()
        
        gradients = OrderedDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(pat in name for pat in self.EXCLUDE_PATTERNS):
                gradients[name] = torch.zeros_like(param, device='cpu')
        
        num_batches = 0
        for batch in train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, batch['labels'])
            
            if not torch.isnan(loss):
                loss.backward()
                
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in gradients and param.grad is not None:
                            gradients[name] += param.grad.cpu()
                
                num_batches += 1
        
        # Average
        for name in gradients:
            gradients[name] /= num_batches
        
        return gradients
    
    def conjugate_gradient_solve(self, b_dict, train_dataloader, max_iter=None):
        """
        Solve H * x = b using conjugate gradient method.
        Returns x = H^(-1) * b without storing H.
        """
        if max_iter is None:
            max_iter = self.cg_max_iter
        
        logger.info(f"Starting conjugate gradient solve (max_iter={max_iter})...")
        
        # Initialize x = 0
        x = OrderedDict()
        for name in b_dict:
            x[name] = torch.zeros_like(b_dict[name])
        
        # r = b - H*x = b (since x=0)
        r = OrderedDict()
        for name in b_dict:
            r[name] = b_dict[name].clone()
        
        # p = r
        p = OrderedDict()
        for name in r:
            p[name] = r[name].clone()
        
        # rsold = r^T * r
        rsold = sum((r[name] * r[name]).sum().item() for name in r)
        
        for iteration in range(max_iter):
            # Compute H*p with damping
            Hp = self.compute_hvp(p, train_dataloader)
            
            # Add damping: Hp = Hp + damping * p
            for name in Hp:
                Hp[name] = Hp[name] + self.damping * p[name]
            
            # alpha = rsold / (p^T * H * p)
            pHp = sum((p[name] * Hp[name]).sum().item() for name in p)
            
            if abs(pHp) < 1e-10:
                logger.warning(f"CG: pHp too small at iteration {iteration}")
                break
            
            alpha = rsold / pHp
            
            # x = x + alpha * p
            for name in x:
                x[name] = x[name] + alpha * p[name]
            
            # r = r - alpha * H * p
            for name in r:
                r[name] = r[name] - alpha * Hp[name]
            
            # Check convergence
            rsnew = sum((r[name] * r[name]).sum().item() for name in r)
            
            if np.sqrt(rsnew) < self.cg_tol:
                logger.info(f"CG converged in {iteration + 1} iterations (residual: {np.sqrt(rsnew):.6f})")
                break
            
            # p = r + (rsnew/rsold) * p
            beta = rsnew / rsold
            for name in p:
                p[name] = r[name] + beta * p[name]
            
            rsold = rsnew
            
            if iteration % 5 == 0:
                logger.info(f"CG iteration {iteration}: residual = {np.sqrt(rsnew):.6f}")
        
        return x
    
    def compute_parameter_influence(self, param_name, param_indices, val_gradient, train_dataloader):
        """
        Compute influence of zeroing specific parameter elements.
        Influence = ∇L_val^T * H^(-1) * v_j
        where v_j = -θ_j * e_j (perturbation from zeroing)
        """
        # Create perturbation vector
        perturbation = OrderedDict()
        for name, param in self.model.named_parameters():
            if name == param_name:
                pert = torch.zeros_like(param)
                for idx in param_indices:
                    pert[idx] = -param.data[idx].item()
                perturbation[name] = pert.cpu()
            elif not any(pat in name for pat in self.EXCLUDE_PATTERNS):
                perturbation[name] = torch.zeros_like(param, device='cpu')
        
        # Solve H^(-1) * perturbation
        h_inv_v = self.conjugate_gradient_solve(perturbation, train_dataloader)
        
        # Compute influence: val_grad^T * H^(-1) * v
        influence = sum((val_gradient[name] * h_inv_v[name]).sum().item() for name in val_gradient)
        
        return influence
    
    def analyze_parameters_with_influence(self, train_dataloader, val_dataloader, 
                                         gradient_analysis_results=None):
        """
        Main analysis function combining gradient and influence analysis.
        """
        logger.info("\n" + "="*60)
        logger.info("INFLUENCE FUNCTION ANALYSIS")
        logger.info("="*60)
        
        # Step 1: Compute validation gradient
        val_gradient = self.compute_validation_gradient(val_dataloader)
        
        # Step 2: Use gradient analysis results if provided, otherwise analyze all
        if gradient_analysis_results:
            logger.info("Using gradient analysis results to filter candidates...")
            candidate_params = self._get_candidate_params_from_gradient_analysis(
                gradient_analysis_results
            )
        else:
            logger.info("Analyzing all non-excluded parameters...")
            candidate_params = self._get_all_candidate_params()
        
        logger.info(f"Total candidate parameters: {len(candidate_params)}")
        
        # Step 3: Phase 1 - Quick diagonal Hessian approximation
        logger.info("\nPhase 1: Diagonal Hessian screening...")
        top_candidates = self._diagonal_screening(
            candidate_params, val_gradient, train_dataloader
        )
        
        # Step 4: Phase 2 - Full influence computation for top candidates
        logger.info(f"\nPhase 2: Full influence computation for top {len(top_candidates)} candidates...")
        
        influence_results = []
        for i, (param_name, indices) in enumerate(tqdm(top_candidates[:100], 
                                                       desc="Computing influences")):
            influence = self.compute_parameter_influence(
                param_name, indices, val_gradient, train_dataloader
            )
            
            influence_results.append({
                'param_name': param_name,
                'indices': indices,
                'influence': influence,
                'would_improve': influence < 0  # Negative influence means improvement
            })
            
            if i % 10 == 0 and i > 0:
                logger.info(f"Computed {i} influences. Found {sum(r['would_improve'] for r in influence_results)} beneficial removals")
        
        # Sort by influence (most beneficial first)
        influence_results.sort(key=lambda x: x['influence'])
        
        # Summary statistics
        total_analyzed = len(influence_results)
        beneficial = sum(1 for r in influence_results if r['would_improve'])
        harmful = total_analyzed - beneficial
        
        logger.info("\n" + "="*60)
        logger.info("INFLUENCE ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Parameters analyzed: {total_analyzed}")
        logger.info(f"Beneficial to remove: {beneficial} ({100*beneficial/total_analyzed:.1f}%)")
        logger.info(f"Harmful to remove: {harmful} ({100*harmful/total_analyzed:.1f}%)")
        
        if beneficial > 0:
            logger.info(f"\nTop 5 most beneficial parameters to remove:")
            for i, result in enumerate(influence_results[:5], 1):
                logger.info(f"  {i}. {result['param_name']} @ {result['indices']}: influence={result['influence']:.6f}")
        
        return {
            'influence_results': influence_results,
            'val_gradient': val_gradient,
            'summary': {
                'total_analyzed': total_analyzed,
                'beneficial_count': beneficial,
                'harmful_count': harmful,
                'beneficial_percentage': 100 * beneficial / total_analyzed if total_analyzed > 0 else 0
            }
        }
    
    def _get_candidate_params_from_gradient_analysis(self, gradient_results):
        """Extract candidate parameters from gradient analysis."""
        candidates = []
        
        for param_info in gradient_results.get('detrimental_params', []):
            name = param_info['name']
            if any(pat in name for pat in self.EXCLUDE_PATTERNS):
                continue
            
            # Get top detrimental indices
            mask = param_info['detrimental_mask']
            movement = param_info['movement_magnitude']
            
            # Get indices of top 10 parameters by movement
            if mask.sum() > 0:
                masked_movement = movement.clone()
                masked_movement[~mask] = -1
                top_k = min(10, mask.sum().item())
                _, top_indices = torch.topk(masked_movement.flatten(), top_k)
                
                # Convert flat indices to multi-dimensional
                indices = []
                for flat_idx in top_indices:
                    idx = []
                    remaining = flat_idx.item()
                    for dim_size in reversed(movement.shape):
                        idx.insert(0, remaining % dim_size)
                        remaining //= dim_size
                    indices.append(tuple(idx))
                
                candidates.append((name, indices))
        
        return candidates
    
    def _get_all_candidate_params(self):
        """Get all non-excluded parameters as candidates."""
        candidates = []
        
        for name, param in self.model.named_parameters():
            if any(pat in name for pat in self.EXCLUDE_PATTERNS):
                continue
            
            # Sample up to 10 random indices per parameter
            num_elements = param.numel()
            num_samples = min(10, num_elements)
            
            flat_indices = torch.randperm(num_elements)[:num_samples]
            indices = []
            for flat_idx in flat_indices:
                idx = []
                remaining = flat_idx.item()
                for dim_size in reversed(param.shape):
                    idx.insert(0, remaining % dim_size)
                    remaining //= dim_size
                indices.append(tuple(idx))
            
            candidates.append((name, indices))
        
        return candidates
    
    def _diagonal_screening(self, candidates, val_gradient, train_dataloader):
        """Quick screening using diagonal Hessian approximation."""
        # For quick screening, just use gradient magnitude * parameter value
        scores = []
        
        for param_name, indices in candidates:
            param = dict(self.model.named_parameters())[param_name]
            
            for idx in indices:
                # Simple approximation: influence ≈ -grad * param_value
                grad_val = val_gradient[param_name][idx].item() if param_name in val_gradient else 0
                param_val = param.data[idx].item()
                approx_influence = -grad_val * param_val
                
                scores.append((param_name, [idx], approx_influence))
        
        # Sort by approximate influence
        scores.sort(key=lambda x: x[2])
        
        # Return top 100 candidates for full analysis
        return [(s[0], s[1]) for s in scores[:100]]

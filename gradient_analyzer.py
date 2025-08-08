import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

class GradientAnalyzer:
    """
    Collects and analyzes gradients without updating model weights.
    Stores actual gradient tensors with same shape as parameters.
    """
    
    def __init__(self, model, config):
        """Initialize gradient analyzer"""

        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        #Storage for accumulated gradients that have the same structure as model parameter
        self.accumulated_gradients = OrderedDict()

        # Storage for tracking metrics
        self.losses_per_step = []

        #Initialize gradient storage with zeros
        self._initialize_gradient_storage()
    def _initialize_gradient_storage(self):
        logger.info("Initializing gradient storage for all parameters")

        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #Create zero tensor with same shape as parameter
                self.accumulated_gradients[name] = torch.zeros_like(param, device ='cpu')
                total_params += param.numel()
        logger.info(f"Initialized gradient storage for {len(self.accumulated_gradients)} parameter tensors")
        logger.info(f"Total parameters being tracked: {total_params:,}")
    
    def collect_gradients(self, dataloader):
        """
        Collect gradients for one epoch without updating weights.
        
        Args:
            dataloader: DataLoader with training samples
            
        Returns:
            Dict containing accumulated gradients and metrics
        """

        logger.info("Starting gradient collection")
        self.model.train() 

        # Reset accumulators
        self.losses_per_step = []
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name].zero_()
        

        #Progress Bar for all batches
        progress_bar = tqdm(dataloader, desc="Collecting gradients")

        for step, batch in enumerate(progress_bar):
            #Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Enable gradient computation
            self.model.zero_grad()

            # Forward Pass
            with torch.set_grad_enabled(True):
                outputs = self.model(**batch)
                logits = outputs.logits

                # Compute loss in float32 fpr numerical stability
                with torch.autocast(self.device.type, enabled=False):
                    logits_fp32 = logits.float()
                    labels = batch['labels']
                    loss = F.cross_entropy(logits_fp32, labels)
                
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at step {step}, skipping batch")
                    continue

                loss.backward()
            # Accumulate gradients without updating weights    
            self._accumulate_gradients()

            #Track Metrics
            loss_value = loss.item()
            self.losses_per_step.append(loss_value)

            #Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'step': f'{step+1}/{len(dataloader)}'
            })
        
        logger.info(f"Gradient collection completed. Process {len(dataloader)} batches.")
        logger.info(f"Average loss: {np.mean(self.losses_per_step):.4f}")

        return {
            'gradients': self.accumulated_gradients,
            'losses': self.losses_per_step,
            'num_batches': len(dataloader)
        }
    
    def _accumulate_gradients(self):
        """Accumulate gradients from current batch WITHOUT updating weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    #add current gradient to accumulator (move to CPU to save GPU)
                    self.accumulated_gradients[name] += param.grad.cpu()
    
    def analyze_detrimental_parameters(self, learning_rate=None):
        """
        Identify parameters that would move toward zero with gradient descent.
        
        A parameter is "detrimental" if applying the gradient update would
        reduce its absolute value (move it closer to zero).
        
        Args:
            learning_rate: Learning rate for theoretical update (default from config)
            
        Returns:
            Dict containing analysis results
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        logger.info(f"Analyzing detrimental parameters with lr={learning_rate}")

        detrimental_params = []
        total_params_analyzed = 0

        # Analyze each parameter
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'classifier' in name:
                logger.info(f"Skipping classifier layer: {name}")
                continue

            param_data = param.data.cpu()
            gradient = self.accumulated_gradients[name]

            theoretical_update = param_data - learning_rate * gradient

            original_abs = torch.abs(param_data)
            updated_abs = torch.abs(theoretical_update)

            moving_to_zero_mask = updated_abs < original_abs

            movement_magnitude = original_abs - updated_abs
            movement_magnitude[~moving_to_zero_mask] = 0

            # Store information about detrimental parameters
            num_detrimental = moving_to_zero_mask.sum().item()
            if num_detrimental > 0:
                detrimental_params.append({
                    'name': name,
                    'shape': list(param.shape),
                    'total_elements': param.numel(),
                    'num_detrimental': num_detrimental,
                    'detrimental_mask': moving_to_zero_mask,
                    'movement_magnitude': movement_magnitude,
                    'max_movement': movement_magnitude.max().item(),
                    'mean_movement': movement_magnitude[moving_to_zero_mask].mean().item() if num_detrimental > 0 else 0
                })

            total_params_analyzed += param.numel()
        
        total_detrimental = sum(p['num_detrimental'] for p in detrimental_params)

        logger.info(f"Analysis complete:")
        logger.info(f"  Total parameters analyzed: {total_params_analyzed:,}")
        logger.info(f"  Total detrimental parameters: {total_detrimental:,}")
        logger.info(f"  Percentage detrimental: {100 * total_detrimental / total_params_analyzed:.2f}%")
        logger.info(f"  Number of layers with detrimental params: {len(detrimental_params)}")

        return {
            'detrimental_params' : detrimental_params,
            'total_params' : total_params_analyzed,
            'total_detrimental' : total_detrimental,
            'learning_rate' : learning_rate
        }
    def get_gradient_statistics(self):
        """
        Compute statistics about collected gradients.
        
        Returns:
            Dict with gradient statistics
        """

        stats = {}

        for name, grad in self.accumulated_gradients.items():
            grad_numpy = grad.numpy()
            stats[name] = {
                'mean': float(np.mean(grad_numpy)),
                'std' : float(np.std(grad_numpy)),
                'min': float(np.min(grad_numpy)),
                'max': float(np.max(grad_numpy)),
                'norm': float(np.linalg.norm(grad_numpy)),
                'shape': list(grad.shape),
                'num_elements': grad.numel()
            }
        return stats
    








        

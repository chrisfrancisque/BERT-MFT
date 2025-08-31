import torch
import numpy as np
import logging
import heapq
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DetrimentalParameterHandler:
    """Identifies and zeros out the top detrimental parameters"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.original_state_dict = None
    
    def save_original_state(self):
        logger.info("Saving original model state")
        self.original_state_dict = {
            name: param.data.clone().cpu() for name, param in self.model.named_parameters()
        }
        
    def identify_top_detrimental_parameters(self, analysis_results: Dict) -> Dict:
        """Enhanced version that works with loss contribution analysis"""
        detrimental_params = analysis_results['detrimental_params']
        
        # Count total detrimental parameters (already filtered by loss contribution)
        total_detrimental = sum(p['num_detrimental'] for p in detrimental_params)
        
        # Check if we're using enhanced analysis
        using_loss_check = analysis_results.get('using_loss_check', False)
        
        if using_loss_check:
            logger.info("Using enhanced analysis results with loss contribution check")
            logger.info(f"  Parameters that would harm if removed: {analysis_results.get('total_harmful_skipped', 0):,}")
            logger.info(f"  Parameters helpful to remove: {analysis_results.get('total_helpful', 0):,}")
            logger.info(f"  Parameters neutral to remove: {analysis_results.get('total_neutral', 0):,}")
            logger.info(f"  Total safe to remove: {total_detrimental:,}")
            
            # Adaptive percentage based on how many safe parameters we found
            if total_detrimental < 1000:
                percentage = 1.0  # Take all if very few
                logger.warning(f"Only {total_detrimental} safe parameters found, zeroing all")
            elif total_detrimental < 10000:
                percentage = 0.5  # Take 50% if few
                logger.info(f"Found {total_detrimental} safe parameters, zeroing 50%")
            else:
                percentage = 0.1  # Take 10% if many
                logger.info(f"Found {total_detrimental} safe parameters, zeroing 10%")
        else:
            # Original method
            percentage = 0.01
            logger.info("Using original analysis (no loss check)")
        
        num_to_keep = int(np.ceil(total_detrimental * percentage))
        
        logger.info(f"Will identify top {num_to_keep:,} parameters from {total_detrimental:,} candidates")
        logger.info(f"This represents {100.0 * num_to_keep / analysis_results['total_params']:.4f}% of total model")
        
        # Use heap to efficiently find top parameters
        top_params_heap = []
        
        for param_idx, param_info in enumerate(detrimental_params):
            if param_idx % 20 == 0:
                logger.info(f"Processing parameter {param_idx}/{len(detrimental_params)}")
            
            name = param_info['name']
            mask = param_info['detrimental_mask']
            movement = param_info['movement_magnitude']
            
            mask_np = mask.numpy()
            movement_np = movement.numpy()
            
            detrimental_indices = np.where(mask_np)
            num_detrimental_in_param = len(detrimental_indices[0])
            
            batch_size = 10000
            for batch_start in range(0, num_detrimental_in_param, batch_size):
                batch_end = min(batch_start + batch_size, num_detrimental_in_param)
                
                for idx in range(batch_start, batch_end):
                    index_tuple = tuple(int(d[idx]) for d in detrimental_indices)
                    movement_value = float(movement_np[index_tuple])
                    
                    if len(top_params_heap) < num_to_keep:
                        heapq.heappush(top_params_heap, (movement_value, name, index_tuple))
                    elif movement_value > top_params_heap[0][0]:
                        heapq.heapreplace(top_params_heap, (movement_value, name, index_tuple))
        
        params_to_zero = []
        while top_params_heap:
            movement, name, index_tuple = heapq.heappop(top_params_heap)
            params_to_zero.append({
                'param_name': name,
                'index': index_tuple,
                'movement': movement
            })
        
        params_to_zero.reverse()
        
        logger.info(f"Identified top {len(params_to_zero):,} detrimental parameters")
        if params_to_zero:
            logger.info(f"  Largest movement: {params_to_zero[0]['movement']:.6f}")
            logger.info(f"  Smallest movement in selection: {params_to_zero[-1]['movement']:.6f}")
        
        # Group by parameter name for efficient zeroing
        params_to_zero_by_name = {}
        for item in params_to_zero:
            name = item['param_name']
            if name not in params_to_zero_by_name:
                params_to_zero_by_name[name] = []
            params_to_zero_by_name[name].append(item['index'])
        
        return {
            'params_to_zero': params_to_zero,
            'params_to_zero_by_name': params_to_zero_by_name,
            'total_detrimental': total_detrimental,
            'num_zeroed': len(params_to_zero),
            'percentage_of_detrimental': 100.0 * len(params_to_zero) / total_detrimental if total_detrimental > 0 else 0,
            'percentage_of_total': 100.0 * len(params_to_zero) / analysis_results['total_params'],
            'using_loss_check': using_loss_check
        }
        
    def zero_parameters(self, zeroing_info: Dict) -> Dict:
        """Zero the identified parameters"""
        if self.original_state_dict is None:
            raise ValueError("Original state not saved. Call save_original_state() first")
        
        logger.info("Zeroing identified parameters")
        
        params_to_zero_by_name = zeroing_info['params_to_zero_by_name']
        
        zeroed_count = 0
        affected_layers = set()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params_to_zero_by_name:
                    affected_layers.add(name)
                    indices_to_zero = params_to_zero_by_name[name]
                    
                    for index_tuple in indices_to_zero:
                        param.data[index_tuple] = 0.0
                        zeroed_count += 1
        
        verification_stats = self._verify_zeroing(params_to_zero_by_name)
        
        logger.info(f"Zeroing complete:")
        logger.info(f"  Parameters zeroed: {zeroed_count:,}")
        logger.info(f"  Layers affected: {len(affected_layers)}")
        logger.info(f"  Verification: {'PASSED' if verification_stats['all_zeroed'] else 'FAILED'}")
        
        return {
            'zeroed_count': zeroed_count,
            'affected_layers': list(affected_layers),
            'num_affected_layers': len(affected_layers),
            'verification': verification_stats
        }
    
    def _verify_zeroing(self, params_to_zero_by_name: Dict) -> Dict:
        """Verify that parameters were actually zeroed"""
        all_zeroed = True
        non_zero_count = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params_to_zero_by_name:
                    for index_tuple in params_to_zero_by_name[name]:
                        value = param.data[index_tuple].item()
                        if abs(value) > 1e-10:
                            all_zeroed = False
                            non_zero_count += 1
        
        return {
            'all_zeroed': all_zeroed,
            'non_zero_count': non_zero_count
        }
    
    def restore_original_model(self):
        """Restore model to original state before zeroing"""
        if self.original_state_dict is None:
            raise ValueError("No original state to restore")
        
        logger.info("Restoring original model state")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = self.original_state_dict[name].to(param.device)
        
        logger.info("Model restored to original state")
    
    def get_zeroing_summary(self, zeroing_info: Dict) -> str:
        """Generate a readable summary of the zeroing operation"""
        summary = [
            "\n" + "="*60,
            "PARAMETER ZEROING SUMMARY",
            "="*60,
            f"Analysis method: {'Enhanced (with loss check)' if zeroing_info.get('using_loss_check') else 'Original'}",
            f"Total model parameters: {zeroing_info.get('total_params', 0):,}",
            f"Safe parameters found: {zeroing_info['total_detrimental']:,}",
            f"Parameters to zero: {zeroing_info['num_zeroed']:,}",
            f"Percentage of safe params: {zeroing_info['percentage_of_detrimental']:.2f}%",
            f"Percentage of total model: {zeroing_info['percentage_of_total']:.4f}%",
            "",
            "Top 5 parameters with largest movement toward zero:"
        ]
        
        for i, param in enumerate(zeroing_info['params_to_zero'][:5], 1):
            summary.append(f"  {i}. {param['param_name']} @ {param['index']}: {param['movement']:.6f}")
        
        summary.append("="*60 + "\n")
        
        return "\n".join(summary)

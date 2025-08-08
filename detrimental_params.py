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
        """
        Memory-efficient version that identifies the top 10% of detrimental parameters.
        Uses a heap to avoid storing all 39M+ parameters in memory.
        
        Args:
            analysis_results: Results from gradient analyzer
            
        Returns:
            Dict containing parameters to zero and statistics
        """
        detrimental_params = analysis_results['detrimental_params']
        
        # First pass: count total detrimental parameters
        total_detrimental = sum(p['num_detrimental'] for p in detrimental_params)
        num_to_keep = int(np.ceil(total_detrimental * 0.1))
        
        logger.info(f"Total detrimental parameters: {total_detrimental:,}")
        logger.info(f"Will identify top 10% ({num_to_keep:,} parameters)")
        
        # Use a min-heap to keep only the top num_to_keep parameters
        # Heap elements are (movement_magnitude, param_name, index_tuple)
        top_params_heap = []
        
        # Process each parameter tensor
        for param_idx, param_info in enumerate(detrimental_params):
            if param_idx % 20 == 0:  # Progress update every 20 parameters
                logger.info(f"Processing parameter {param_idx}/{len(detrimental_params)}")
            
            name = param_info['name']
            mask = param_info['detrimental_mask']
            movement = param_info['movement_magnitude']
            
            # Convert to numpy for easier manipulation
            mask_np = mask.numpy()
            movement_np = movement.numpy()
            
            # Get indices of detrimental parameters using numpy
            detrimental_indices = np.where(mask_np)
            
            # Process in batches to avoid creating too many objects at once
            batch_size = 10000  # Process 10k at a time
            num_detrimental_in_param = len(detrimental_indices[0])
            
            for batch_start in range(0, num_detrimental_in_param, batch_size):
                batch_end = min(batch_start + batch_size, num_detrimental_in_param)
                
                for idx in range(batch_start, batch_end):
                    # Get index tuple for this parameter
                    index_tuple = tuple(int(d[idx]) for d in detrimental_indices)
                    
                    # Get movement magnitude for this specific parameter
                    movement_value = float(movement_np[index_tuple])
                    
                    if len(top_params_heap) < num_to_keep:
                        # Heap not full yet, add this parameter
                        heapq.heappush(top_params_heap, (movement_value, name, index_tuple))
                    elif movement_value > top_params_heap[0][0]:
                        # This parameter has larger movement than the smallest in heap
                        heapq.heapreplace(top_params_heap, (movement_value, name, index_tuple))
        
        # Convert heap to list of parameters to zero
        params_to_zero = []
        while top_params_heap:
            movement, name, index_tuple = heapq.heappop(top_params_heap)
            params_to_zero.append({
                'param_name': name,
                'index': index_tuple,
                'movement': movement
            })
        
        # Reverse to get descending order
        params_to_zero.reverse()
        
        logger.info(f"Identified top {len(params_to_zero):,} detrimental parameters")
        if params_to_zero:
            logger.info(f"  Largest movement: {params_to_zero[0]['movement']:.6f}")
            logger.info(f"  Smallest movement in top 10%: {params_to_zero[-1]['movement']:.6f}")
        
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
            'percentage_of_detrimental': 1.0,
            'percentage_of_total': 100.0 * len(params_to_zero) / analysis_results['total_params']
        }
        
    def zero_parameters(self, zeroing_info: Dict) -> Dict:
        """
        Identify the top 10% of detrimental parameters based on movement magnitude.
        
        Args:
            analysis_results: Results from gradient analyzer
            
        Returns:
            Dict containing parameters to zero and statistics
        """
        if self.original_state_dict is None:
            raise ValueError("Original stae not saved. Call save_original_state() first")
        
        logger.info("Zeroing top 10% detrimenal parameters")

        params_to_zero_by_name = zeroing_info['params_to_zero_by_name']

        #Track what we're zeroing
        zeroed_count = 0
        affected_layers = set()

        #Zero the parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params_to_zero_by_name:
                    affected_layers.add(name)
                    indices_to_zero = params_to_zero_by_name[name]

                    #Zero each identified parameter
                    for index_tuple in indices_to_zero:
                        param.data[index_tuple] = 0.0
                        zeroed_count +=1
        
        verification_stats = self._verify_zeroing(params_to_zero_by_name)

        logger.info(f"Zeroing complete:")
        logger.info(f"Parameters zeroed: {zeroed_count:,}")
        logger.info(f"Layers affected: {len(affected_layers)}")
        logger.info(f" Verification: {'PASSED' if verification_stats['all_zeroed'] else 'FAILED'}")

        return{
            'zeroed_count': zeroed_count,
            'affected_layers': list(affected_layers),
            'num_affected_layers': len(affected_layers),
            'verification': verification_stats 
        }
    def _verify_zeroing(self, params_to_zero_by_name: Dict) -> Dict:
        """
        Verify that parameters were actually zeroed.
        
        Args:
            params_to_zero_by_name: Dict of parameters that should be zeroed
            
        Returns:
            Verification statistics
        """
        all_zeroed = True
        non_zero_count = 0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params_to_zero_by_name:
                    for index_tuple in params_to_zero_by_name[name]:
                        value = param.data[index_tuple].item()
                        if abs(value) > 1e-10: #floating point tolerance
                            all_zeroed= False
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
        """
        Generate a readable summary of the zeroing operation.
        
        Args:
            zeroing_info: Dict from identify_top_detrimental_parameters
            
        Returns:
            Summary string
        """
         
        summary = [
            "\n" + "="*60,
            "DETRIMENTAL PARAMETER ZEROING SUMMARY",
            "="*60,
            f"Total model parameters: {zeroing_info.get('total_params', 0):,}",
            f"Detrimental parameters found: {zeroing_info['total_detrimental']:,}",
            f"Parameters to zero (top 10%): {zeroing_info['num_zeroed']:,}",
            f"Percentage of total parameters zeroed: {zeroing_info['percentage_of_total']:.4f}%",
            "",
            "Top 5 parameters with largest movement toward zero:"
        ]
        
        # Show top 5 parameters
        for i, param in enumerate(zeroing_info['params_to_zero'][:5], 1):
            summary.append(f"  {i}. {param['param_name']} @ {param['index']}: {param['movement']:.6f}")
        
        summary.append("="*60 + "\n")
        
        return "\n".join(summary)
            


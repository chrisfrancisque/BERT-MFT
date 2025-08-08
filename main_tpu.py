import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo"
os.environ["XLA_USE_TORCH_COMPILE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"

import torch
torch._dynamo.config.disable = True

import logging
from datetime import datetime
import json
import traceback
import time
import numpy as np 

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("Error: TPU libraries not available. This script must be run on a TPU VM")
    sys.exit(1)

from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from evaluation import ModelEvaluator
from detrimental_params import DetrimentalParameterHandler

def setup_logging(output_dir, index):
    """Setup logging for TPU processes"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if index ==0:
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'tpu_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level = logging.INFO,
            format = log_format,
            handlers =[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:

        logging.basicConfig(
            level=logging.ERROR,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    return logging.getLogger(__name__)

class TPUGradientAnalyzer:
    """TPU optimized gradient analyzer that handles distrubuted collection"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.accumulated_gradients = OrderedDict()
        self.losses_per_step = []
        self._initialize_gradient_storage()

    def _initialize_gradient_storage(self):
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.accumulated_gradients[name] = torch.zeros_like(param, device = self.device)
                total_params += param.numel()
        
        xm.master_print(f"Initialized gradient storage for {len(self.accumulated_gradients)}")
        xm.master_print(f"Total parameters being tracked: {total_params:,}")

    def collect_gradients(self, dataloader, index):
        """
        Collect gradients with TPU optimization.
        
        Args:
            dataloader: TPU-wrapped dataloader
            index: TPU core index
        """

        is_master = (index ==0)

        if is_master:
            logger = logging.getLogger(__name__)
            logger.info("Starting gradient collection on TPU")

        self.model.train()

        self.losses_per_step = []
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name].zero_()

        if is_master:
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Collecting gradients")
        else:
            progress_bar = enumerate(dataloader)

        for step, batch in progress_bar:
            self.model.zero_grad()

            #Forward Pass
            outputs = self.model(**batch)
            logits = outputs.logits

            with torch.autocast('xla', enabled=False):
                logits_fp32 = logits.float()
                labels = batch['labels']
                loss = F.cross_entropy(logits_fp32, labels)

            if torch.isnan(loss):
                xm.master_print(f"Warning: NaN loss at step {step}, skipping batch")
                self.model.zero_grad()
                continue

            #backward pass 
            loss.backward()

            self._accumulate_gradients()

            if is_master:
                loss_value = loss.item()
                self.losses_per_step.append(loss_value)
                progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
            
            xm.mark_step()

        self._synchronize_gradients()

        if is_master:
            logger.info(f"Gradient collection completed. Processed {len(dataloader)} batches.")
            logger.info(f"Average loss: {np.mean(self.losses_per_step):.4f}")

        
        return {
            'gradients' : self.accumulated_gradients,
            'losses' : self.losses_per_step,
            'num_batches': len(dataloader)
        }
    
    def _accumulate_gradients(self):
        """Accumulate gradients without updating weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.accumulated_gradients[name]+= param.grad
    
    def _synchronize_gradients(self):
        xm.master_print("Synchronizing gradients across TPU cores")

        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] = xm.all_reduce(
                xm.REDUCE_SUM,
                self.accumulated_gradients[name]
            )

        xm.mark_step()
        xm.master_print("Gradient synchronization complete")

    def analyze_detrimental_parameters(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        xm.master_print(f"Analyzing detrimental parameters with lr={learning_rate}")

        detrimental_params = []
        total_params_analyzed = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Issue where only classifier weights were too dominant
            if 'classifier' in name:
                xm.master_print(f"Skipping classifier layer: {name}")
                continue

            param_data = param.data
            gradient = self.accumulated_gradients[name]

            theoretical_update = param_data - learning_rate * gradient

            original_abs = torch.abs(param_data)
            updated_abs = torch.abs(theoretical_update)

            moving_to_zero_mask = updated_abs < original_abs

            movement_magnitude = original_abs - updated_abs
            movement_magnitude[~moving_to_zero_mask] = 0

            num_detrimental = moving_to_zero_mask.sum().item()
            if num_detrimental > 0:
                detrimental_params.append({
                    'name': name,
                    'shape' : list(param.shape),
                    'total_elements': param.numel(),
                    'num_detrimental': num_detrimental,
                    'detrimental_mask': moving_to_zero_mask.cpu(),
                    'movement_magnitude': movement_magnitude.cpu(),
                    'max_movement': movement_magnitude.max().item(),
                    'mean_movement': movement_magnitude[moving_to_zero_mask].mean().item()
                    
                    })
            
            total_params_analyzed += param.numel()

        total_detrimental = sum(p['num_detrimental'] for p in detrimental_params)

        xm.master_print(f"Analysis complete:")
        xm.master_print(f"  Total parameters analyzed: {total_params_analyzed:,}")
        xm.master_print(f"  Total detrimental parameters: {total_detrimental:,}")
        xm.master_print(f"  Percentage detrimental: {100 * total_detrimental / total_params_analyzed:.2f}%")
        
        return {
            'detrimental_params': detrimental_params,
            'total_params': total_params_analyzed,
            'total_detrimental': total_detrimental,
            'learning_rate': learning_rate
        }
    
def run_on_tpu(index):

        is_master = (index ==0)
        device = xm.xla_device()
        config.use_tpu = True
        config.device = str(device)

        if is_master:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(config.output_dir, f'tpu_experiment_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = None
    
        # Broadcast output_dir to all processes
        output_dir = xm.mesh_reduce('broadcast', output_dir, lambda x: x[0] if x else None)
    
        # Setup logging
        logger = setup_logging(output_dir, index)
    
        if is_master:
            logger.info("="*60)
            logger.info("TPU MASK FINE TUNING")
            logger.info("="*60)
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"TPU Device: {device}")
            logger.info(f"Number of TPU cores: {xm.xrt_world_size()}")

        try:
            xm.rendezvous("data_loading_start")

            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 1: Loading Dataset")
                logger.info("="*40)
            
            train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
            train_dataloader, eval_dataloader = create_dataloaders(
                train_dataset, eval_dataset, config
            )
            
            #Wrap dataloaderf for TPU
            train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
            eval_device_loader = pl.MpDeviceLoader(eval_dataloader, device)

            xm.rendezvous("data_loading_complete")

            if is_master:
                logger.info(f"Train samples: {len(train_dataset)}")
                logger.info(f"Eval samples: {len(eval_dataset)}")
                logger.info(f"Train batches per core: {len(train_dataloader)}")


            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 2: Loading BERT Model")
                logger.info("="*40)

            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_labels
            )

            model.to(device)

            if is_master:
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"Model loaded: {config.model_name}")
                logger.info(f"Total trainable parameters: {total_params:,}")

            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 3: Baseline Evaluation")
                logger.info("="*40)

                evaluator = ModelEvaluator(model, config)
                baseline_metrics = evaluator.evaluate(eval_device_loader, "Baseline Evaluator")

            xm.rendezvous("baseline_complete")
            
            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 4: Collecting Gradients (No Weight Updates)")
                logger.info("="*40)
            
            analyzer = TPUGradientAnalyzer(model, config, device)
            gradient_results = analyzer.collect_gradients(train_device_loader, index)
        
            xm.rendezvous("gradient_collection_complete")

            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 5: Analyzing Detrimental Parameters")
                logger.info("="*40)

                analysis_results = analyzer.analyze_detrimental_parameters()

                logger.info("\n" + "="*40)
                logger.info("STEP 6: Zeroing Top 10% Detrimental Parameters")
                logger.info("="*40)


                model_cpu = model.cpu()
                handler = DetrimentalParameterHandler(model_cpu, config)
                handler.save_original_state()

                zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
                logger.info(handler.get_zeroing_summary({**zeroing_info, 'total_params': total_params}))

                zeroing_stats = handler.zero_parameters(zeroing_info)

                model = model_cpu.to(device)
            
            # Broadcast model parameters to all cores
            for param in model.parameters():
                xm.collective_broadcast([param.data], 0)
            xm.mark_step()

            xm.rendezvous("model_zeroing_complete")

            if not is_master:
                # Non-master cores need to receive the updated model
                # This is handled by XLA's automatic synchronization
                xm.mark_step()

            if is_master:
                logger.info("\n" + "="*40)
                logger.info("STEP 7: Evaluating Modified Model")
                logger.info("="*40)

                modified_metrics = evaluator.evaluate(eval_device_loader, "Modified Model Evaluation")

                logger.info("\n" + "="*40)
                logger.info("STEP 8: Comparing Results")
                logger.info("="*40)

                comparison = evaluator.compare_results(baseline_metrics, modified_metrics)

                logger.info("\nMetric Changes (Modified - Baseline):")
                for metric, changes in comparison['differences'].items():
                    logger.info(f"  {metric}: {changes['absolute']:+.4f} ({changes['percentage']:+.2f}%)")
            
                # Save all results
                all_results = {
                    'config': {
                        'model_name': config.model_name,
                        'train_samples': config.train_samples,
                        'batch_size': config.batch_size,
                        'learning_rate': config.learning_rate,
                        'tpu_cores': xm.xrt_world_size(),
                        'device': 'TPU'
                    },
                    'baseline_evaluation': baseline_metrics,
                    'gradient_collection': {
                        'num_batches': gradient_results['num_batches'],
                        'avg_loss': np.mean(gradient_results['losses']) if gradient_results['losses'] else 0
                    },
                    'detrimental_analysis': {
                        'total_params': analysis_results['total_params'],
                        'total_detrimental': analysis_results['total_detrimental'],
                        'percentage_detrimental': 100.0 * analysis_results['total_detrimental'] / analysis_results['total_params']
                    },
                    'zeroing_info': {
                        'num_zeroed': zeroing_info['num_zeroed'],
                        'percentage_of_total': zeroing_info['percentage_of_total']
                    },
                    'modified_evaluation': modified_metrics,
                    'comparison': comparison
                }
                
                # Save results
                evaluator.save_results(all_results, output_dir)
                evaluator.plot_training_loss(gradient_results['losses'], output_dir)
                
                logger.info("\n" + "="*60)
                logger.info("TPU EXPERIMENT COMPLETE")
                logger.info(f"All results saved to: {output_dir}")
                logger.info("="*60)
            
            xm.rendezvous("experiment_complete")
        
        except Exception as e:
            if is_master:
                logger.error(f"Experiment failed with error: {str(e)}")
                logger.error(traceback.format_exc())
            raise
    
def main():
    if not TPU_AVAILABLE:
        print("ERROR: TPU libraries are not available")
        print("This script must be run on a TPU VM")
        sys.exit(1)

    print(f"Found {xr.world_size()} TPU cores")

    xmp.spawn(run_on_tpu, args =())

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        config.train_samples = 100
        config.batch_size = 8
        print("Running test with 100 samples")

    main()



            

                




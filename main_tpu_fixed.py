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

    if index == 0:
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'tpu_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
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
    """TPU optimized gradient analyzer with loss contribution check"""

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
                self.accumulated_gradients[name] = torch.zeros_like(param, device=self.device)
                total_params += param.numel()
        
        xm.master_print(f"Initialized gradient storage for {len(self.accumulated_gradients)} tensors")
        xm.master_print(f"Total parameters being tracked: {total_params:,}")

    def collect_gradients(self, dataloader, index):
        """Collect gradients with TPU optimization"""
        is_master = (index == 0)

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

            loss.backward()
            self._accumulate_gradients()

            if is_master:
                loss_value = loss.item()
                self.losses_per_step.append(loss_value)
                progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
            
            xm.mark_step()

        # Only synchronize if we're in multi-core mode
        if index == 0 or index is None:
            # Single core mode, no synchronization needed
            xm.master_print("Single core mode - skipping gradient synchronization")
        else:
            self._synchronize_gradients()

        if is_master:
            logger.info(f"Gradient collection completed. Processed {len(dataloader)} batches.")
            logger.info(f"Average loss: {np.mean(self.losses_per_step):.4f}")
        
        return {
            'gradients': self.accumulated_gradients,
            'losses': self.losses_per_step,
            'num_batches': len(dataloader)
        }
    
    def _accumulate_gradients(self):
        """Accumulate gradients without updating weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.accumulated_gradients[name] += param.grad
    
    def _synchronize_gradients(self):
        xm.master_print("Synchronizing gradients across TPU cores")
        for name in self.accumulated_gradients:
            self.accumulated_gradients[name] = xm.all_reduce(
                xm.REDUCE_SUM,
                self.accumulated_gradients[name]
            )
        xm.mark_step()
        xm.master_print("Gradient synchronization complete")

    def analyze_detrimental_parameters_with_loss_check(self, learning_rate=None):
        """Enhanced version that verifies parameters actually harm the model"""
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        xm.master_print(f"Analyzing parameters with loss contribution check (lr={learning_rate})")
        
        detrimental_params = []
        total_params_analyzed = 0
        total_harmful = 0
        total_helpful = 0
        total_neutral = 0
        
        EXCLUDE_PATTERNS = ("classifier.", "pooler.", "LayerNorm", "embeddings")
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(pat in name for pat in EXCLUDE_PATTERNS):
                xm.master_print(f"Skipping critical layer: {name}")
                continue
            
            param_data = param.data
            gradient = self.accumulated_gradients[name]
            
            # Original criterion: moving toward zero
            theoretical_update = param_data - learning_rate * gradient
            original_abs = torch.abs(param_data)
            updated_abs = torch.abs(theoretical_update)
            moving_to_zero_mask = updated_abs < original_abs
            
            # Loss contribution: gradient * parameter
            loss_contribution = gradient * param_data
            
            # Calculate threshold for "small" contribution (bottom 10%)
            abs_contribution = torch.abs(loss_contribution)
            if abs_contribution.numel() > 0:
                contribution_threshold = torch.quantile(abs_contribution, 0.1)
            else:
                contribution_threshold = 0.0
            
            # Classify each parameter
            harmful_mask = moving_to_zero_mask & (loss_contribution > contribution_threshold)
            helpful_mask = moving_to_zero_mask & (loss_contribution < -contribution_threshold)
            neutral_mask = moving_to_zero_mask & (abs_contribution <= contribution_threshold)
            
            # Only mark as detrimental if helpful or neutral (safe to remove)
            truly_detrimental_mask = helpful_mask | neutral_mask
            
            # Calculate movement magnitude for ranking
            movement_magnitude = original_abs - updated_abs
            movement_magnitude[~truly_detrimental_mask] = 0
            
            num_harmful = harmful_mask.sum().item()
            num_helpful = helpful_mask.sum().item()
            num_neutral = neutral_mask.sum().item()
            num_detrimental = truly_detrimental_mask.sum().item()
            
            if num_detrimental > 0:
                detrimental_params.append({
                    'name': name,
                    'shape': list(param.shape),
                    'total_elements': param.numel(),
                    'num_detrimental': num_detrimental,
                    'num_harmful_skipped': num_harmful,
                    'num_helpful': num_helpful,
                    'num_neutral': num_neutral,
                    'detrimental_mask': truly_detrimental_mask.cpu(),
                    'movement_magnitude': movement_magnitude.cpu(),
                    'loss_contribution': loss_contribution.cpu(),
                    'max_movement': movement_magnitude.max().item() if num_detrimental > 0 else 0,
                    'mean_contribution': loss_contribution[truly_detrimental_mask].mean().item() if num_detrimental > 0 else 0
                })
                
                total_harmful += num_harmful
                total_helpful += num_helpful
                total_neutral += num_neutral
            
            total_params_analyzed += param.numel()
        
        total_detrimental = sum(p['num_detrimental'] for p in detrimental_params)
        
        xm.master_print("\n" + "="*60)
        xm.master_print("LOSS CONTRIBUTION ANALYSIS COMPLETE")
        xm.master_print("="*60)
        xm.master_print(f"Total parameters analyzed: {total_params_analyzed:,}")
        xm.master_print(f"Parameters moving toward zero: {total_harmful + total_helpful + total_neutral:,}")
        xm.master_print(f"  ❌ Harmful if removed (skipped): {total_harmful:,}")
        xm.master_print(f"  ✓ Helpful if removed: {total_helpful:,}")
        xm.master_print(f"  ◎ Neutral if removed: {total_neutral:,}")
        xm.master_print(f"Final safe-to-remove count: {total_detrimental:,} ({100 * total_detrimental / total_params_analyzed:.2f}%)")
        
        if total_harmful > 0:
            xm.master_print(f"Safety improvement: Avoided zeroing {total_harmful:,} critical parameters")
        
        return {
            'detrimental_params': detrimental_params,
            'total_params': total_params_analyzed,
            'total_detrimental': total_detrimental,
            'total_harmful_skipped': total_harmful,
            'total_helpful': total_helpful,
            'total_neutral': total_neutral,
            'learning_rate': learning_rate,
            'using_loss_check': True
        }

def run_on_tpu_single_core():
    """Simplified version for single TPU core"""
    index = 0
    is_master = True
    device = xm.xla_device()
    config.use_tpu = True
    config.device = str(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f'tpu_enhanced_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logging(output_dir, index)
    
    logger.info("="*60)
    logger.info("TPU SINGLE CORE TEST - LOSS CONTRIBUTION CHECK")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"TPU Device: {device}")

    try:
        logger.info("\n" + "="*40)
        logger.info("STEP 1: Loading Dataset")
        logger.info("="*40)
        
        train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
        train_dataloader, eval_dataloader = create_dataloaders(
            train_dataset, eval_dataset, config
        )
        
        train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
        eval_device_loader = pl.MpDeviceLoader(eval_dataloader, device)

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")

        logger.info("\n" + "="*40)
        logger.info("STEP 2: Loading BERT Model")
        logger.info("="*40)

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {config.model_name}")
        logger.info(f"Total trainable parameters: {total_params:,}")

        logger.info("\n" + "="*40)
        logger.info("STEP 3: Baseline Evaluation")
        logger.info("="*40)

        evaluator = ModelEvaluator(model, config)
        baseline_metrics = evaluator.evaluate(eval_device_loader, "Baseline Evaluation")
        
        logger.info("\n" + "="*40)
        logger.info("STEP 4: Collecting Gradients")
        logger.info("="*40)
        
        analyzer = TPUGradientAnalyzer(model, config, device)
        gradient_results = analyzer.collect_gradients(train_device_loader, index)

        logger.info("\n" + "="*40)
        logger.info("STEP 5: Analyzing with Loss Contribution Check")
        logger.info("="*40)

        analysis_results = analyzer.analyze_detrimental_parameters_with_loss_check()

        logger.info("\n" + "="*40)
        logger.info("STEP 6: Zeroing Safe Parameters")
        logger.info("="*40)

        model_cpu = model.cpu()
        handler = DetrimentalParameterHandler(model_cpu, config)
        handler.save_original_state()

        zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
        logger.info(handler.get_zeroing_summary({**zeroing_info, 'total_params': total_params}))

        zeroing_stats = handler.zero_parameters(zeroing_info)
        model = model_cpu.to(device)

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
        
        logger.info("\n" + "="*60)
        logger.info("SINGLE CORE TEST COMPLETE")
        logger.info(f"All results saved to: {output_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    if not TPU_AVAILABLE:
        print("ERROR: TPU libraries are not available")
        print("This script must be run on a TPU VM")
        sys.exit(1)

    print("Running single-core TPU test...")
    run_on_tpu_single_core()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        config.train_samples = 100
        config.batch_size = 8
        print("Running test with 100 samples")
    
    main()

import torch
import logging
import sys
import os
from datetime import datetime
import json
import traceback

from config import config
from data_utils import load_and_prepare_dataset, create_dataloaders
from gradient_analyzer import GradientAnalyzer
from detrimental_params import DetrimentalParameterHandler
from evaluation import ModelEvaluator
from transformers import AutoModelForSequenceClassification

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level = logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def main():
    """Main experiment workflow."""
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f'experiment_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("MASK FINE TUNING EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    config.device = str(device)

    try:
        # ============================================================
        # STEP 1: Load Data
        # ============================================================

        logger.info("\n" +"="*40)
        logger.info("Step 1: Loading Dataset")
        logger.info("="*40)

        train_dataset, eval_dataset, tokenizer = load_and_prepare_dataset(config)
        train_dataloader, eval_dataloader = create_dataloaders(
            train_dataset, eval_dataset, config
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        logger.info(f"Train batches: {len(train_dataloader)}")
        logger.info(f"Eval batches: {len(eval_dataloader)}")

        # ============================================================
        # STEP 2: Load Model
        # ============================================================
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

        # ============================================================
        # STEP 3: Baseline Evaluation
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 3: Baseline Evaluation")
        logger.info("="*40)

        evaluator = ModelEvaluator(model, config)
        baseline_metrics = evaluator.evaluate(eval_dataloader, "Baseline Evaluation")

        # ============================================================
        # STEP 4: Gradient Collection
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 4: Collecting Gradients (No Weight Updates)")
        logger.info("="*40)

        analyzer = GradientAnalyzer(model, config)
        gradient_results = analyzer.collect_gradients(train_dataloader)

        # Plot training loss
        evaluator.plot_training_loss(gradient_results['losses'], output_dir)

        #Gradient statistics
        gradient_stats = analyzer.get_gradient_statistics()
        logger.info(f"Collected gradients for {len(gradient_stats)} parameter tensors")

        # ============================================================
        # STEP 5: Analyze Detrimental Parameters
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 5: Analyzing Detrimental Parameters")
        logger.info("="*40)

        analysis_results = analyzer.analyze_detrimental_parameters(
            learning_rate = config.learning_rate
        )

        # ============================================================
        # STEP 6: Zero Top 10% Detrimental Parameters
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 6: Zeroing Top 10% Detrimental Parameters")
        logger.info("="*40)

        handler = DetrimentalParameterHandler(model, config)
        handler.save_original_state()

        zeroing_info = handler.identify_top_detrimental_parameters(analysis_results)
        logger.info(handler.get_zeroing_summary({**zeroing_info, 'total_params' : total_params}))

        zeroing_stats = handler.zero_parameters(zeroing_info)

        # ============================================================
        # STEP 7: Evaluate Modified Model
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 7: Evaluating Modified Model")
        logger.info("="*40)

        modified_metrics = evaluator.evaluate(eval_dataloader, "Modified Model Evaluation")

        # ============================================================
        # STEP 8: Compare Results
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 8: Comparing Results")
        logger.info("="*40)

        comparison = evaluator.compare_results(baseline_metrics, modified_metrics)

        logger.info("\nMetric Changes (Modified - Baseline):")
        for metric, changes in comparison['differences'].items():
            logger.info(f" {metric}: {changes['absolute']:+.4f} ({changes['percentage']:+.2f}%)")
        
        # ============================================================
        # STEP 9: Save All Results
        # ============================================================
        logger.info("\n" + "="*40)
        logger.info("STEP 9: Saving Results")
        logger.info("="*40)

        #Compile all results
        all_results = {
            'config': {
                'model_name': config.model_name,
                'train_samples': config.train_samples,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'device': config.device
            }, 
            'baseline_evaluation': baseline_metrics,
            'gradient_collection': {
                'num_batches': gradient_results['num_batches'],
                'avg_loss': sum(gradient_results['losses']) / len(gradient_results['losses']),
                'min_loss': min(gradient_results['losses']),
                'max_loss': max(gradient_results['losses'])
            },
            'detrimental_analysis':{
                'total_params': analysis_results['total_params'],
                'total_detrimental': analysis_results['total_detrimental'],
                'percentage_detrimental': 100.0 * analysis_results['total_detrimental'] / analysis_results['total_params']
            },
            'zeroing_info': {
                'num_zeroed': zeroing_info['num_zeroed'],
                'percentage_of_total': zeroing_info['percentage_of_total'],
                'affected_layers': zeroing_stats['affected_layers'][:10]
            },
            'modified_evaluation': modified_metrics,
            'comparison': comparison
        }

        evaluator.save_results(all_results, output_dir)

        if config.save_gradients:
            grad_stats_path = os.path.join(output_dir, 'gradient_statistics.json')
            with open(grad_stats_path, 'w') as f:
                json.dump(gradient_stats, f, indent =2)
            logger.info(f"Gradient statistics saved to {grad_stats_path}")

        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info(f"All results saved to: {output_dir}")
        logger.info("="*60)

        return all_results
    
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        config.train_samples = 100
        config.batch_size = 10
        print("Running in test mode with 100 samples")

    results = main()




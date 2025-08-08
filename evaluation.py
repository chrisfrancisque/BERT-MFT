import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
import json
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Performance Evaluation on validation set"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

    
    def evaluate(self, dataloader, phase_name="Evaluation"):
        """
        Evaluate model on given dataloader.
        
        Args:
            dataloader: DataLoader with evaluation samples
            phase_name: Name for this evaluation phase (e.g., "Baseline", "After Masking")
            
        Returns:
            Dict containing metrics
        """
        logger.info(f"Starting {phase_name}...")

        self.model.eval()

        all_predicitions = []
        all_labels = []
        all_proba =[]
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=phase_name)

            for batch in progress_bar:
                batch = {k: v.to(self.device) for k,v in batch.items()}

                #Forward Pass
                outputs = self.model(**batch)
                logits = outputs.logits

                # Compute loss in float 32 for stability
                with torch.autocast(self.device.type, enabled=False):
                    logits_fp32 = logits.float()
                    labels = batch['labels']
                    loss = F.cross_entropy(logits_fp32, labels)

                if torch.isnan(loss):
                    logger.warning(f"NaN loss in evaluation, skipping batch")
                    continue

                total_loss += loss.item()
                num_batches += 1

                #Get predictions and probabilities
                proba = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                # Collect results
                all_predicitions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_proba.extend(proba.cpu().numpy())

                #Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = self._calculate_metrics(
            np.array(all_predicitions),
            np.array(all_labels),
            np.array(all_proba)
        )

        metrics['avg_loss'] = total_loss / num_batches if num_batches > 0 else 0
        metrics['phase'] = phase_name

        logger.info(f"{phase_name} Results:")
        logger.info(f"  Loss: {metrics['avg_loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions, labels, probabilities):
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted class labels
            labels: True class labels
            probabilities: Prediction probabilities
            
        Returns:
            Dict of metrics
        """

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        try:
            if len(np.unique(labels)) > 1:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
            else:
                roc_auc = None
                logger.warning("Cannot compute ROC-AUC: only one class present in labels")
        except Exception as e:
            logger.warning(f"Error computing ROC-AUC: {e}")
            roc_auc = None
        
        return{
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'num_samples': len(labels)
        }
    
    def compare_results(self, baseline_metrics, modified_metrics):
        """
        Compare baseline and modified model metrics.
        
        Args:
            baseline_metrics: Metrics from original model
            modified_metrics: Metrics after parameter zeroing
            
        Returns:
            Dict with comparison
        """

        comparison = {
            'baseline': baseline_metrics,
            'modified': modified_metrics,
            'differences': {}
        }

        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'avg_loss']:
            if key in baseline_metrics and key in modified_metrics:
                baseline_val = baseline_metrics[key]
                modified_val = modified_metrics[key]

                if baseline_val is not None and modified_val is not None:
                    diff = modified_val - baseline_val
                    pct_change = (diff/ baseline_val * 100) if baseline_val != 0 else 0

                    comparison['differences'][key] = {
                        'absolute': diff,
                        'percentage': pct_change
                    }
        return comparison
    
    def plot_training_loss(self, losses, output_dir):
        """
        Plot training loss curve.
        
        Args:
            losses: List of loss values per step
            output_dir: Directory to save plot
        """
        
        plt.figure(figsize=(10,6))
        steps = list(range(1, len(losses) + 1))

        plt.plot(steps, losses, 'b-', linewidth =2, alpha = 0.7)
        plt.scatter(steps, losses, c='blue', s=20, alpha=0.5)

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss per Step', fontsize = 14)
        plt.grid(True, alpha = 0.3)

        #Add statistics
        avg_loss = np.mean(losses)
        plt.axhline(y = avg_loss, color='r', linestyle = '--', alpha=0.5, label=f'Avg: {avg_loss:.4f}')
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f"Training loss plot saved to {save_path}")

    def save_results(self, all_results, output_dir):
        """
        Save all evaluation results to files.
        
        Args:
            all_results: Dict containing all experiment results
            output_dir: Directory to save results
        """

        #Save as JSON
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types to python tpes for JSON serialization
            json_safe_results = self._make_json_serializable(all_results)
            json.dump(json_safe_results, f, indent = 2)

        logger.info(f"Results saved to {json_path}")

        #Create readable summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self._generate_summary(all_results))
        
        logger.info(f"Summary saved to {summary_path}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _generate_summary(self, results):
        """Generate human-readable summary of results."""
        summary = [
            "="*60,
            "GRADIENT ANALYSIS EXPERIMENT RESULTS",
            "="*60,
            "",
            "BASELINE EVALUATION:",
            "-"*40
        ]
        
        # Baseline metrics
        baseline = results.get('baseline_evaluation', {})
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if key in baseline:
                value = baseline[key]
                if value is not None:
                    summary.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        summary.extend([
            "",
            "AFTER PARAMETER ZEROING:",
            "-"*40
        ])
        
        # Modified metrics
        modified = results.get('modified_evaluation', {})
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if key in modified:
                value = modified[key]
                if value is not None:
                    summary.append(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        # Comparison
        if 'comparison' in results:
            summary.extend([
                "",
                "COMPARISON (Modified - Baseline):",
                "-"*40
            ])
            
            diffs = results['comparison'].get('differences', {})
            for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                if key in diffs:
                    abs_diff = diffs[key]['absolute']
                    pct_diff = diffs[key]['percentage']
                    summary.append(f"  {key.replace('_', ' ').title()}: {abs_diff:+.4f} ({pct_diff:+.2f}%)")
        
        summary.append("="*60)
        
        return "\n".join(summary)

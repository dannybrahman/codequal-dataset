"""
Model Evaluator

Evaluates trained models against human test set.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.base_model import BaseQualityModel
from .metrics import compute_agreement_metrics, format_metrics_table

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate trained model on human-annotated test set.

    Computes same metrics as LLM-human agreement analysis for comparison.
    """

    def __init__(
        self,
        model: BaseQualityModel,
        test_loader: DataLoader,
        device: Optional[str] = None,
        output_dir: str = "generated/results"
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model to evaluate
            test_loader: Test data loader (with human median scores)
            device: Device for evaluation
            output_dir: Directory for saving results
        """
        self.model = model
        self.test_loader = test_loader

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluator initialized on device: {self.device}")

    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dict with evaluation metrics (overall, per-dimension, per-source)
        """
        logger.info("Evaluating model on test set...")

        all_predictions = []
        all_targets = []
        all_sources = []

        with torch.no_grad():
            for batch_features, batch_targets, batch_sources in tqdm(self.test_loader, desc="Evaluating"):
                # Move to device
                batch_targets = batch_targets.to(self.device)

                if isinstance(batch_features, dict):
                    batch_features = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_features.items()
                    }
                else:
                    batch_features = batch_features.to(self.device)

                # Predict
                predictions = self.model(batch_features)

                # Collect results
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())
                all_sources.extend(batch_sources)

        # Concatenate all batches
        predictions = np.vstack(all_predictions)  # [n_samples, 5]
        targets = np.vstack(all_targets)  # [n_samples, 5]

        # Compute overall metrics
        metrics = compute_agreement_metrics(predictions, targets)

        # Compute per-source metrics
        unique_sources = sorted(set(all_sources))
        metrics['per_source'] = {}

        for source in unique_sources:
            # Get indices for this source
            source_indices = [i for i, s in enumerate(all_sources) if s == source]
            source_preds = predictions[source_indices]
            source_targets = targets[source_indices]

            # Compute metrics for this source
            source_metrics = compute_agreement_metrics(source_preds, source_targets)
            metrics['per_source'][source] = source_metrics

        logger.info(f"\nEvaluation complete!")
        logger.info(f"Spearman r (overall): {metrics['overall']['spearman_r']:.4f}")
        logger.info(f"MAE (overall): {metrics['overall']['mae']:.4f}")

        # Log per-source summary
        logger.info(f"\nPer-source Spearman r:")
        for source in unique_sources:
            spearman = metrics['per_source'][source]['overall']['spearman_r']
            n = metrics['per_source'][source]['n_samples']
            logger.info(f"  {source}: {spearman:.4f} (n={n})")

        return metrics

    def evaluate_and_save(self, filename: str = "evaluation_results.json") -> Dict:
        """
        Evaluate and save results.

        Args:
            filename: Output filename

        Returns:
            Evaluation metrics
        """
        metrics = self.evaluate()

        # Save to JSON
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Also save formatted text
        text_path = self.output_dir / filename.replace('.json', '.txt')
        with open(text_path, 'w') as f:
            f.write(format_metrics_table(metrics))

        logger.info(f"Formatted results saved to: {text_path}")

        # Print summary
        print("\n" + format_metrics_table(metrics))

        return metrics

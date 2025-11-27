"""
Base Model Class

Abstract base class for code quality prediction models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn


class BaseQualityModel(nn.Module, ABC):
    """
    Abstract base class for quality prediction models.

    All models predict 5-dimensional quality scores:
    - Functionality
    - Readability
    - Idiomatic
    - Error Handling
    - Efficiency
    """

    NUM_DIMENSIONS = 5
    DIMENSION_NAMES = ['functionality', 'readability', 'idiomatic',
                      'error_handling', 'efficiency']

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Dict of input tensors (format depends on model)

        Returns:
            Tensor of shape [batch_size, 5] with predicted quality scores
        """
        pass

    def predict(self, inputs: Dict) -> torch.Tensor:
        """
        Predict quality scores (inference mode).

        Args:
            inputs: Dict of input tensors

        Returns:
            Tensor of shape [batch_size, 5] with predicted scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss and metrics.

        Args:
            predictions: Predicted scores [batch_size, 5]
            targets: Ground truth scores [batch_size, 5]
            loss_fn: Loss function

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Overall loss
        total_loss = loss_fn(predictions, targets)

        # Per-dimension losses for monitoring
        metrics = {'total_loss': total_loss.item()}

        for i, dim_name in enumerate(self.DIMENSION_NAMES):
            dim_loss = loss_fn(predictions[:, i], targets[:, i])
            metrics[f'{dim_name}_loss'] = dim_loss.item()

        return total_loss, metrics

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, **kwargs):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
            **kwargs
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    @abstractmethod
    def get_config(self) -> Dict:
        """
        Get model configuration.

        Returns:
            Dict with model hyperparameters
        """
        pass

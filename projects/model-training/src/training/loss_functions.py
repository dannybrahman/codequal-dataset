"""
Loss Functions

Custom loss functions for multi-output quality prediction.
"""

import torch
import torch.nn as nn


class QualityLoss(nn.Module):
    """
    Multi-output loss for 5-dimensional quality prediction.

    Supports different loss types and dimension weighting.
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        dimension_weights: dict = None
    ):
        """
        Initialize quality loss.

        Args:
            loss_type: Type of loss ('mse', 'huber', 'mae')
            dimension_weights: Optional weights for each dimension
        """
        super().__init__()

        self.loss_type = loss_type

        # Initialize base loss function
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'huber':
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Set dimension weights (default: equal weights)
        if dimension_weights is None:
            self.dimension_weights = torch.ones(5) / 5.0  # Equal weights
        else:
            # Convert dict to tensor
            dim_names = ['functionality', 'readability', 'idiomatic',
                        'error_handling', 'efficiency']
            weights = [dimension_weights.get(name, 1.0) for name in dim_names]
            self.dimension_weights = torch.tensor(weights)
            self.dimension_weights = self.dimension_weights / self.dimension_weights.sum()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss across all dimensions.

        Args:
            predictions: Predicted scores [batch_size, 5]
            targets: Ground truth scores [batch_size, 5]

        Returns:
            Scalar loss value
        """
        # Compute per-element loss [batch_size, 5]
        element_loss = self.base_loss(predictions, targets)

        # Apply dimension weights
        weights = self.dimension_weights.to(predictions.device)
        weighted_loss = element_loss * weights.unsqueeze(0)

        # Average across batch and dimensions
        total_loss = weighted_loss.mean()

        return total_loss

    def compute_per_dimension_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Compute loss for each dimension separately.

        Args:
            predictions: Predicted scores [batch_size, 5]
            targets: Ground truth scores [batch_size, 5]

        Returns:
            Dict mapping dimension names to loss values
        """
        dim_names = ['functionality', 'readability', 'idiomatic',
                    'error_handling', 'efficiency']

        element_loss = self.base_loss(predictions, targets)

        per_dim_loss = {}
        for i, name in enumerate(dim_names):
            per_dim_loss[name] = element_loss[:, i].mean().item()

        return per_dim_loss

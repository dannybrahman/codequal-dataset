"""
Simple MLP Model

Lightweight baseline model using simple code features.
"""

import logging
from typing import Dict
import torch
import torch.nn as nn

from .base_model import BaseQualityModel

logger = logging.getLogger(__name__)


class SimpleMLP(BaseQualityModel):
    """
    Simple multi-layer perceptron for quality prediction.

    Uses simple statistical features (LOC, complexity, etc.)
    as input instead of learned embeddings.

    Much faster than CodeBERT but potentially less accurate.
    """

    def __init__(
        self,
        input_dim: int = 21,  # From simple feature extractor
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize simple MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.NUM_DIMENSIONS))

        self.mlp = nn.Sequential(*layers)

        logger.info(f"Simple MLP initialized with {self.get_num_parameters():,} parameters")

    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Dict with 'features' tensor [batch_size, input_dim]

        Returns:
            Predicted quality scores [batch_size, 5]
        """
        features = inputs['features']

        # Forward through MLP
        quality_scores = self.mlp(features)

        # Clip to valid range [1, 5]
        quality_scores = torch.clamp(quality_scores, 1.0, 5.0)

        return quality_scores

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'model_type': 'mlp',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout_prob
        }

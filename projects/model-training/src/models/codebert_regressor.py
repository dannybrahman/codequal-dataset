"""
CodeBERT Regressor

Fine-tunes CodeBERT for 5-dimensional quality prediction.
"""

import logging
from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel

from .base_model import BaseQualityModel

logger = logging.getLogger(__name__)


class CodeBERTRegressor(BaseQualityModel):
    """
    CodeBERT-based quality prediction model.

    Fine-tunes microsoft/codebert-base with a regression head
    for 5-dimensional quality score prediction.
    """

    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base',
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Initialize CodeBERT regressor.

        Args:
            model_name: Pretrained model name
            hidden_dim: Hidden layer dimension for regression head
            dropout: Dropout probability
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.freeze_encoder = freeze_encoder

        # Load pretrained CodeBERT
        logger.info(f"Loading {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing encoder weights")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head
        encoder_dim = self.encoder.config.hidden_size

        self.regression_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.NUM_DIMENSIONS)
        )

        logger.info(f"Model initialized with {self.get_num_parameters():,} parameters")

    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Dict with 'embeddings' or 'input_ids' and 'attention_mask'

        Returns:
            Predicted quality scores [batch_size, 5]
        """
        # Get code embeddings
        if 'embeddings' in inputs:
            # Pre-computed embeddings
            code_embedding = inputs['embeddings']
        else:
            # Compute embeddings from input_ids
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Use [CLS] token embedding
            code_embedding = outputs.last_hidden_state[:, 0, :]

        # Predict quality scores
        quality_scores = self.regression_head(code_embedding)

        # Only clip during inference (not training) to preserve gradients
        if not self.training:
            quality_scores = torch.clamp(quality_scores, 1.0, 5.0)

        return quality_scores

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'model_type': 'codebert',
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_prob,
            'freeze_encoder': self.freeze_encoder
        }

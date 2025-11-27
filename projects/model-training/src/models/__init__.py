"""
Model architectures for code quality prediction.
"""

from .base_model import BaseQualityModel
from .codebert_regressor import CodeBERTRegressor
from .simple_mlp import SimpleMLP

__all__ = [
    'BaseQualityModel',
    'CodeBERTRegressor',
    'SimpleMLP'
]


def get_model(model_type: str, **kwargs):
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('codebert', 'mlp')
        **kwargs: Model-specific parameters

    Returns:
        Initialized model
    """
    if model_type == 'codebert':
        return CodeBERTRegressor(**kwargs)
    elif model_type == 'mlp':
        return SimpleMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

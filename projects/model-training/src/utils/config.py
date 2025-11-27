"""
Configuration Management
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_type: str = 'codebert'  # 'codebert' or 'mlp'
    sources: List[str] = None
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    device: Optional[str] = None
    output_dir: str = 'generated/models'

    def __post_init__(self):
        if self.sources is None:
            self.sources = ['codenet', 'codeeval']


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    model_path: str
    sources: List[str] = None
    batch_size: int = 32
    device: Optional[str] = None
    output_dir: str = 'generated/results'

    def __post_init__(self):
        if self.sources is None:
            self.sources = ['codenet', 'codeeval']

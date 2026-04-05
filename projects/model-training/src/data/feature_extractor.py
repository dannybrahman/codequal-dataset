"""
Code Feature Extractor

Extracts features from code for model training.
Supports CodeBERT embeddings and simple statistical features.
"""

import logging
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class CodeFeatureExtractor:
    """
    Extract features from code samples.

    Supports multiple feature extraction strategies:
    - CodeBERT embeddings (transformer-based)
    - Simple statistical features (LOC, complexity, etc.)
    """

    def __init__(
        self,
        method: str = 'codebert',
        model_name: str = 'microsoft/codebert-base',
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize feature extractor.

        Args:
            method: Extraction method ('codebert', 'simple', or 'hybrid')
            model_name: Pretrained model name for embeddings
            max_length: Maximum token length
            device: Device for model ('cuda', 'cpu', or None for auto)
        """
        self.method = method
        self.max_length = max_length

        # Set device with MPS support for Apple Silicon
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Initialize tokenizer for CodeBERT (model is loaded in regressor for fine-tuning)
        if method in ['codebert', 'hybrid']:
            logger.info(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = None  # Model loaded in regressor for gradient flow
            logger.info("Tokenizer loaded successfully")
        else:
            self.tokenizer = None
            self.model = None

    def __call__(self, features: Dict) -> Dict:
        """
        Extract features from code sample.

        Args:
            features: Dict with 'code', 'problem', 'language', 'source'

        Returns:
            Dict with extracted features
        """
        if self.method == 'codebert':
            return self._extract_codebert(features)
        elif self.method == 'simple':
            return self._extract_simple(features)
        elif self.method == 'hybrid':
            codebert_features = self._extract_codebert(features)
            simple_features = self._extract_simple(features)
            # Combine features
            codebert_features['simple_features'] = simple_features['features']
            return codebert_features
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_codebert(self, features: Dict) -> Dict:
        """Extract CodeBERT token inputs for fine-tuning.

        Returns tokenized inputs (not embeddings) so the model can
        fine-tune the encoder during training.
        """
        code = features['code']

        # Tokenize only - don't compute embeddings
        # Model will compute embeddings during forward pass for gradient flow
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

    def _extract_simple(self, features: Dict) -> Dict:
        """Extract simple statistical features."""
        code = features['code']

        # Basic code metrics
        lines = code.split('\n')
        loc = len([line for line in lines if line.strip()])
        total_lines = len(lines)
        blank_lines = total_lines - loc
        avg_line_length = sum(len(line) for line in lines) / max(total_lines, 1)

        # Character counts
        char_count = len(code)
        alpha_count = sum(c.isalpha() for c in code)
        digit_count = sum(c.isdigit() for c in code)
        space_count = sum(c.isspace() for c in code)

        # Simple complexity indicators
        indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        max_indent = max(indent_levels) if indent_levels else 0
        avg_indent = sum(indent_levels) / max(len(indent_levels), 1)

        # Keyword counts (Python-specific)
        python_keywords = ['def', 'class', 'if', 'for', 'while', 'try', 'except',
                          'import', 'from', 'return', 'yield']
        keyword_counts = {kw: code.count(kw) for kw in python_keywords}

        # Combine into feature vector
        feature_vector = torch.tensor([
            loc,
            total_lines,
            blank_lines,
            avg_line_length,
            char_count,
            alpha_count,
            digit_count,
            space_count,
            max_indent,
            avg_indent,
        ] + list(keyword_counts.values()), dtype=torch.float32)

        return {
            'features': feature_vector,
            'feature_names': [
                'loc', 'total_lines', 'blank_lines', 'avg_line_length',
                'char_count', 'alpha_count', 'digit_count', 'space_count',
                'max_indent', 'avg_indent'
            ] + [f'keyword_{kw}' for kw in python_keywords]
        }

    def get_embedding_dim(self) -> int:
        """Get dimension of extracted embeddings."""
        if self.method == 'codebert':
            return 768  # CodeBERT hidden size
        elif self.method == 'simple':
            return 10 + 11  # Basic metrics + keyword counts
        elif self.method == 'hybrid':
            return 768 + 21  # CodeBERT + simple features
        else:
            raise ValueError(f"Unknown method: {self.method}")

"""
Dataset Loader

Loads code quality datasets from ../dataset/generated/integrated/
- Train/Valid: Single LLM score per sample
- Test: Multiple human scores → median aggregation per dimension
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import statistics

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class QualitySample:
    """Single code quality sample with scores."""
    problem_id: str
    submission_id: str
    problem: str
    submission: str
    language: str
    source: str

    # Quality scores (1-5 scale)
    functionality: Optional[float] = None
    readability: Optional[float] = None
    idiomatic: Optional[float] = None
    error_handling: Optional[float] = None
    efficiency: Optional[float] = None

    # Metadata
    score_type: str = 'unknown'  # 'llm' or 'human_median'
    llm_model: Optional[str] = None


class QualityDatasetLoader:
    """
    Load quality datasets from the dataset project's integrated folder.

    Data path: ../dataset/generated/integrated/{source}/{split}.jsonl

    Train/Valid splits: Single LLM score per sample
    Test split: Multiple human scores with configurable aggregation
    """

    QUALITY_DIMENSIONS = ['functionality', 'readability', 'idiomatic',
                         'error_handling', 'efficiency']

    def __init__(self, dataset_root: str = "../dataset/generated/integrated",
                 human_score_aggregation: str = "median"):
        """
        Initialize dataset loader.

        Args:
            dataset_root: Path to integrated datasets (relative to this project)
            human_score_aggregation: Aggregation method for human scores ('mean' or 'median')
        """
        self.dataset_root = Path(__file__).parent.parent.parent / dataset_root
        self.human_score_aggregation = human_score_aggregation

        if human_score_aggregation not in ('mean', 'median'):
            raise ValueError(f"human_score_aggregation must be 'mean' or 'median', got: {human_score_aggregation}")

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {self.dataset_root}")

        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Human score aggregation: {self.human_score_aggregation}")

    def load_split(self, source: str, split: str) -> List[QualitySample]:
        """
        Load a specific split from a source dataset.

        Args:
            source: Dataset source (e.g., 'codenet', 'codeeval')
            split: Data split ('train', 'valid', 'test')

        Returns:
            List of quality samples with appropriate score aggregation
        """
        split_file = self.dataset_root / source / f"{split}.jsonl"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        samples = []
        skipped = 0

        with open(split_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    sample = self._parse_sample(data, source, split)

                    if sample is not None:
                        samples.append(sample)
                    else:
                        skipped += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num} in {split_file}: {e}")
                    skipped += 1
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num} in {split_file}: {e}")
                    skipped += 1

        logger.info(f"Loaded {len(samples)} samples from {source}/{split} (skipped {skipped})")
        return samples

    def _parse_sample(self, data: Dict, source: str, split: str) -> Optional[QualitySample]:
        """
        Parse a sample from JSON data.

        Train/Valid: Extract single LLM score
        Test: Compute median across human annotators
        """

        # Extract required fields
        try:
            problem_id = data.get('problem_id', '')
            submission_id = data.get('submission_id', data.get('sample_id', ''))
            problem = data.get('problem', '')
            submission = data.get('submission', '')
            language = data.get('language', 'python')

            if not problem or not submission:
                return None

        except Exception as e:
            logger.debug(f"Missing required fields: {e}")
            return None

        # Initialize sample
        sample = QualitySample(
            problem_id=problem_id,
            submission_id=submission_id,
            problem=problem,
            submission=submission,
            language=language,
            source=source
        )

        # Extract quality scores based on split type
        if split in ['train', 'valid']:
            # Train/Valid: Single LLM score
            success = self._extract_llm_scores(sample, data)
        else:  # test
            # Test: Aggregated human scores
            success = self._extract_human_scores(sample, data)

        if not success:
            return None

        return sample

    def _extract_llm_scores(self, sample: QualitySample, data: Dict) -> bool:
        """
        Extract single LLM score for train/valid splits.

        Handles two formats:
        1. List format: llm_scores = [{model_name: "...", scores: {...}}, ...]
        2. Dict format: llm_scores = {functionality: 3, ...} or quality_scores = {...}

        Returns:
            True if scores extracted successfully
        """
        llm_scores = data.get('llm_scores', [])
        quality_scores = data.get('quality_scores', {})

        scores_dict = None
        llm_model = 'unknown'

        # Handle list format (new format with model entries)
        if isinstance(llm_scores, list) and len(llm_scores) > 0:
            # Use the first model's scores (could be enhanced to select specific model)
            first_entry = llm_scores[0]
            if isinstance(first_entry, dict):
                scores_dict = first_entry.get('scores', {})
                llm_model = first_entry.get('model_name', 'unknown')

        # Handle dict format (legacy format)
        elif isinstance(llm_scores, dict) and llm_scores:
            scores_dict = llm_scores

        # Fallback to quality_scores
        elif quality_scores and isinstance(quality_scores, dict):
            scores_dict = quality_scores

        if not scores_dict:
            logger.debug(f"No LLM scores found for {sample.submission_id}")
            return False

        # Extract scores for each dimension
        for dim in self.QUALITY_DIMENSIONS:
            if dim in scores_dict and scores_dict[dim] is not None:
                setattr(sample, dim, float(scores_dict[dim]))

        # Validate at least some scores exist
        has_scores = any(getattr(sample, dim) is not None
                        for dim in self.QUALITY_DIMENSIONS)

        if has_scores:
            sample.score_type = 'llm'
            sample.llm_model = llm_model

        return has_scores

    def _extract_human_scores(self, sample: QualitySample, data: Dict) -> bool:
        """
        Extract and aggregate human scores for test split.

        Expects: data['human_scores'] = [
            {annotator_id: "...", scores: {functionality: 3, ...}},
            {annotator_id: "...", scores: {functionality: 4, ...}},
            ...
        ]

        Returns:
            True if scores extracted successfully
        """
        human_scores = data.get('human_scores', [])

        if not human_scores:
            logger.debug(f"No human scores found for {sample.submission_id}")
            return False

        # Aggregate scores for each dimension
        for dim in self.QUALITY_DIMENSIONS:
            dim_scores = []
            for annotator_entry in human_scores:
                scores_dict = annotator_entry.get('scores', {})
                if dim in scores_dict and scores_dict[dim] is not None:
                    dim_scores.append(float(scores_dict[dim]))

            # Aggregate scores
            if dim_scores:
                if self.human_score_aggregation == 'median':
                    agg_score = statistics.median(dim_scores)
                else:  # mean
                    agg_score = statistics.mean(dim_scores)
                setattr(sample, dim, agg_score)

        # Validate at least some scores exist
        has_scores = any(getattr(sample, dim) is not None
                        for dim in self.QUALITY_DIMENSIONS)

        if has_scores:
            sample.score_type = f'human_{self.human_score_aggregation}'

        return has_scores

    def load_multi_source(self, sources: List[str], split: str) -> List[QualitySample]:
        """
        Load from multiple sources.

        Args:
            sources: List of dataset sources
            split: Data split

        Returns:
            Combined list of samples from all sources
        """
        all_samples = []

        for source in sources:
            try:
                samples = self.load_split(source, split)
                all_samples.extend(samples)
            except FileNotFoundError:
                logger.warning(f"Skipping {source}/{split} - file not found")

        logger.info(f"Loaded {len(all_samples)} total samples from {len(sources)} sources")
        return all_samples

    def get_available_sources(self) -> List[str]:
        """Get list of available dataset sources."""
        sources = []
        for path in self.dataset_root.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                sources.append(path.name)
        return sorted(sources)


class QualityDataset(Dataset):
    """
    PyTorch Dataset for code quality samples.

    Wraps loaded samples for use with PyTorch DataLoader.
    """

    QUALITY_DIMENSIONS = ['functionality', 'readability', 'idiomatic',
                         'error_handling', 'efficiency']

    def __init__(self, samples: List[QualitySample], feature_extractor=None):
        """
        Initialize dataset.

        Args:
            samples: List of quality samples
            feature_extractor: Optional feature extractor for code
        """
        self.samples = samples
        self.feature_extractor = feature_extractor

        # Filter samples with complete scores
        self.valid_samples = [
            s for s in samples
            if all(getattr(s, dim) is not None for dim in self.QUALITY_DIMENSIONS)
        ]

        logger.info(f"Dataset: {len(self.valid_samples)}/{len(samples)} samples with complete scores")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[Dict, torch.Tensor, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (features_dict, targets_tensor, source_name)
        """
        sample = self.valid_samples[idx]

        # Extract features (code text or embeddings)
        features = {
            'code': sample.submission,
            'problem': sample.problem,
            'language': sample.language,
            'source': sample.source
        }

        # Apply feature extractor if provided
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)

        # Extract targets (5D quality scores)
        targets = torch.tensor([
            getattr(sample, dim)
            for dim in self.QUALITY_DIMENSIONS
        ], dtype=torch.float32)

        return features, targets, sample.source

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        import numpy as np

        stats = {
            'total_samples': len(self.valid_samples),
            'score_types': {}
        }

        # Count score types
        for sample in self.valid_samples:
            score_type = sample.score_type
            stats['score_types'][score_type] = stats['score_types'].get(score_type, 0) + 1

        # Statistics per dimension
        for dim in self.QUALITY_DIMENSIONS:
            scores = [getattr(s, dim) for s in self.valid_samples
                     if getattr(s, dim) is not None]

            if scores:
                stats[dim] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores),
                    'count': len(scores)
                }

        return stats

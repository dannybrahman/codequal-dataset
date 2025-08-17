"""
Data loader for CodeQual datasets.

Handles loading and filtering of integrated datasets with support for
multiple splits and various filtering criteria.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random


logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Represents a single dataset sample with metadata."""
    problem_id: str
    submission_id: str
    problem: str
    submission: str
    source: str
    split: str
    language: str
    lines_of_code: int
    description_length: int
    repository: Optional[str] = None
    function_name: Optional[str] = None
    quality_scores: Optional[Dict] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], split: str) -> 'DatasetSample':
        """Create DatasetSample from dictionary data."""
        metadata = data.get('metadata', {})
        
        # Calculate metrics
        lines_of_code = len(data.get('submission', '').split('\n'))
        description_length = len(data.get('problem', ''))
        
        return cls(
            problem_id=data.get('problem_id', ''),
            submission_id=data.get('submission_id', ''),
            problem=data.get('problem', ''),
            submission=data.get('submission', ''),
            source=data.get('source', ''),
            split=split,
            language=metadata.get('language', 'unknown'),
            lines_of_code=lines_of_code,
            description_length=description_length,
            repository=metadata.get('repository_name'),
            function_name=metadata.get('function_name'),
            quality_scores=data.get('quality_scores')
        )


class DatasetLoader:
    """Loads and manages dataset samples with filtering capabilities."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the integrated dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.samples: List[DatasetSample] = []
        self.metadata: Dict[str, Any] = {}
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the complete dataset from all splits."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Load metadata
        metadata_file = self.dataset_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        
        # Load samples from all splits
        splits = ['train', 'valid', 'test']
        total_loaded = 0
        
        for split in splits:
            split_file = self.dataset_path / f"{split}.jsonl"
            if split_file.exists():
                split_samples = self._load_split(split_file, split)
                self.samples.extend(split_samples)
                total_loaded += len(split_samples)
                logger.info(f"Loaded {len(split_samples)} samples from {split} split")
        
        logger.info(f"Total samples loaded: {total_loaded}")
        
        if not self.samples:
            raise ValueError(f"No samples found in dataset: {self.dataset_path}")
    
    def _load_split(self, file_path: Path, split: str) -> List[DatasetSample]:
        """Load samples from a single split file."""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line.strip())
                    sample = DatasetSample.from_dict(data, split)
                    samples.append(sample)
                    
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error parsing {file_path}:{line_num} - {e}")
        
        return samples
    
    def get_samples(self, 
                   split: Optional[str] = None,
                   language: Optional[str] = None,
                   min_lines: Optional[int] = None,
                   max_lines: Optional[int] = None,
                   min_description: Optional[int] = None,
                   max_description: Optional[int] = None,
                   limit: Optional[int] = None,
                   offset: int = 0,
                   random_seed: Optional[int] = None) -> Tuple[List[DatasetSample], int]:
        """
        Get filtered samples with pagination.
        
        Args:
            split: Filter by split (train/valid/test)
            language: Filter by programming language
            min_lines: Minimum lines of code
            max_lines: Maximum lines of code
            min_description: Minimum description length
            max_description: Maximum description length
            limit: Maximum number of samples to return
            offset: Number of samples to skip
            random_seed: Random seed for shuffling (if provided)
            
        Returns:
            Tuple of (filtered_samples, total_count)
        """
        # Apply filters
        filtered_samples = self.samples
        
        if split:
            filtered_samples = [s for s in filtered_samples if s.split == split]
        
        if language:
            filtered_samples = [s for s in filtered_samples if s.language == language]
        
        if min_lines is not None:
            filtered_samples = [s for s in filtered_samples if s.lines_of_code >= min_lines]
        
        if max_lines is not None:
            filtered_samples = [s for s in filtered_samples if s.lines_of_code <= max_lines]
        
        if min_description is not None:
            filtered_samples = [s for s in filtered_samples if s.description_length >= min_description]
        
        if max_description is not None:
            filtered_samples = [s for s in filtered_samples if s.description_length <= max_description]
        
        total_count = len(filtered_samples)
        
        # Apply random shuffling if seed provided
        if random_seed is not None:
            random.Random(random_seed).shuffle(filtered_samples)
        
        # Apply pagination
        if limit is not None:
            end_idx = offset + limit
            filtered_samples = filtered_samples[offset:end_idx]
        elif offset > 0:
            filtered_samples = filtered_samples[offset:]
        
        return filtered_samples, total_count
    
    def get_sample_by_id(self, submission_id: str) -> Optional[DatasetSample]:
        """Get a specific sample by submission ID."""
        for sample in self.samples:
            if sample.submission_id == submission_id:
                return sample
        return None
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        if not self.samples:
            return {}
        
        stats = {
            'total_samples': len(self.samples),
            'splits': {},
            'languages': {},
            'lines_of_code': {
                'min': min(s.lines_of_code for s in self.samples),
                'max': max(s.lines_of_code for s in self.samples),
                'avg': sum(s.lines_of_code for s in self.samples) / len(self.samples)
            },
            'description_length': {
                'min': min(s.description_length for s in self.samples),
                'max': max(s.description_length for s in self.samples),
                'avg': sum(s.description_length for s in self.samples) / len(self.samples)
            }
        }
        
        # Count by split
        for sample in self.samples:
            split = sample.split
            stats['splits'][split] = stats['splits'].get(split, 0) + 1
        
        # Count by language
        for sample in self.samples:
            lang = sample.language
            stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
        
        return stats
    
    def get_filter_options(self) -> Dict[str, Any]:
        """Get available filter options."""
        splits = sorted(set(s.split for s in self.samples))
        languages = sorted(set(s.language for s in self.samples))
        
        lines_range = [
            min(s.lines_of_code for s in self.samples),
            max(s.lines_of_code for s in self.samples)
        ]
        
        desc_range = [
            min(s.description_length for s in self.samples),
            max(s.description_length for s in self.samples)
        ]
        
        return {
            'splits': splits,
            'languages': languages,
            'lines_of_code_range': lines_range,
            'description_length_range': desc_range
        }
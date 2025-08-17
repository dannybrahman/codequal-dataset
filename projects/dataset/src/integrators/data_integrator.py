"""
Generic data source integration framework for CodeQual dataset.

This module provides a flexible architecture for integrating different data sources
(CodeEval, CodeContests, MBPP, GitHub repositories, etc.) into the unified CodeQual schema.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from ..processors import CodeSample, QualityScores


@dataclass
class RawDataSample:
    """Base class for raw data samples from any source."""
    source_id: str
    raw_data: Dict[str, Any]
    
    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Get a field from raw_data with optional default."""
        return self.raw_data.get(field_name, default)


class DataSourceIntegrator(ABC):
    """
    Abstract base class for data source integrators.
    
    Each data source (CodeEval, CodeContests, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, data_path: Union[str, Path], source_name: str):
        """
        Initialize the integrator.
        
        Args:
            data_path: Path to the data source directory/file
            source_name: Name of the data source (e.g., 'codeeval', 'codecontests')
        """
        self.data_path = Path(data_path)
        self.source_name = source_name
        self.raw_samples: List[RawDataSample] = []
        self.converted_samples: List[CodeSample] = []
        self.custom_test_samples: Optional[set] = None  # Set of (problem_id, submission_id) tuples
        self.custom_test_source: Optional[str] = None   # Source file/description for metadata
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"{source_name} data path not found: {data_path}")
        
        logging.info(f"Initialized {source_name} integrator with path: {self.data_path}")
    
    @abstractmethod
    def load_raw_data(self) -> List[RawDataSample]:
        """
        Load raw data from the source into RawDataSample objects.
        
        Returns:
            List of RawDataSample objects
        """
        pass
    
    @abstractmethod
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """
        Parse a single raw data entry into a RawDataSample.
        
        Args:
            raw_data: Raw data dictionary from the source
            
        Returns:
            RawDataSample object
        """
        pass
    
    @abstractmethod
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, 
                                  sample_index: int) -> CodeSample:
        """
        Convert a raw sample to unified CodeSample format.
        
        Args:
            raw_sample: Raw sample to convert
            sample_index: Index of the sample for ID generation
            
        Returns:
            CodeSample in unified schema
        """
        pass
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics and analysis
        """
        if not self.raw_samples:
            self.load_raw_data()
        
        analysis = {
            'source': self.source_name,
            'total_samples': len(self.raw_samples),
            'sample_fields': self._analyze_sample_fields(),
            'data_statistics': self._compute_data_statistics()
        }
        
        return analysis
    
    def _analyze_sample_fields(self) -> Dict[str, Any]:
        """Analyze the fields present in raw samples."""
        if not self.raw_samples:
            return {}
        
        all_fields = set()
        field_coverage = {}
        
        for sample in self.raw_samples:
            sample_fields = set(sample.raw_data.keys())
            all_fields.update(sample_fields)
            
            for field in sample_fields:
                field_coverage[field] = field_coverage.get(field, 0) + 1
        
        # Calculate coverage percentages
        total_samples = len(self.raw_samples)
        field_coverage_percent = {
            field: (count / total_samples) * 100 
            for field, count in field_coverage.items()
        }
        
        return {
            'unique_fields': list(sorted(all_fields)),
            'field_coverage': field_coverage_percent,
            'total_unique_fields': len(all_fields)
        }
    
    def _compute_data_statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about the dataset."""
        # Base implementation - subclasses can override for source-specific stats
        return {
            'samples_loaded': len(self.raw_samples),
            'source_name': self.source_name
        }
    
    def convert_to_unified_schema(self, 
                                 initial_quality_assessment: bool = True,
                                 quality_assessor: Optional[callable] = None) -> List[CodeSample]:
        """
        Convert all raw samples to unified schema format.
        
        Args:
            initial_quality_assessment: Whether to perform initial quality assessment
            quality_assessor: Optional quality assessment function
            
        Returns:
            List of CodeSample objects in unified schema
        """
        if not self.raw_samples:
            self.load_raw_data()
        
        self.converted_samples = []
        
        for idx, raw_sample in enumerate(self.raw_samples):
            try:
                # Convert to unified sample
                unified_sample = self._convert_to_unified_sample(raw_sample, idx)
                
                # Apply quality assessment if requested
                if initial_quality_assessment:
                    if quality_assessor:
                        unified_sample.quality_scores = quality_assessor(raw_sample)
                    else:
                        unified_sample.quality_scores = self._default_quality_assessment(raw_sample)
                
                self.converted_samples.append(unified_sample)
                
            except Exception as e:
                logging.error(f"Error converting sample {idx} from {self.source_name}: {e}")
                continue
        
        logging.info(f"Successfully converted {len(self.converted_samples)} samples "
                    f"from {self.source_name} to unified schema")
        return self.converted_samples
    
    def _default_quality_assessment(self, raw_sample: RawDataSample) -> QualityScores:
        """
        Provide default quality scores when no assessor is provided.
        
        Subclasses should override this for source-specific default scoring.
        
        Args:
            raw_sample: Raw sample to assess
            
        Returns:
            QualityScores with default values
        """
        # Conservative default scores
        return QualityScores(
            functionality=3.0,
            readability=3.0,
            idiomatic=3.0,
            error_handling=2.5,
            efficiency=3.0
        )
    
    def validate_conversion(self) -> Dict[str, Any]:
        """
        Validate the converted samples.
        
        Returns:
            Dictionary with validation results
        """
        if not self.converted_samples:
            raise ValueError("No converted samples to validate")
        
        validation_results = {
            'source': self.source_name,
            'total_converted': len(self.converted_samples),
            'unique_problems': len(set(s.problem_id for s in self.converted_samples)),
            'unique_submissions': len(set(s.submission_id for s in self.converted_samples)),
            'source_consistency': all(s.source == self.source_name for s in self.converted_samples),
            'quality_score_ranges': self._validate_quality_scores(),
            'metadata_completeness': self._validate_metadata(),
            'errors': []
        }
        
        return validation_results
    
    def _validate_quality_scores(self) -> Dict[str, Dict[str, float]]:
        """Validate quality score distributions."""
        dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
        score_ranges = {}
        
        for dim in dimensions:
            scores = [getattr(s.quality_scores, dim) for s in self.converted_samples 
                     if s.quality_scores]
            if scores:
                score_ranges[dim] = {
                    'min': min(scores),
                    'max': max(scores),
                    'avg': sum(scores) / len(scores),
                    'count': len(scores)
                }
        
        return score_ranges
    
    def _validate_metadata(self) -> Dict[str, str]:
        """Validate metadata completeness."""
        # Base validation - subclasses can override for source-specific validation
        total_samples = len(self.converted_samples)
        
        return {
            'samples_with_metadata': f"{sum(1 for s in self.converted_samples if s.metadata)}/{total_samples}",
            'source_field_consistency': f"{sum(1 for s in self.converted_samples if s.source == self.source_name)}/{total_samples}"
        }
    
    def set_custom_test_samples(self, test_sample_ids: set, source_description: str = None):
        """
        Set specific samples to be forced into the test set.
        
        Args:
            test_sample_ids: Set of (problem_id, submission_id) tuples to force into test set
            source_description: Description of where these test samples came from (for metadata)
        """
        self.custom_test_samples = test_sample_ids
        self.custom_test_source = source_description
        logging.info(f"Set {len(test_sample_ids)} custom test samples for {self.source_name}")
    
    def save_converted_dataset(self, output_path: Path,
                              train_ratio: float = 0.8,
                              valid_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save converted dataset with train/validation/test splits.
        Supports custom test set if set_custom_test_samples() was called.
        
        Args:
            output_path: Directory to save the dataset
            train_ratio: Ratio of samples for training set
            valid_ratio: Ratio of samples for validation set
            
        Returns:
            Dictionary mapping split names to file paths
        """
        if not self.converted_samples:
            raise ValueError("No converted samples to save")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Handle custom test set if specified
        if self.custom_test_samples:
            logging.info(f"Using custom test set with {len(self.custom_test_samples)} specific samples")
            
            test_samples = []
            other_samples = []
            
            for sample in self.converted_samples:
                sample_id = (sample.problem_id, sample.submission_id)
                if sample_id in self.custom_test_samples:
                    test_samples.append(sample)
                else:
                    other_samples.append(sample)
            
            logging.info(f"Found {len(test_samples)} test samples from the specified list")
            
            # Split remaining samples into train/valid
            n_other = len(other_samples)
            n_train = int(n_other * (train_ratio / (train_ratio + valid_ratio)))
            
            train_samples = other_samples[:n_train]
            valid_samples = other_samples[n_train:]
            
            logging.info(f"Split sizes - Train: {len(train_samples)}, Valid: {len(valid_samples)}, Test: {len(test_samples)}")
        else:
            # Use default splitting
            total = len(self.converted_samples)
            train_size = int(total * train_ratio)
            valid_size = int(total * valid_ratio)
            
            # Sort samples for reproducible splits
            samples_sorted = sorted(self.converted_samples, key=lambda x: x.problem_id)
            
            # Split dataset
            train_samples = samples_sorted[:train_size]
            valid_samples = samples_sorted[train_size:train_size + valid_size]
            test_samples = samples_sorted[train_size + valid_size:]
        
        # Save splits
        output_files = {}
        splits = [
            ('train', train_samples),
            ('valid', valid_samples),
            ('test', test_samples)
        ]
        
        for split_name, samples in splits:
            if not samples:
                continue
            
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(sample.to_jsonl_line() + '\n')
            
            output_files[split_name] = file_path
            logging.info(f"Saved {len(samples)} samples to {file_path}")
        
        # Save metadata
        metadata = {
            'source': self.source_name,
            'total_samples': len(train_samples) + len(valid_samples) + len(test_samples),
            'splits': {
                'train': len(train_samples),
                'valid': len(valid_samples),
                'test': len(test_samples)
            },
            'conversion_stats': self.validate_conversion(),
            'dataset_analysis': self.analyze_dataset()
        }
        
        # Add custom test set metadata if applicable
        if self.custom_test_samples:
            metadata['custom_test_set'] = True
            metadata['test_samples_from'] = self.custom_test_source
        else:
            metadata['custom_test_set'] = False
        
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        output_files['metadata'] = metadata_file
        
        logging.info(f"{self.source_name} integration completed. Files saved to {output_path}")
        return output_files


class DataSourceRegistry:
    """Registry for managing multiple data source integrators."""
    
    def __init__(self):
        self._integrators: Dict[str, DataSourceIntegrator] = {}
    
    def register(self, integrator: DataSourceIntegrator):
        """Register a data source integrator."""
        self._integrators[integrator.source_name] = integrator
        logging.info(f"Registered data source: {integrator.source_name}")
    
    def get_integrator(self, source_name: str) -> Optional[DataSourceIntegrator]:
        """Get an integrator by source name."""
        return self._integrators.get(source_name)
    
    def list_sources(self) -> List[str]:
        """List all registered data sources."""
        return list(self._integrators.keys())
    
    def analyze_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Analyze all registered data sources."""
        analyses = {}
        for source_name, integrator in self._integrators.items():
            try:
                analyses[source_name] = integrator.analyze_dataset()
            except Exception as e:
                logging.error(f"Error analyzing {source_name}: {e}")
                analyses[source_name] = {'error': str(e)}
        
        return analyses
    
    def convert_all_sources(self, **kwargs) -> Dict[str, List[CodeSample]]:
        """Convert all registered sources to unified schema."""
        all_samples = {}
        for source_name, integrator in self._integrators.items():
            try:
                samples = integrator.convert_to_unified_schema(**kwargs)
                all_samples[source_name] = samples
                logging.info(f"Converted {len(samples)} samples from {source_name}")
            except Exception as e:
                logging.error(f"Error converting {source_name}: {e}")
                all_samples[source_name] = []
        
        return all_samples
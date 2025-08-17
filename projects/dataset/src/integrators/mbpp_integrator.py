"""
MBPP (Mostly Basic Python Problems) dataset integration for CodeQual v2.

MBPP is a benchmark of 1,000 crowd-sourced Python programming problems 
designed to be solvable by entry-level programmers. Each problem consists 
of a task description, code solution, and 3 automated test cases.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .data_integrator import DataSourceIntegrator, RawDataSample
from ..processors import CodeSample, QualityScores


@dataclass
class MBPPSample:
    """Raw MBPP sample structure."""
    task_id: int
    prompt: str
    code: str
    test_list: List[str]
    test_imports: Optional[List[str]] = None
    test_setup_code: Optional[str] = None
    challenge_test_list: Optional[List[str]] = None
    source_file: Optional[str] = None


class MBPPIntegrator(DataSourceIntegrator):
    """Integrates MBPP dataset into unified CodeQual schema."""
    
    def __init__(self, mbpp_data_path: str, use_sanitized: bool = False, preserve_original_splits: bool = False):
        """
        Initialize the MBPP integrator.
        
        Args:
            mbpp_data_path: Path to MBPP data directory
            use_sanitized: Whether to use sanitized-mbpp.json (427 samples) or full mbpp.jsonl (974 samples, default)
            preserve_original_splits: Whether to preserve MBPP's original train/valid/test splits
        """
        super().__init__(mbpp_data_path, "mbpp")
        self.use_sanitized = use_sanitized
        self.preserve_original_splits = preserve_original_splits
        self.data_file = "sanitized-mbpp.json" if use_sanitized else "mbpp.jsonl"
        
        logging.info(f"Initialized MBPP integrator using {self.data_file}")
    
    def load_raw_data(self) -> List[RawDataSample]:
        """Load MBPP samples from the dataset file."""
        self.raw_samples = []
        
        data_file_path = self.data_path / self.data_file
        
        if not data_file_path.exists():
            raise ValueError(f"MBPP data file not found: {data_file_path}")
        
        logging.info(f"Loading MBPP data from {data_file_path}")
        
        if self.use_sanitized:
            # Load sanitized JSON format
            with open(data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                sample = self._parse_single_sample(item)
                self.raw_samples.append(sample)
                
        else:
            # Load JSONL format
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        sample = self._parse_single_sample(data)
                        self.raw_samples.append(sample)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Error parsing {data_file_path}:{line_num} - {e}")
        
        logging.info(f"Loaded {len(self.raw_samples)} MBPP samples")
        return self.raw_samples
    
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """Parse a single MBPP sample into RawDataSample format."""
        task_id = raw_data.get('task_id', len(self.raw_samples))
        source_id = f"mbpp_{task_id}"
        return RawDataSample(source_id=source_id, raw_data=raw_data)
    
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, 
                                  sample_index: int) -> CodeSample:
        """Convert a raw MBPP sample to unified CodeSample format."""
        data = raw_sample.raw_data
        
        # Generate unique IDs
        task_id = data.get('task_id', sample_index)
        problem_id = f"mbpp_{task_id}"
        submission_id = f"mbpp_sub_{sample_index:05d}"
        
        # Get problem description and code
        problem_description = self._clean_problem_description(data)
        code = self._clean_code_solution(data)
        
        # Create metadata
        metadata = {
            'task_id': task_id,
            'language': 'python',
            'lines_of_code': len(code.split('\n')),
            'test_cases': data.get('test_list', []),
            'test_imports': data.get('test_imports', []),
            'test_setup_code': data.get('test_setup_code', ''),
            'challenge_tests': data.get('challenge_test_list', []),
            'source_file': data.get('source_file', ''),
            'dataset_split': self._determine_split(task_id),
            'assessment_method': None  # Will be set when quality assessment is performed
        }
        
        # Create unified sample
        return CodeSample(
            problem_id=problem_id,
            problem=problem_description,
            submission_id=submission_id,
            submission=code,
            source=self.source_name,
            quality_scores=None,  # Will be set during assessment
            metadata=metadata
        )
    
    def _clean_problem_description(self, data: Dict[str, Any]) -> str:
        """Extract and clean the problem description."""
        # Use 'prompt' field for sanitized data, 'text' for JSONL
        description = data.get('prompt', data.get('text', ''))
        
        # Clean up the description
        description = description.strip()
        
        # Ensure it ends with a period for consistency
        if description and not description.endswith('.'):
            description += '.'
        
        return description
    
    def _clean_code_solution(self, data: Dict[str, Any]) -> str:
        """Extract and clean the code solution."""
        code = data.get('code', '')
        
        # Remove Windows line endings and normalize
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Clean up any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _determine_split(self, task_id: int) -> str:
        """
        Determine the original MBPP dataset split based on task_id.
        
        According to the paper:
        - Task IDs 1-10: Few-shot prompting (not for training)
        - Task IDs 11-510: Testing
        - Task IDs 511-600: Validation
        - Task IDs 601-974: Training
        """
        if 1 <= task_id <= 10:
            return "prompt"
        elif 11 <= task_id <= 510:
            return "test"
        elif 511 <= task_id <= 600:
            return "validation"
        elif 601 <= task_id <= 974:
            return "train"
        else:
            return "unknown"
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the MBPP dataset structure and statistics."""
        if not self.raw_samples:
            self.load_raw_data()
        
        # Count samples by original split
        split_counts = {"prompt": 0, "test": 0, "validation": 0, "train": 0, "unknown": 0}
        code_lengths = []
        test_counts = []
        has_imports = 0
        has_setup = 0
        
        for sample in self.raw_samples:
            data = sample.raw_data
            task_id = data.get('task_id', 0)
            
            # Count by split
            split = self._determine_split(task_id)
            split_counts[split] += 1
            
            # Code analysis
            code = data.get('code', '')
            code_lengths.append(len(code))
            
            # Test analysis
            test_list = data.get('test_list', [])
            test_counts.append(len(test_list))
            
            # Metadata analysis
            if data.get('test_imports'):
                has_imports += 1
            if data.get('test_setup_code'):
                has_setup += 1
        
        analysis = {
            'source': self.source_name,
            'data_file': self.data_file,
            'total_samples': len(self.raw_samples),
            'original_split_distribution': split_counts,
            'code_length_stats': {
                'min': min(code_lengths) if code_lengths else 0,
                'max': max(code_lengths) if code_lengths else 0,
                'avg': sum(code_lengths) / len(code_lengths) if code_lengths else 0
            },
            'test_case_stats': {
                'min': min(test_counts) if test_counts else 0,
                'max': max(test_counts) if test_counts else 0,
                'avg': sum(test_counts) / len(test_counts) if test_counts else 0
            },
            'samples_with_imports': has_imports,
            'samples_with_setup': has_setup,
            'data_path': str(self.data_path)
        }
        
        return analysis
    
    def _compute_data_statistics(self) -> Dict[str, Any]:
        """Compute MBPP-specific statistics."""
        if not self.raw_samples:
            return super()._compute_data_statistics()
        
        # Split distribution
        split_dist = {"prompt": 0, "test": 0, "validation": 0, "train": 0, "unknown": 0}
        complexity_indicators = []
        
        for sample in self.raw_samples:
            data = sample.raw_data
            task_id = data.get('task_id', 0)
            
            # Split distribution
            split = self._determine_split(task_id)
            split_dist[split] += 1
            
            # Estimate complexity by code length and test count
            code_length = len(data.get('code', ''))
            test_count = len(data.get('test_list', []))
            complexity_indicators.append(code_length + test_count * 50)  # Weighted complexity
        
        stats = super()._compute_data_statistics()
        stats.update({
            'original_split_distribution': split_dist,
            'avg_complexity_indicator': sum(complexity_indicators) / len(complexity_indicators) if complexity_indicators else 0,
            'data_source': 'sanitized' if self.use_sanitized else 'full'
        })
        
        return stats
    
    def _validate_metadata(self) -> Dict[str, str]:
        """Validate MBPP-specific metadata completeness."""
        base_validation = super()._validate_metadata()
        
        # Check MBPP-specific required metadata fields
        required_fields = ['task_id', 'language', 'test_cases', 'dataset_split']
        total_samples = len(self.converted_samples)
        
        for field in required_fields:
            count = sum(1 for s in self.converted_samples if field in s.metadata)
            base_validation[f'{field}_completeness'] = f"{count}/{total_samples}"
        
        return base_validation
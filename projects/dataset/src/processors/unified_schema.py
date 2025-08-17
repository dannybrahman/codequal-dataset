"""
Unified dataset schema for CodeQual v2 with 5-dimensional quality scoring.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import json
from pathlib import Path


@dataclass
class QualityScores:
    """5-dimensional quality scores (1.0-5.0 scale)."""
    functionality: float  # Does the code work as per the description?
    readability: float    # How cognitively manageable is the code?
    idiomatic: float      # Does it follow language best practices?
    error_handling: float # Does it manage edge cases and failures?
    efficiency: float     # Is it scalable and well-performing?
    
    def __post_init__(self):
        """Validate score ranges."""
        for field_name, value in self.__dict__.items():
            if not (1.0 <= value <= 5.0):
                raise ValueError(f"{field_name} must be between 1.0 and 5.0, got {value}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'functionality': self.functionality,
            'readability': self.readability,
            'idiomatic': self.idiomatic,
            'error_handling': self.error_handling,
            'efficiency': self.efficiency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'QualityScores':
        """Create from dictionary."""
        return cls(
            functionality=data['functionality'],
            readability=data['readability'],
            idiomatic=data['idiomatic'],
            error_handling=data['error_handling'],
            efficiency=data['efficiency']
        )
    
    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall score."""
        if weights is None:
            # Equal weights by default
            weights = {
                'functionality': 0.2,
                'readability': 0.2,
                'idiomatic': 0.2,
                'error_handling': 0.2,
                'efficiency': 0.2
            }
        
        return sum(getattr(self, dim) * weights.get(dim, 0.0) 
                  for dim in ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency'])


@dataclass
class CodeSample:
    """Unified dataset schema for code quality assessment."""
    problem_id: str           # Unique identifier for the problem
    problem: str              # Problem description/statement
    submission_id: str        # Unique identifier for this submission
    submission: str           # The code solution
    quality_scores: Optional[QualityScores] = None  # 5-dimensional quality assessment (optional for pre-assessment)
    source: str = "codeeval"  # Data source: "codequal", "codeeval", "codecontests", "mbpp", "github"
    metadata: Dict[str, Any] = None  # Source-specific metadata
    
    def __post_init__(self):
        """Validate required fields."""
        if not self.problem_id:
            raise ValueError("problem_id cannot be empty")
        if not self.submission_id:
            raise ValueError("submission_id cannot be empty")
        if not self.submission.strip():
            raise ValueError("submission cannot be empty")
        if self.source not in DatasetSchema.VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source}")
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'problem_id': self.problem_id,
            'problem': self.problem,
            'submission_id': self.submission_id,
            'submission': self.submission,
            'quality_scores': self.quality_scores.to_dict() if self.quality_scores else None,
            'source': self.source,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSample':
        """Create from dictionary."""
        quality_scores = None
        if data.get('quality_scores'):
            quality_scores = QualityScores.from_dict(data['quality_scores'])
        
        return cls(
            problem_id=data['problem_id'],
            problem=data['problem'],
            submission_id=data['submission_id'],
            submission=data['submission'],
            quality_scores=quality_scores,
            source=data['source'],
            metadata=data.get('metadata', {})
        )
    
    def to_jsonl_line(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def get_code_metrics(self) -> Dict[str, Any]:
        """Extract basic code metrics."""
        lines = self.submission.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'char_count': len(self.submission),
            'has_imports': any('import ' in line for line in lines),
            'has_functions': any('def ' in line for line in lines),
            'has_classes': any('class ' in line for line in lines),
            'has_comments': any(line.strip().startswith('#') for line in lines),
            'has_docstrings': '"""' in self.submission or "'''" in self.submission
        }


class DatasetSchema:
    """Schema validation and utilities for the unified dataset."""
    
    REQUIRED_FIELDS = ['problem_id', 'problem', 'submission_id', 'submission', 'quality_scores', 'source']
    VALID_SOURCES = ['codequal', 'codeeval', 'codecontests', 'mbpp', 'github', 'codenet', 'humaneval-x', 'codesearchnet']
    
    @classmethod
    def validate_sample(cls, sample_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a sample dictionary against the schema.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in sample_dict:
                errors.append(f"Missing required field: {field}")
        
        # Check source validity
        if 'source' in sample_dict and sample_dict['source'] not in cls.VALID_SOURCES:
            errors.append(f"Invalid source: {sample_dict['source']}")
        
        # Check quality scores structure
        if 'quality_scores' in sample_dict:
            qs = sample_dict['quality_scores']
            required_dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
            
            if not isinstance(qs, dict):
                errors.append("quality_scores must be a dictionary")
            else:
                for dim in required_dimensions:
                    if dim not in qs:
                        errors.append(f"Missing quality dimension: {dim}")
                    elif not isinstance(qs[dim], (int, float)) or not (1.0 <= qs[dim] <= 5.0):
                        errors.append(f"Invalid {dim} score: must be float between 1.0 and 5.0")
        
        return errors
    
    @classmethod
    def validate_jsonl_file(cls, file_path: Path) -> Dict[str, Any]:
        """
        Validate an entire JSONL dataset file.
        
        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            'valid': True,
            'total_samples': 0,
            'valid_samples': 0,
            'errors': [],
            'source_distribution': {},
            'quality_score_stats': {}
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    results['total_samples'] += 1
                    
                    try:
                        sample_dict = json.loads(line.strip())
                        validation_errors = cls.validate_sample(sample_dict)
                        
                        if validation_errors:
                            results['valid'] = False
                            for error in validation_errors:
                                results['errors'].append(f"Line {line_num}: {error}")
                        else:
                            results['valid_samples'] += 1
                            
                            # Track source distribution
                            source = sample_dict.get('source', 'unknown')
                            results['source_distribution'][source] = results['source_distribution'].get(source, 0) + 1
                    
                    except json.JSONDecodeError as e:
                        results['valid'] = False
                        results['errors'].append(f"Line {line_num}: Invalid JSON - {e}")
        
        except FileNotFoundError:
            results['valid'] = False
            results['errors'].append(f"File not found: {file_path}")
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Unexpected error: {e}")
        
        return results
    
    @classmethod
    def create_sample_from_legacy(cls, legacy_sample: Dict[str, Any], 
                                  quality_scores: QualityScores) -> CodeSample:
        """
        Convert legacy CodeQual format to unified schema.
        
        Args:
            legacy_sample: Original CodeQual sample
            quality_scores: New 5-dimensional scores
        """
        return CodeSample(
            problem_id=legacy_sample['problem_id'],
            problem=legacy_sample['problem'],
            submission_id=legacy_sample['submission_id'],
            submission=legacy_sample['submission'],
            quality_scores=quality_scores,
            source='codequal',
            metadata={
                'original_label': legacy_sample.get('label'),
                'language': 'python',
                'lines_of_code': len(legacy_sample['submission'].split('\n')),
                'assessment_method': 'gpt4_migrated'
            }
        )


def load_jsonl_dataset(file_path: Path) -> List[CodeSample]:
    """Load a dataset from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line.strip())
                samples.append(CodeSample.from_dict(sample_dict))
    return samples


def save_jsonl_dataset(samples: List[CodeSample], file_path: Path) -> None:
    """Save dataset to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample.to_jsonl_line() + '\n')
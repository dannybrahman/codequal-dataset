"""
CodeNet data integrator for CodeQual dataset.

This integrator processes CodeNet data that contains the original 2,250 CodeQual samples
with their human-annotated 5-dimensional quality scores.
"""
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging

from .data_integrator import DataSourceIntegrator, RawDataSample
from ..processors.unified_schema import CodeSample, QualityScores


logger = logging.getLogger(__name__)


class CodeNetIntegrator(DataSourceIntegrator):
    """
    Integrates CodeNet data for the CodeQual dataset.
    
    Processes the subset of CodeNet that contains the 2,250 CodeQual samples
    with their human-annotated quality scores.
    """
    
    def __init__(self, data_path: str, quality_scores_file: Optional[str] = None, 
                 test_samples_file: Optional[str] = None):
        """
        Initialize CodeNet integrator.
        
        Args:
            data_path: Path to CodeNet subset directory
            quality_scores_file: Optional path to CSV file with quality scores
                                If provided, loads existing quality scores
            test_samples_file: Optional path to file with specific samples for test set
                              Format: problem_id,submission_id per line
        """
        super().__init__(data_path, "codenet")
        self.quality_scores_file = quality_scores_file
        self.test_samples_file = test_samples_file
        self.quality_scores_lookup: Dict[str, QualityScores] = {}
        self.test_sample_ids: Set[Tuple[str, str]] = set()
        
        # Validate CodeNet-specific paths
        required_dirs = ['problem_descriptions', 'metadata', 'python_800']
        for req_dir in required_dirs:
            if not (self.data_path / req_dir).exists():
                raise FileNotFoundError(f"Required CodeNet directory not found: {req_dir}")
        
        # Load test sample IDs if provided
        if self.test_samples_file:
            self._load_test_sample_ids()
    
    def load_raw_data(self) -> List[RawDataSample]:
        """Load raw CodeNet samples."""
        logger.info("Loading CodeNet data...")
        
        # Load quality scores if provided
        if self.quality_scores_file:
            self._load_quality_scores()
        
        self.raw_samples = []
        problems_dir = self.data_path / "python_800"
        
        # Process each problem directory
        for problem_dir in problems_dir.iterdir():
            if not problem_dir.is_dir():
                continue
                
            problem_id = problem_dir.name
            
            # Load problem description
            problem_desc = self._load_problem_description(problem_id)
            if not problem_desc:
                logger.warning(f"No problem description found for {problem_id}")
                continue
            
            # Load metadata for this problem
            metadata_dict = self._load_problem_metadata(problem_id)
            
            # Process each submission
            for submission_file in problem_dir.glob("*.py"):
                submission_id = submission_file.stem
                
                try:
                    # Read code
                    code = submission_file.read_text(encoding='utf-8')
                    
                    # Get submission metadata
                    sub_metadata = metadata_dict.get(submission_id, {})
                    
                    # Create raw data dictionary
                    raw_data = {
                        'problem_id': problem_id,
                        'submission_id': submission_id,
                        'problem_description': problem_desc,
                        'code': code,
                        'metadata': sub_metadata
                    }
                    
                    sample = self._parse_single_sample(raw_data)
                    self.raw_samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {submission_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.raw_samples)} CodeNet samples")
        return self.raw_samples
    
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """Parse a single raw data entry into a RawDataSample."""
        return RawDataSample(
            source_id=f"{raw_data['problem_id']}_{raw_data['submission_id']}",
            raw_data=raw_data
        )
    
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, sample_index: int) -> CodeSample:
        """Convert a raw sample to unified CodeSample format."""
        raw_data = raw_sample.raw_data
        
        # Get quality scores if available
        submission_id = raw_data['submission_id']
        quality_scores = self.quality_scores_lookup.get(submission_id)
        
        # Create metadata
        metadata = {
            'source': 'codenet',
            'assessment_method': None if not quality_scores else 'human_annotation',
            'original_metadata': raw_data.get('metadata', {}),
            **raw_data.get('metadata', {})  # Include original metadata fields
        }
        
        return CodeSample(
            problem_id=raw_data['problem_id'],
            problem=raw_data['problem_description'],
            submission_id=raw_data['submission_id'],
            submission=raw_data['code'],
            quality_scores=quality_scores,
            source='codenet',
            metadata=metadata
        )
    
    def convert_to_unified_schema(self, initial_quality_assessment: bool = False) -> List[CodeSample]:
        """Convert CodeNet samples to unified schema."""
        logger.info("Converting CodeNet samples to unified schema...")
        
        if not self.raw_samples:
            self.load_raw_data()
        
        unified_samples = []
        for i, raw_sample in enumerate(self.raw_samples):
            unified_sample = self._convert_to_unified_sample(raw_sample, i)
            unified_samples.append(unified_sample)
        
        # Store in base class attribute for save_converted_dataset()
        self.converted_samples = unified_samples
        
        logger.info(f"Converted {len(unified_samples)} samples to unified schema")
        return unified_samples
    
    def _load_problem_description(self, problem_id: str) -> Optional[str]:
        """Load problem description from HTML file."""
        desc_file = self.data_path / "problem_descriptions" / f"{problem_id}.html"
        
        if not desc_file.exists():
            return None
        
        try:
            html_content = desc_file.read_text(encoding='utf-8')
            # Basic HTML to text conversion - extract text from HTML
            # This is a simple approach; could use BeautifulSoup for better parsing
            description = self._html_to_text(html_content)
            return description.strip()
        except Exception as e:
            logger.warning(f"Failed to load description for {problem_id}: {e}")
            return None
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean problem description using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract clean problem description based on the structure
        # Look for the first section element which typically contains the problem statement
        first_section = soup.find('section')
        if first_section:
            # Remove the h3 element (usually contains "Problem Statement" header)
            h3 = first_section.find('h3')
            if h3:
                h3.extract()
            
            # Get the text content and clean it up
            description = first_section.get_text(strip=True)
            
            # Further clean up by removing everything after "Constraints" or "Input" sections
            import re
            patterns = [
                r'\s*Constraints\s+.*$',
                r'\s*Input\s+.*$', 
                r'\s*Sample\s+Input\s+.*$'
            ]
            
            for pattern in patterns:
                description = re.sub(pattern, '', description, flags=re.IGNORECASE | re.DOTALL)
            
            return description.strip()
        
        # Fallback: look for first paragraph if no section found
        first_p = soup.find('p')
        if first_p:
            return first_p.get_text(strip=True)
        
        # Final fallback: return cleaned full text
        return soup.get_text(strip=True)
    
    def _load_problem_metadata(self, problem_id: str) -> Dict[str, Dict[str, Any]]:
        """Load metadata for a specific problem."""
        metadata_file = self.data_path / "metadata" / f"{problem_id}.csv"
        
        if not metadata_file.exists():
            return {}
        
        metadata_dict = {}
        try:
            with open(metadata_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    submission_id = row['submission_id']
                    metadata_dict[submission_id] = row
        except Exception as e:
            logger.warning(f"Failed to load metadata for {problem_id}: {e}")
        
        return metadata_dict
    
    def _load_quality_scores(self):
        """Load quality scores from CSV file."""
        if not self.quality_scores_file or not Path(self.quality_scores_file).exists():
            logger.warning(f"Quality scores file not found: {self.quality_scores_file}")
            return
        
        logger.info(f"Loading quality scores from {self.quality_scores_file}")
        
        try:
            import pandas as pd
            
            df = pd.read_csv(self.quality_scores_file)
            
            for _, row in df.iterrows():
                submission_id = row['submission_id']
                
                quality_scores = QualityScores(
                    functionality=float(row['functionality']),
                    readability=float(row['readability']), 
                    idiomatic=float(row['pythonic']),  # Note: CSV uses 'pythonic'
                    error_handling=float(row['error_handling']),
                    efficiency=float(row['efficiency'])
                )
                
                self.quality_scores_lookup[submission_id] = quality_scores
                
            logger.info(f"Loaded quality scores for {len(self.quality_scores_lookup)} submissions")
            
        except ImportError:
            logger.error("pandas required for loading quality scores CSV")
            raise
        except Exception as e:
            logger.error(f"Failed to load quality scores: {e}")
            raise
    
    def _load_test_sample_ids(self):
        """Load specific samples that should be in test set."""
        if not self.test_samples_file or not Path(self.test_samples_file).exists():
            logger.warning(f"Test samples file not found: {self.test_samples_file}")
            return
        
        logger.info(f"Loading test sample IDs from {self.test_samples_file}")
        
        try:
            with open(self.test_samples_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        problem_id, submission_id = line.split(',', 1)
                        self.test_sample_ids.add((problem_id.strip(), submission_id.strip()))
            
            logger.info(f"Loaded {len(self.test_sample_ids)} test sample IDs")
        except Exception as e:
            logger.error(f"Failed to load test sample IDs: {e}")
            raise
    
    def save_converted_dataset(self, output_path: Path, train_ratio: float = 0.8,
                              valid_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save converted dataset with custom test set if specified.
        
        Uses parent class functionality for custom test set splitting.
        """
        # Set custom test samples in parent class if we have them
        if self.test_sample_ids:
            if not self.test_samples_file:
                raise ValueError("test_samples_file must be provided when using custom test samples")
            self.set_custom_test_samples(self.test_sample_ids, self.test_samples_file)
        
        # Use parent class method (now supports custom test sets)
        return super().save_converted_dataset(output_path, train_ratio, valid_ratio)
    
    def get_source_name(self) -> str:
        """Return the source name for this integrator.""" 
        return "codenet"
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the CodeNet dataset structure and statistics based on final converted samples."""
        # Analyze the final converted samples that will be written to disk
        if not self.converted_samples:
            # If no converted samples yet, return empty stats
            return {
                'source': self.source_name,
                'total_samples': 0,
                'lines_of_code_stats': {'min': 0, 'max': 0, 'avg': 0.0},
                'note': 'No converted samples available for analysis'
            }
        
        # Count unique problems and submissions from converted samples
        unique_problems = set()
        unique_submissions = set()
        code_lengths = []
        
        for sample in self.converted_samples:
            unique_problems.add(sample.problem_id)
            unique_submissions.add(sample.submission_id)
            
            # Calculate lines of code from the actual submission
            lines_of_code = self._count_lines_of_code(sample.submission, 'python')  # CodeNet is Python-only
            code_lengths.append(lines_of_code)
        
        # Calculate quality score ranges if available
        quality_ranges = {}
        if self.quality_scores_lookup:
            dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
            for dim in dimensions:
                values = [getattr(qs, dim) for qs in self.quality_scores_lookup.values()]
                quality_ranges[dim] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values)
                }
        
        total_samples = len(self.converted_samples)
        analysis = {
            'source': self.source_name,
            'total_samples': total_samples,
            'unique_problems': len(unique_problems),
            'unique_submissions': len(unique_submissions),
            'lines_of_code_stats': {
                'min': min(code_lengths) if code_lengths else 0,
                'max': max(code_lengths) if code_lengths else 0,
                'avg': sum(code_lengths) / len(code_lengths) if code_lengths else 0
            },
            'quality_scores_available': len(self.quality_scores_lookup) > 0,
            'quality_score_coverage': len(self.quality_scores_lookup) if self.quality_scores_lookup else 0,
            'quality_score_ranges': quality_ranges
        }
        
        return analysis
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get summary metadata about the dataset."""
        if not self.raw_samples:
            self.load_raw_data()
        
        # Count unique problems and submissions
        unique_problems = set(s.raw_data['problem_id'] for s in self.raw_samples)
        unique_submissions = set(s.raw_data['submission_id'] for s in self.raw_samples)
        
        # Calculate quality score ranges if available
        quality_ranges = {}
        if self.quality_scores_lookup:
            dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
            for dim in dimensions:
                values = [getattr(qs, dim) for qs in self.quality_scores_lookup.values()]
                quality_ranges[dim] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values)
                }
        
        return {
            'source': self.get_source_name(),
            'total_samples': len(self.raw_samples),
            'unique_problems': len(unique_problems),
            'unique_submissions': len(unique_submissions), 
            'quality_scores_available': len(self.quality_scores_lookup) > 0,
            'quality_score_coverage': len(self.quality_scores_lookup) if self.quality_scores_lookup else 0,
            'quality_score_ranges': quality_ranges
        }
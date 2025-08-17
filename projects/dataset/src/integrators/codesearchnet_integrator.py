"""
CodeSearchNet dataset integration for CodeQual v2.

CodeSearchNet is a dataset of 2 million (docstring, function) pairs from 
GitHub repositories. It contains functions and their natural language 
documentation across 6 programming languages: Python, JavaScript, Ruby, 
Go, Java, and PHP.

Each sample consists of a function with its associated docstring, providing
natural problem descriptions for real-world code quality assessment.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from .data_integrator import DataSourceIntegrator, RawDataSample
from ..processors import CodeSample, QualityScores


logger = logging.getLogger(__name__)


@dataclass
class CodeSearchNetSample:
    """Raw CodeSearchNet sample structure."""
    id: str
    repository_name: str
    func_name: str
    language: str
    func_code_string: str
    func_documentation_string: str
    split: Optional[str] = None  # train/test/valid from original dataset


class CodeSearchNetIntegrator(DataSourceIntegrator):
    """Integrates CodeSearchNet dataset into unified CodeQual schema."""
    
    def __init__(self, codesearchnet_data_path: str, languages: Optional[List[str]] = None, 
                 preserve_original_splits: bool = False, max_samples_per_language: Optional[int] = None):
        """
        Initialize the CodeSearchNet integrator.
        
        Args:
            codesearchnet_data_path: Path to CodeSearchNet data directory
            languages: List of languages to process (default: all available languages)
            preserve_original_splits: Whether to preserve CodeSearchNet's original train/valid/test splits
            max_samples_per_language: Maximum samples per language (for testing/development)
        """
        super().__init__(codesearchnet_data_path, "codesearchnet")
        # Use provided languages or all available languages in the dataset
        available_languages = ['python', 'javascript', 'java', 'go', 'php', 'ruby']
        self.languages = languages or available_languages
        self.preserve_original_splits = preserve_original_splits
        self.max_samples_per_language = max_samples_per_language
        
        logger.info(f"Initialized CodeSearchNet integrator for languages: {self.languages}")
        if max_samples_per_language:
            logger.info(f"Limited to {max_samples_per_language} samples per language")
    
    def load_raw_data(self) -> List[RawDataSample]:
        """Load CodeSearchNet samples from JSONL files."""
        self.raw_samples = []
        
        # Check for different possible file naming patterns and structures
        file_patterns = [
            "{language}_{split}.jsonl",  # Simple format
            "{language}_{split}.json",   # Alternative JSON format
            "{split}_{language}.jsonl",  # Alternative ordering
            "{language}/{split}.jsonl",  # Subdirectory structure
            "{language}/{language}/final/jsonl/{split}/{language}_{split}_*.jsonl"  # Kaggle structure
        ]
        
        splits = ['train', 'test', 'valid']
        
        logger.info(f"Loading CodeSearchNet data from {self.data_path}")
        
        for language in self.languages:
            language_samples = 0
            
            for split in splits:
                samples_found = False
                
                # Try different file patterns
                for pattern in file_patterns:
                    file_pattern = pattern.format(language=language, split=split)
                    
                    # Handle wildcard patterns (for Kaggle structure with multiple files)
                    if '*' in file_pattern:
                        import glob
                        file_paths = glob.glob(str(self.data_path / file_pattern))
                        if file_paths:
                            for file_path in sorted(file_paths):
                                logger.info(f"Loading {language} {split} data from {Path(file_path).name}")
                                samples_loaded = self._load_jsonl_file(Path(file_path), language, split)
                                language_samples += samples_loaded
                            samples_found = True
                            break
                    else:
                        file_path = self.data_path / file_pattern
                        if file_path.exists():
                            logger.info(f"Loading {language} {split} data from {file_path}")
                            samples_loaded = self._load_jsonl_file(file_path, language, split)
                            language_samples += samples_loaded
                            samples_found = True
                            break
                
                if not samples_found:
                    logger.debug(f"No {language} {split} file found (tried multiple patterns)")
            
            if language_samples == 0:
                logger.warning(f"No data files found for language: {language}")
            else:
                logger.info(f"Loaded {language_samples} samples for {language}")
                
                # Apply max samples limit per language
                if self.max_samples_per_language and language_samples > self.max_samples_per_language:
                    # Remove excess samples for this language
                    current_count = 0
                    filtered_samples = []
                    
                    for sample in self.raw_samples:
                        if sample.raw_data.get('language') == language:
                            if current_count < self.max_samples_per_language:
                                filtered_samples.append(sample)
                                current_count += 1
                        else:
                            filtered_samples.append(sample)
                    
                    self.raw_samples = filtered_samples
                    logger.info(f"Limited {language} to {self.max_samples_per_language} samples")
        
        logger.info(f"Loaded {len(self.raw_samples)} total CodeSearchNet samples")
        return self.raw_samples
    
    def _load_jsonl_file(self, file_path: Path, language: str, split: str) -> int:
        """Load samples from a JSONL file."""
        samples_loaded = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Add split and language info if not present
                        if 'split' not in data:
                            data['split'] = split
                        if 'language' not in data:
                            data['language'] = language
                        
                        sample = self._parse_single_sample(data)
                        self.raw_samples.append(sample)
                        samples_loaded += 1
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error parsing {file_path}:{line_num} - {e}")
                        
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
        
        return samples_loaded
    
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """Parse a single CodeSearchNet sample into RawDataSample format.
        
        Handles both original HuggingFace format and Kaggle format.
        """
        # Generate unique source ID
        # Try different ID fields for compatibility
        sample_id = (raw_data.get('id') or 
                    raw_data.get('sha') or 
                    raw_data.get('url') or 
                    f"unknown_{len(self.raw_samples)}")
        language = raw_data.get('language', 'unknown')
        source_id = f"codesearchnet_{language}_{sample_id}"
        
        return RawDataSample(source_id=source_id, raw_data=raw_data)
    
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, 
                                  sample_index: int) -> CodeSample:
        """Convert a raw CodeSearchNet sample to unified CodeSample format.
        
        Handles both HuggingFace format and Kaggle format fields.
        """
        data = raw_sample.raw_data
        
        # Extract basic information - handle both formats
        # Kaggle uses 'sha' or 'url', HuggingFace uses 'id'
        sample_id = (data.get('id') or 
                    data.get('sha') or 
                    data.get('url') or 
                    f'unknown_{sample_index}')
        
        language = data.get('language', 'unknown')
        func_name = data.get('func_name', 'unknown_function')
        
        # Kaggle uses 'repo', HuggingFace uses 'repository_name'
        repository = data.get('repository_name') or data.get('repo', 'unknown/repository')
        
        # Generate unique IDs
        problem_id = f"codesearchnet_{language}_{sample_id[:40]}"  # Truncate long SHAs
        submission_id = f"codesearchnet_sub_{sample_index:06d}"
        
        # Get function code and documentation
        code = self._clean_function_code(data)
        problem_description = self._clean_documentation(data, func_name)
        
        # Handle path information (Kaggle includes 'path' field)
        file_path = data.get('path', '')
        
        # Create metadata
        metadata = {
            'original_id': sample_id,
            'language': language,
            'function_name': func_name,
            'repository_name': repository,
            'file_path': file_path,
            'lines_of_code': len(code.split('\n')),
            'has_documentation': bool(self._get_docstring_from_data(data)),
            'original_split': data.get('split', 'unknown'),
            'github_repository': repository,
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
    
    def _clean_function_code(self, data: Dict[str, Any]) -> str:
        """Extract and clean the function code.
        
        Handles both formats:
        - HuggingFace: 'func_code_string'
        - Kaggle: 'code' or 'original_string'
        """
        # Try different field names
        code = (data.get('func_code_string') or 
                data.get('code') or 
                data.get('original_string', ''))
        
        # Remove Windows line endings and normalize
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Clean up any leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _get_docstring_from_data(self, data: Dict[str, Any]) -> str:
        """Extract docstring from various data formats."""
        # First try the direct docstring field
        docstring = data.get('func_documentation_string', '').strip()
        if docstring:
            return docstring
        
        # For Kaggle format, try to extract from 'docstring' field
        docstring = data.get('docstring', '').strip()
        if docstring:
            return docstring
        
        # If not found, try to extract from the code itself
        code = data.get('original_string') or data.get('code', '')
        if '"""' in code:
            start = code.find('"""') + 3
            end = code.find('"""', start)
            if end > start:
                return code[start:end].strip()
        elif "'''" in code:
            start = code.find("'''") + 3
            end = code.find("'''", start)
            if end > start:
                return code[start:end].strip()
        
        return ''
    
    def _clean_documentation(self, data: Dict[str, Any], func_name: str) -> str:
        """Extract and clean the function documentation to create a problem description."""
        documentation = self._get_docstring_from_data(data)
        
        if not documentation:
            # Fallback: generate basic description from function name
            func_display = func_name.replace('_', ' ').replace('-', ' ').title()
            return f"Implement the function '{func_name}' ({func_display})."
        
        # Clean up the documentation
        # Remove common docstring markers
        documentation = documentation.strip('"""').strip("'''").strip()
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in documentation.split('\n')]
        documentation = '\n'.join(lines).strip()
        
        # Ensure it ends with a period for consistency
        if documentation and not documentation.endswith('.'):
            documentation += '.'
        
        # If documentation is just the function name, expand it
        if documentation.lower().replace('_', '').replace(' ', '') == func_name.lower().replace('_', ''):
            func_display = func_name.replace('_', ' ').replace('-', ' ').title()
            documentation = f"Implement the function '{func_name}' ({func_display})."
        
        return documentation
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the CodeSearchNet dataset structure and statistics."""
        if not self.raw_samples:
            self.load_raw_data()
        
        # Count samples by language and split
        language_counts = {}
        split_counts = {'train': 0, 'test': 0, 'valid': 0, 'unknown': 0}
        code_lengths = []
        documentation_lengths = []
        has_documentation = 0
        repositories = set()
        
        for sample in self.raw_samples:
            data = sample.raw_data
            language = data.get('language', 'unknown')
            split = data.get('split', 'unknown')
            
            # Count by language
            language_counts[language] = language_counts.get(language, 0) + 1
            
            # Count by split
            split_counts[split] = split_counts.get(split, 0) + 1
            
            # Code analysis
            code = data.get('func_code_string', '')
            code_lengths.append(len(code))
            
            # Documentation analysis
            doc = data.get('func_documentation_string', '').strip()
            if doc:
                has_documentation += 1
                documentation_lengths.append(len(doc))
            
            # Repository analysis
            repo = data.get('repository_name', '')
            if repo:
                repositories.add(repo)
        
        analysis = {
            'source': self.source_name,
            'total_samples': len(self.raw_samples),
            'language_distribution': language_counts,
            'original_split_distribution': split_counts,
            'code_length_stats': {
                'min': min(code_lengths) if code_lengths else 0,
                'max': max(code_lengths) if code_lengths else 0,
                'avg': sum(code_lengths) / len(code_lengths) if code_lengths else 0
            },
            'documentation_stats': {
                'samples_with_docs': has_documentation,
                'documentation_coverage': has_documentation / len(self.raw_samples) if self.raw_samples else 0,
                'avg_doc_length': sum(documentation_lengths) / len(documentation_lengths) if documentation_lengths else 0
            },
            'repository_stats': {
                'unique_repositories': len(repositories),
                'avg_samples_per_repo': len(self.raw_samples) / len(repositories) if repositories else 0
            },
            'data_path': str(self.data_path),
            'languages_processed': self.languages,
            'max_samples_per_language': self.max_samples_per_language
        }
        
        return analysis
    
    def _compute_data_statistics(self) -> Dict[str, Any]:
        """Compute CodeSearchNet-specific statistics."""
        if not self.raw_samples:
            return super()._compute_data_statistics()
        
        # Language and split distribution
        language_dist = {}
        split_dist = {'train': 0, 'test': 0, 'valid': 0, 'unknown': 0}
        complexity_indicators = []
        
        for sample in self.raw_samples:
            data = sample.raw_data
            language = data.get('language', 'unknown')
            split = data.get('split', 'unknown')
            
            # Language distribution
            language_dist[language] = language_dist.get(language, 0) + 1
            
            # Split distribution
            split_dist[split] = split_dist.get(split, 0) + 1
            
            # Estimate complexity by code length and documentation presence
            code_length = len(data.get('func_code_string', ''))
            has_docs = bool(data.get('func_documentation_string', '').strip())
            complexity_indicators.append(code_length + (100 if has_docs else 0))  # Weighted complexity
        
        stats = super()._compute_data_statistics()
        stats.update({
            'language_distribution': language_dist,
            'original_split_distribution': split_dist,
            'avg_complexity_indicator': sum(complexity_indicators) / len(complexity_indicators) if complexity_indicators else 0,
            'languages_included': self.languages,
            'max_samples_limit': self.max_samples_per_language
        })
        
        return stats
    
    def _validate_metadata(self) -> Dict[str, str]:
        """Validate CodeSearchNet-specific metadata completeness."""
        base_validation = super()._validate_metadata()
        
        # Check CodeSearchNet-specific required metadata fields
        required_fields = ['original_id', 'language', 'function_name', 'repository_name', 'original_split']
        total_samples = len(self.converted_samples)
        
        for field in required_fields:
            count = sum(1 for s in self.converted_samples if field in s.metadata)
            base_validation[f'{field}_completeness'] = f"{count}/{total_samples}"
        
        return base_validation
    
    def get_language_specific_samples(self, language: str) -> List[CodeSample]:
        """Get samples for a specific programming language."""
        if not self.converted_samples:
            self.convert_to_unified_schema()
        
        return [
            sample for sample in self.converted_samples 
            if sample.metadata.get('language') == language
        ]
    
    def get_samples_by_repository(self, repository_pattern: str) -> List[CodeSample]:
        """Get samples from repositories matching a pattern."""
        if not self.converted_samples:
            self.convert_to_unified_schema()
        
        return [
            sample for sample in self.converted_samples 
            if repository_pattern.lower() in sample.metadata.get('repository_name', '').lower()
        ]
    
    def save_converted_dataset(self, output_path: Path,
                              train_ratio: float = 0.8,
                              valid_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save CodeSearchNet dataset with original or custom splits.
        
        If preserve_original_splits=True, uses original train/valid/test splits.
        Otherwise, creates new splits with stratification by language.
        Languages are co-located in the output (sorted by language then by problem_id).
        """
        if not self.converted_samples:
            raise ValueError("No converted samples to save")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.preserve_original_splits:
            # Use original splits
            train_samples = []
            valid_samples = []
            test_samples = []
            
            for sample in self.converted_samples:
                original_split = sample.metadata.get('original_split', 'unknown')
                
                if original_split == 'train':
                    train_samples.append(sample)
                elif original_split == 'valid':
                    valid_samples.append(sample)
                elif original_split == 'test':
                    test_samples.append(sample)
                else:
                    # Default to train if split is unknown
                    train_samples.append(sample)
                    logging.warning(f"Unknown split '{original_split}' for sample {sample.submission_id}, defaulting to train")
            
            logging.info("Using original CodeSearchNet splits")
            
        else:
            # Create new stratified splits by language
            # Group samples by language
            samples_by_language = {}
            for sample in self.converted_samples:
                language = sample.metadata.get('language', 'unknown')
                if language not in samples_by_language:
                    samples_by_language[language] = []
                samples_by_language[language].append(sample)
            
            # Sort samples within each language for reproducibility
            for language in samples_by_language:
                samples_by_language[language].sort(key=lambda x: x.problem_id)
            
            # Calculate split sizes for each language
            train_samples = []
            valid_samples = []
            test_samples = []
            
            for language, lang_samples in samples_by_language.items():
                total = len(lang_samples)
                train_size = int(total * train_ratio)
                valid_size = int(total * valid_ratio)
                
                # Split this language's samples
                lang_train = lang_samples[:train_size]
                lang_valid = lang_samples[train_size:train_size + valid_size]
                lang_test = lang_samples[train_size + valid_size:]
                
                train_samples.extend(lang_train)
                valid_samples.extend(lang_valid)
                test_samples.extend(lang_test)
                
                logging.info(f"Language {language}: Train={len(lang_train)}, Valid={len(lang_valid)}, Test={len(lang_test)}")
        
        # Sort each split to co-locate languages (group by language, then by problem_id)
        # This ensures that problems of the same language are co-located
        train_samples.sort(key=lambda x: (x.metadata.get('language', ''), x.problem_id))
        valid_samples.sort(key=lambda x: (x.metadata.get('language', ''), x.problem_id))
        test_samples.sort(key=lambda x: (x.metadata.get('language', ''), x.problem_id))
        
        logging.info(f"Total split sizes - Train: {len(train_samples)}, Valid: {len(valid_samples)}, Test: {len(test_samples)}")
        
        # Save splits
        output_files = {}
        splits = [
            ('train', train_samples),
            ('valid', valid_samples),
            ('test', test_samples)
        ]
        
        for split_name, samples in splits:
            file_path = output_path / f"{split_name}.jsonl"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + '\n')
            
            output_files[split_name] = file_path
            logging.info(f"Saved {len(samples)} samples to {file_path}")
        
        # Calculate language distribution for metadata
        language_distribution = {}
        for split_name, samples in splits:
            lang_counts = {}
            for sample in samples:
                lang = sample.metadata.get('language', 'unknown')
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            language_distribution[split_name] = lang_counts
        
        # Save metadata
        metadata = {
            'source': self.source_name,
            'total_samples': len(self.converted_samples),
            'languages': self.languages,
            'preserve_original_splits': self.preserve_original_splits,
            'language_distribution': language_distribution,
            'splits': {
                'train_size': len(train_samples),
                'valid_size': len(valid_samples),
                'test_size': len(test_samples)
            }
        }
        
        if not self.preserve_original_splits:
            metadata['splits'].update({
                'train_ratio': train_ratio,
                'valid_ratio': valid_ratio,
                'test_ratio': 1.0 - train_ratio - valid_ratio
            })
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        output_files['metadata'] = metadata_file
        
        logging.info(f"{self.source_name} integration completed. Files saved to {output_path}")
        
        return output_files
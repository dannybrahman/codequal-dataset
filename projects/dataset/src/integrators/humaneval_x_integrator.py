"""
HumanEval-X dataset integration for CodeQual v2.

HumanEval-X is a multilingual benchmark with 164 problems in 5 languages:
Python, C++, Java, JavaScript, and Go.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from .data_integrator import DataSourceIntegrator, RawDataSample
from ..processors import CodeSample, QualityScores


@dataclass
class HumanEvalXSample:
    """Raw HumanEval-X sample structure."""
    task_id: str
    prompt: str
    declaration: str
    canonical_solution: str
    test: str
    example_test: str
    text: str


class HumanEvalXIntegrator(DataSourceIntegrator):
    """Integrates HumanEval-X dataset into unified CodeQual schema."""
    
    SUPPORTED_LANGUAGES = ["python", "cpp", "go", "java", "js"]
    
    def __init__(self, humaneval_x_data_path: str, languages: Optional[List[str]] = None):
        """
        Initialize the HumanEval-X integrator.
        
        Args:
            humaneval_x_data_path: Path to HumanEval-X data directory
            languages: List of languages to include (default: all supported)
        """
        super().__init__(humaneval_x_data_path, "humaneval-x")
        
        # Set languages to process
        if languages is None:
            self.languages = self.SUPPORTED_LANGUAGES.copy()
        else:
            invalid_langs = set(languages) - set(self.SUPPORTED_LANGUAGES)
            if invalid_langs:
                raise ValueError(f"Unsupported languages: {invalid_langs}. "
                               f"Supported: {self.SUPPORTED_LANGUAGES}")
            self.languages = languages
        
        logging.info(f"Initialized HumanEval-X integrator for languages: {self.languages}")
    
    def load_raw_data(self) -> List[RawDataSample]:
        """Load HumanEval-X samples from all specified languages."""
        self.raw_samples = []
        
        for language in self.languages:
            lang_path = self.data_path / "data" / language / "data" / "humaneval.jsonl"
            
            if not lang_path.exists():
                logging.warning(f"Language data not found: {lang_path}")
                continue
            
            logging.info(f"Loading {language} samples from {lang_path}")
            
            with open(lang_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        # Add language info to the data
                        data['language'] = language
                        sample = self._parse_single_sample(data)
                        self.raw_samples.append(sample)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Error parsing {lang_path}:{line_num} - {e}")
        
        logging.info(f"Loaded {len(self.raw_samples)} HumanEval-X samples across {len(self.languages)} languages")
        return self.raw_samples
    
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """Parse a single HumanEval-X sample into RawDataSample format."""
        # Use task_id as source_id, which includes language (e.g., "Python/0")
        source_id = raw_data.get('task_id', f"humaneval-x_{len(self.raw_samples)}")
        return RawDataSample(source_id=source_id, raw_data=raw_data)
    
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, 
                                  sample_index: int) -> CodeSample:
        """Convert a raw HumanEval-X sample to unified CodeSample format."""
        data = raw_sample.raw_data
        language = data['language']
        task_id = data['task_id']
        
        # Extract problem number from task_id (e.g., "Python/0" -> "0")
        problem_num = task_id.split('/')[-1]
        
        # Generate unique IDs
        problem_id = f"humaneval_x_{problem_num}"
        submission_id = f"humaneval_x_{language}_{problem_num}"
        
        # Create problem description from prompt text
        problem_description = self._extract_problem_description(data)
        
        # Get the canonical solution (complete function)
        solution = self._build_complete_solution(data)
        
        # Create metadata
        metadata = {
            'language': language,
            'task_id': task_id,
            'problem_number': problem_num,
            'lines_of_code': len(solution.split('\n')),
            'test_cases': data.get('test', ''),
            'example_test': data.get('example_test', ''),
            'assessment_method': None  # Will be set when quality assessment is performed
        }
        
        # Create unified sample
        return CodeSample(
            problem_id=problem_id,
            problem=problem_description,
            submission_id=submission_id,
            submission=solution,
            source=self.source_name,
            quality_scores=None,  # Will be set during assessment
            metadata=metadata
        )
    
    def _extract_problem_description(self, data: Dict[str, Any]) -> str:
        """Extract clean problem description from HumanEval-X data."""
        # Use the 'text' field which contains just the docstring content
        if 'text' in data and data['text'].strip():
            return data['text'].strip()
        
        # Fallback: extract from prompt based on language
        prompt = data.get('prompt', '')
        language = data.get('language', '')
        
        if language == 'cpp':
            # C++ uses /* */ comments
            if '/*' in prompt and '*/' in prompt:
                start = prompt.find('/*') + 2
                end = prompt.find('*/')
                if start < end:
                    return prompt[start:end].strip()
        
        elif language in ['python', 'java']:
            # Python and Java use """ or ''' 
            if '"""' in prompt:
                parts = prompt.split('"""')
                if len(parts) >= 3:
                    return parts[1].strip()
            elif "'''" in prompt:
                parts = prompt.split("'''")
                if len(parts) >= 3:
                    return parts[1].strip()
        
        elif language == 'js':
            # JavaScript uses /* */ comments
            if '/*' in prompt and '*/' in prompt:
                start = prompt.find('/*') + 2
                end = prompt.find('*/')
                if start < end:
                    return prompt[start:end].strip()
        
        elif language == 'go':
            # Go uses // comments - extract lines that start with //
            lines = prompt.split('\n')
            comment_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('//'):
                    # Remove the // and any leading space
                    comment_text = line[2:].strip()
                    if comment_text:  # Skip empty comment lines
                        comment_lines.append(comment_text)
                elif comment_lines:  # Stop when we hit non-comment line after comments
                    break
            
            if comment_lines:
                return '\n'.join(comment_lines)
        
        # Last resort: use declaration as problem description
        return f"Implement the function: {data.get('declaration', '').strip()}"
    
    def _build_complete_solution(self, data: Dict[str, Any]) -> str:
        """Build complete function solution from HumanEval-X data."""
        declaration = data.get('declaration', '').strip()
        canonical_solution = data.get('canonical_solution', '').strip()
        
        # For most languages, we need to combine declaration + solution
        if data['language'] == 'python':
            # Python: combine declaration with solution
            if declaration and canonical_solution:
                return declaration + canonical_solution
            return canonical_solution
        
        elif data['language'] == 'cpp':
            # C++: usually complete function in canonical_solution
            return canonical_solution
        
        elif data['language'] == 'java':
            # Java: usually complete function in canonical_solution
            return canonical_solution
        
        elif data['language'] == 'js':
            # JavaScript: combine declaration with solution
            if declaration and canonical_solution:
                return declaration + canonical_solution
            return canonical_solution
        
        elif data['language'] == 'go':
            # Go: usually complete function in canonical_solution
            return canonical_solution
        
        # Default: return canonical_solution
        return canonical_solution
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the HumanEval-X dataset structure and statistics."""
        if not self.raw_samples:
            self.load_raw_data()
        
        # Count samples by language
        language_counts = {}
        for sample in self.raw_samples:
            lang = sample.raw_data.get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Calculate average solution length by language
        avg_lengths = {}
        for lang in language_counts:
            lang_samples = [s for s in self.raw_samples if s.raw_data.get('language') == lang]
            if lang_samples:
                total_length = sum(len(s.raw_data.get('canonical_solution', '')) 
                                 for s in lang_samples)
                avg_lengths[lang] = total_length / len(lang_samples)
        
        analysis = {
            'source': self.source_name,
            'total_samples': len(self.raw_samples),
            'languages': list(language_counts.keys()),
            'samples_per_language': language_counts,
            'average_solution_length': avg_lengths,
            'problems_per_language': 164,  # HumanEval-X has 164 problems per language
            'unique_problems': 164,  # All languages have the same 164 problems
            'data_path': str(self.data_path)
        }
        
        return analysis
    
    def save_converted_dataset(self, output_path: Path,
                              train_ratio: float = 0.8,
                              valid_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save HumanEval-X dataset with stratified splitting by language.
        
        Ensures equal distribution of each language across train/valid/test splits.
        """
        if not self.converted_samples:
            raise ValueError("No converted samples to save")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group samples by language
        samples_by_language = {}
        for sample in self.converted_samples:
            language = sample.metadata.get('language', 'unknown')
            if language not in samples_by_language:
                samples_by_language[language] = []
            samples_by_language[language].append(sample)
        
        # Sort samples within each language by problem number for reproducibility
        for language in samples_by_language:
            samples_by_language[language].sort(
                key=lambda x: int(x.metadata.get('problem_number', '0'))
            )
        
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
        
        # Sort each split by problem_id for consistent ordering (languages will be colocated)
        train_samples.sort(key=lambda x: x.problem_id)
        valid_samples.sort(key=lambda x: x.problem_id)
        test_samples.sort(key=lambda x: x.problem_id)
        
        logging.info(f"Total split sizes - Train: {len(train_samples)}, Valid: {len(valid_samples)}, Test: {len(test_samples)}")
        
        # Save splits using the parent class method structure
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
        
        # Save metadata
        metadata = {
            'source': self.source_name,
            'total_samples': len(self.converted_samples),
            'languages': list(samples_by_language.keys()),
            'samples_per_language': {lang: len(samples) for lang, samples in samples_by_language.items()},
            'splits': {
                'train_ratio': train_ratio,
                'valid_ratio': valid_ratio,
                'test_ratio': 1.0 - train_ratio - valid_ratio,
                'train_size': len(train_samples),
                'valid_size': len(valid_samples),
                'test_size': len(test_samples)
            },
            'stratified_by_language': True
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        output_files['metadata'] = metadata_file
        
        logging.info(f"{self.source_name} integration completed. Files saved to {output_path}")
        
        return output_files
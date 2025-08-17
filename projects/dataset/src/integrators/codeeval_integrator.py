"""
CodeEval dataset integration for CodeQual v2.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .data_integrator import DataSourceIntegrator, RawDataSample
from ..processors import CodeSample, QualityScores


@dataclass
class CodeEvalSample:
    """Raw CodeEval sample structure."""
    problem: str
    name: str
    object: str
    topic: str
    complexity: int
    canonical_solution: str
    tests: str
    task_id: str


class CodeEvalIntegrator(DataSourceIntegrator):
    """Integrates CodeEval dataset into unified CodeQual schema."""
    
    def __init__(self, codeeval_data_path: str):
        """Initialize the CodeEval integrator."""
        super().__init__(codeeval_data_path, "codeeval")
    
    def load_raw_data(self) -> List[RawDataSample]:
        """Load all CodeEval JSONL files from the dataset directory."""
        self.raw_samples = []
        
        # Find all JSONL files in the directory
        jsonl_files = list(self.data_path.glob("*.jsonl"))
        
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in {self.data_path}")
        
        logging.info(f"Found {len(jsonl_files)} category files")
        
        for file_path in jsonl_files:
            category_name = file_path.stem
            logging.info(f"Loading category: {category_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        sample = self._parse_single_sample(data)
                        self.raw_samples.append(sample)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Error parsing {file_path}:{line_num} - {e}")
        
        logging.info(f"Loaded {len(self.raw_samples)} CodeEval samples")
        return self.raw_samples
    
    def _parse_single_sample(self, raw_data: Dict[str, Any]) -> RawDataSample:
        """Parse a single CodeEval sample into RawDataSample format."""
        source_id = raw_data.get('task_id', f"codeeval_{len(self.raw_samples)}")
        return RawDataSample(source_id=source_id, raw_data=raw_data)
    
    def _convert_to_unified_sample(self, raw_sample: RawDataSample, 
                                  sample_index: int) -> CodeSample:
        """Convert a raw CodeEval sample to unified CodeSample format."""
        data = raw_sample.raw_data
        
        # Generate unique IDs
        problem_id = f"codeeval_{data['topic']}_{data['task_id']}"
        submission_id = f"codeeval_sub_{sample_index:05d}"
        
        # Create enhanced problem description
        problem_description = self._enhance_problem_description(data)
        
        # Create metadata
        metadata = {
            'original_complexity': data['complexity'],
            'topic': data['topic'],
            'function_name': data['name'],
            'object_type': data['object'],
            'tests': data['tests'],
            'task_id': data['task_id'],
            'language': 'python',
            'lines_of_code': len(data['canonical_solution'].split('\n')),
            'assessment_method': None  # Will be set when quality assessment is performed
        }
        
        # Create unified sample
        return CodeSample(
            problem_id=problem_id,
            problem=problem_description,
            submission_id=submission_id,
            submission=data['canonical_solution'],
            quality_scores=None,  # Will be filled by quality assessment
            source='codeeval',
            metadata=metadata
        )
    
    def _enhance_problem_description(self, data: Dict[str, Any]) -> str:
        """Create clean problem description from CodeEval data."""
        # Return only the core problem description
        # Function name, topic, and complexity are stored in metadata
        return data['problem'].strip()
    
    def _parse_test_cases(self, tests_str: str) -> List[str]:
        """Parse test cases string to extract examples."""
        try:
            tests = json.loads(tests_str)
            examples = []
            
            for test in tests[:5]:  # Limit to first 5 tests
                if 'assertion' in test:
                    assertion = test['assertion']
                    # Clean up the assertion for readability
                    if 'func(' in assertion:
                        examples.append(assertion.replace('func(', f'{tests[0].get("ctx", "")}('))
            
            return examples
        except (json.JSONDecodeError, TypeError):
            return []
    
    def _default_quality_assessment(self, raw_sample: RawDataSample) -> QualityScores:
        """
        Convert CodeEval complexity to initial quality scores.
        
        This provides reasonable baseline scores that can be refined by GPT-4.
        """
        complexity = raw_sample.raw_data.get('complexity', 2)
        
        if complexity == 1:
            # Basic functionality, likely high on idiomatic and readability
            return QualityScores(
                functionality=4.0,      # Should work correctly for basic tasks
                readability=4.0,        # Simple code is usually readable
                idiomatic=4.0,          # CodeEval canonical solutions are clean
                error_handling=2.5,     # Basic solutions may lack error handling
                efficiency=3.5          # Decent but not necessarily optimal
            )
        elif complexity == 2:
            # Intermediate functionality, balanced scores
            return QualityScores(
                functionality=4.0,      # Should be correct
                readability=3.5,        # May be more complex to read
                idiomatic=4.0,          # Still follows good practices
                error_handling=3.0,     # May have some error handling
                efficiency=3.5          # Reasonable efficiency
            )
        else:  # complexity == 3
            # Advanced functionality, potentially complex code
            return QualityScores(
                functionality=4.5,      # Advanced solutions should be robust
                readability=3.0,        # May be harder to read due to complexity
                idiomatic=4.0,          # Advanced patterns should be idiomatic
                error_handling=3.5,     # Advanced code often handles edge cases
                efficiency=4.0          # Focus on performance for complex problems
            )
    
    def _compute_data_statistics(self) -> Dict[str, Any]:
        """Compute CodeEval-specific statistics."""
        if not self.raw_samples:
            return super()._compute_data_statistics()
        
        # Topic distribution
        topic_dist = {}
        complexity_dist = {1: 0, 2: 0, 3: 0}
        code_lengths = []
        function_patterns = []
        
        for sample in self.raw_samples:
            data = sample.raw_data
            
            # Topic distribution
            topic = data.get('topic', 'unknown')
            topic_dist[topic] = topic_dist.get(topic, 0) + 1
            
            # Complexity distribution
            complexity = data.get('complexity', 2)
            if complexity in complexity_dist:
                complexity_dist[complexity] += 1
            
            # Code analysis
            canonical_solution = data.get('canonical_solution', '')
            code_lengths.append(len(canonical_solution))
            
            # Extract function patterns
            if 'def ' in canonical_solution:
                function_patterns.append('function')
            if 'class ' in canonical_solution:
                function_patterns.append('class')
            if 'import ' in canonical_solution:
                function_patterns.append('import')
        
        stats = super()._compute_data_statistics()
        stats.update({
            'topic_distribution': dict(sorted(topic_dist.items())),
            'complexity_distribution': complexity_dist,
            'code_length_stats': {
                'min': min(code_lengths) if code_lengths else 0,
                'max': max(code_lengths) if code_lengths else 0,
                'avg': sum(code_lengths) / len(code_lengths) if code_lengths else 0
            },
            'unique_topics': len(topic_dist),
            'avg_samples_per_topic': len(self.raw_samples) / len(topic_dist) if topic_dist else 0
        })
        
        return stats
    
    def _validate_metadata(self) -> Dict[str, str]:
        """Validate CodeEval-specific metadata completeness."""
        base_validation = super()._validate_metadata()
        
        # Check CodeEval-specific required metadata fields
        required_fields = ['topic', 'function_name', 'original_complexity', 'tests']
        total_samples = len(self.converted_samples)
        
        for field in required_fields:
            count = sum(1 for s in self.converted_samples if field in s.metadata)
            base_validation[f'{field}_completeness'] = f"{count}/{total_samples}"
        
        return base_validation
    
    def save_converted_dataset(self, output_path: Path,
                              train_ratio: float = 0.8,
                              valid_ratio: float = 0.1) -> Dict[str, Path]:
        """
        Save CodeEval dataset with stratified splitting by topic/category.
        
        Ensures proportional representation of each topic across train/valid/test splits.
        """
        if not self.converted_samples:
            raise ValueError("No converted samples to save")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group samples by topic
        samples_by_topic = {}
        for sample in self.converted_samples:
            topic = sample.metadata.get('topic', 'unknown')
            if topic not in samples_by_topic:
                samples_by_topic[topic] = []
            samples_by_topic[topic].append(sample)
        
        # Sort samples within each topic by problem_id for reproducibility
        for topic in samples_by_topic:
            samples_by_topic[topic].sort(key=lambda x: x.problem_id)
        
        # Calculate split sizes for each topic
        train_samples = []
        valid_samples = []
        test_samples = []
        
        for topic, topic_samples in samples_by_topic.items():
            total = len(topic_samples)
            train_size = int(total * train_ratio)
            valid_size = int(total * valid_ratio)
            
            # Split this topic's samples
            topic_train = topic_samples[:train_size]
            topic_valid = topic_samples[train_size:train_size + valid_size]
            topic_test = topic_samples[train_size + valid_size:]
            
            train_samples.extend(topic_train)
            valid_samples.extend(topic_valid)
            test_samples.extend(topic_test)
            
            logging.info(f"Topic {topic}: Train={len(topic_train)}, Valid={len(topic_valid)}, Test={len(topic_test)}")
        
        # Sort each split by problem_id for consistent ordering (topics will be colocated)
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
            'topics': list(samples_by_topic.keys()),
            'samples_per_topic': {topic: len(samples) for topic, samples in samples_by_topic.items()},
            'splits': {
                'train_ratio': train_ratio,
                'valid_ratio': valid_ratio,
                'test_ratio': 1.0 - train_ratio - valid_ratio,
                'train_size': len(train_samples),
                'valid_size': len(valid_samples),
                'test_size': len(test_samples)
            },
            'stratified_by_topic': True
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        output_files['metadata'] = metadata_file
        
        logging.info(f"{self.source_name} integration completed. Files saved to {output_path}")
        
        return output_files
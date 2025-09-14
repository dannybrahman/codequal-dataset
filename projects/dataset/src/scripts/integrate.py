#!/usr/bin/env python3
"""
Data integration workflow script.

Transforms source datasets into unified CodeQual format with deterministic output paths.

Usage:
    python scripts/integrate.py --source codeeval --input-path /path/to/codeeval
    python scripts/integrate.py --source codecontests --input-path /path/to/codecontests
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from ..integrators import CodeEvalIntegrator, HumanEvalXIntegrator, MBPPIntegrator, CodeSearchNetIntegrator
from ..integrators.codenet_integrator import CodeNetIntegrator


def run_integration(source: str, input_path: str, test_samples_file: str = None,
                   train_ratio: float = 0.8, valid_ratio: float = 0.1,
                   min_description_length: int = None, max_description_length: int = None,
                   min_lines_of_code: int = None, max_lines_of_code: int = None,
                   random_sample: int = None, skip_flagged: bool = False,
                   languages: List[str] = None) -> Dict[str, Any]:
    """
    Run data integration workflow with deterministic output paths.
    
    Args:
        source: Source dataset name (codeeval, codecontests, mbpp, etc.)
        input_path: Path to source dataset
        train_ratio: Training set ratio
        valid_ratio: Validation set ratio
        
    Returns:
        Dict with integration results and paths
    """
    logging.info(f"Starting integration workflow for {source}")
    
    # Deterministic output path
    output_path = Path("generated") / "integrated" / source
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create appropriate integrator
    if source == 'codeeval':
        integrator = CodeEvalIntegrator(input_path)
    elif source == 'codenet':
        # CodeNet integrator with optional test samples file
        if test_samples_file:
            logging.info(f"Using test samples from {test_samples_file}")
            integrator = CodeNetIntegrator(input_path, test_samples_file=test_samples_file)
        else:
            integrator = CodeNetIntegrator(input_path)
    elif source == 'humaneval-x':
        if languages:
            integrator = HumanEvalXIntegrator(input_path, languages=languages)
        else:
            integrator = HumanEvalXIntegrator(input_path)
    elif source == 'codecontests':
        # Future: CodeContestsIntegrator(input_path)
        raise NotImplementedError(f"CodeContests integration not yet implemented")
    elif source == 'mbpp':
        integrator = MBPPIntegrator(input_path)
    elif source == 'codesearchnet':
        # CodeSearchNet integrator with filtering options
        # Build kwargs dynamically to avoid passing None values
        kwargs = {
            'languages': languages if languages else ["python", "javascript", "java", "go"]
        }
        
        # Add filtering parameters only if they're not None
        if min_description_length is not None:
            kwargs['min_description_length'] = min_description_length
        if max_description_length is not None:
            kwargs['max_description_length'] = max_description_length
        if min_lines_of_code is not None:
            kwargs['min_lines_of_code'] = min_lines_of_code
        if max_lines_of_code is not None:
            kwargs['max_lines_of_code'] = max_lines_of_code
        if random_sample is not None:
            kwargs['random_sample'] = random_sample
        if skip_flagged:
            kwargs['skip_flagged'] = True
        
        integrator = CodeSearchNetIntegrator(input_path, **kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")
    
    # Load and convert data
    logging.info("Loading raw data...")
    raw_samples = integrator.load_raw_data()
    logging.info(f"Loaded {len(raw_samples)} raw samples")
    
    # Analyze dataset before conversion
    analysis = integrator.analyze_dataset()
    logging.info(f"Dataset analysis: {analysis['total_samples']} samples")
    
    logging.info("Converting to unified schema...")
    samples = integrator.convert_to_unified_schema(initial_quality_assessment=False)
    logging.info(f"Converted {len(samples)} samples to unified format")
    
    # Save with deterministic splits
    logging.info(f"Saving integrated data to {output_path}")
    output_files = integrator.save_converted_dataset(
        output_path=output_path,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio
    )
    
    # Validate the conversion
    validation_results = integrator.validate_conversion()
    
    results = {
        'source': source,
        'input_path': input_path,
        'output_path': str(output_path),
        'output_files': {k: str(v) for k, v in output_files.items()},
        'samples_converted': len(samples),
        'analysis': analysis,
        'validation': validation_results,
        'splits': {
            'train_ratio': train_ratio,
            'valid_ratio': valid_ratio,
            'test_ratio': 1.0 - train_ratio - valid_ratio
        }
    }
    
    logging.info("Integration workflow completed successfully!")
    logging.info(f"Files saved:")
    for split_name, file_path in output_files.items():
        if split_name != 'metadata':
            logging.info(f"  {split_name}: {file_path}")
    
    logging.info(f"Next step: python main.py run-assessment --source {source} --model <model-name>")
    
    return results


def get_available_sources() -> list:
    """Get list of available source integrators."""
    return ['codeeval', 'codenet', 'humaneval-x', 'mbpp', 'codesearchnet']  # Add more as they're implemented


def validate_source_data(source: str, input_path: str) -> bool:
    """
    Validate that the source data exists and is in expected format.
    
    Args:
        source: Source dataset name
        input_path: Path to source dataset
        
    Returns:
        True if valid, False otherwise
    """
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
    
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_path}")
        return False
    
    if source == 'codeeval':
        # Check for JSONL files
        jsonl_files = list(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            logging.error(f"No JSONL files found in CodeEval directory: {input_path}")
            return False
        logging.info(f"Found {len(jsonl_files)} JSONL files in CodeEval directory")
        
    elif source == 'codenet':
        # Check for required CodeNet directories
        required_dirs = ['problem_descriptions', 'metadata', 'python_800']
        for req_dir in required_dirs:
            dir_path = input_dir / req_dir
            if not dir_path.exists():
                logging.error(f"Required CodeNet directory not found: {dir_path}")
                return False
        logging.info("Found all required CodeNet directories")
        
    elif source == 'humaneval-x':
        # Check for data directory structure
        data_dir = input_dir / "data"
        if not data_dir.exists():
            logging.error(f"HumanEval-X data directory not found: {data_dir}")
            return False
        
        # Check for at least one language directory
        language_dirs = ['python', 'cpp', 'go', 'java', 'js']
        found_langs = []
        for lang in language_dirs:
            lang_file = data_dir / lang / "data" / "humaneval.jsonl"
            if lang_file.exists():
                found_langs.append(lang)
        
        if not found_langs:
            logging.error(f"No HumanEval-X language data found in: {data_dir}")
            return False
        
        logging.info(f"Found HumanEval-X data for languages: {found_langs}")
        
    elif source == 'codesearchnet':
        # Check for any JSONL files or Kaggle structure
        jsonl_files = list(input_dir.glob("*.jsonl"))
        kaggle_structure = input_dir / "python" / "python" / "final" / "jsonl"
        
        if jsonl_files:
            logging.info(f"Found {len(jsonl_files)} JSONL files in CodeSearchNet directory")
        elif kaggle_structure.exists():
            logging.info("Found Kaggle CodeSearchNet structure")
        else:
            logging.error(f"No CodeSearchNet data found in: {input_dir}")
            logging.error("Expected either *.jsonl files or Kaggle structure (language/language/final/jsonl/)")
            return False
    
    # Add validation for other sources as needed
    
    return True
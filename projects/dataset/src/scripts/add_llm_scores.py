#!/usr/bin/env python3
"""
Add LLM assessment scores to integrated datasets.

This script reads LLM assessment scores from a session directory and updates
the specified split (train/valid/test) with the LLM scores.

Usage:
    python main.py add-llm-scores --input-path projects/llm_benchmarks/generated/session_XXX --split test
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ..processors.unified_schema import CodeSample, QualityScores


def validate_llm_scores_setup(input_path: str) -> bool:
    """Validate that LLM scores directory exists and has expected structure."""
    input_dir = Path(input_path)

    if not input_dir.exists():
        logging.error(f"LLM scores directory not found: {input_path}")
        return False

    if not input_dir.is_dir():
        logging.error(f"LLM scores path is not a directory: {input_path}")
        return False

    # Check if there are any model subdirectories
    model_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        logging.error(f"No model directories found in: {input_path}")
        return False

    logging.info(f"LLM scores setup validation passed")
    logging.info(f"Found {len(model_dirs)} model directories")
    return True


def load_llm_scores_from_session(input_path: str, source: str) -> Dict[str, Dict[str, QualityScores]]:
    """
    Load LLM scores from a session directory for a specific source.

    Args:
        input_path: Path to session directory (e.g., projects/llm_benchmarks/generated/session_XXX)
        source: Source dataset name (e.g., 'codeeval', 'mbpp')

    Returns:
        Dictionary mapping model_name -> {submission_id -> QualityScores}
    """
    session_dir = Path(input_path)
    model_scores = {}

    # Iterate through model directories
    for model_dir in session_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Look for source JSONL file (e.g., codeeval.jsonl, mbpp.jsonl)
        source_file = model_dir / f"{source}.jsonl"

        if not source_file.exists():
            logging.debug(f"No scores file for {source} in model {model_name}")
            continue

        # Load scores from the file
        scores_dict = {}
        with open(source_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Extract sample_id (which corresponds to submission_id in the dataset)
                    sample_id = data.get('sample_id')

                    if not sample_id:
                        logging.warning(f"{model_name}/{source}.jsonl line {line_num}: Missing sample_id")
                        continue

                    # Extract quality scores
                    if 'quality_scores' not in data:
                        logging.warning(f"{model_name}/{source}.jsonl line {line_num}: Missing quality_scores")
                        continue

                    qs = data['quality_scores']

                    # Validate all dimensions are present
                    required_dims = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
                    if not all(dim in qs for dim in required_dims):
                        logging.warning(f"{model_name}/{source}.jsonl line {line_num}: Missing quality dimensions")
                        continue

                    # Create QualityScores object
                    quality_scores = QualityScores(
                        functionality=float(qs['functionality']) if qs['functionality'] is not None else None,
                        readability=float(qs['readability']) if qs['readability'] is not None else None,
                        idiomatic=float(qs['idiomatic']) if qs['idiomatic'] is not None else None,
                        error_handling=float(qs['error_handling']) if qs['error_handling'] is not None else None,
                        efficiency=float(qs['efficiency']) if qs['efficiency'] is not None else None
                    )

                    # Only add if all scores are present
                    if all([quality_scores.functionality is not None,
                           quality_scores.readability is not None,
                           quality_scores.idiomatic is not None,
                           quality_scores.error_handling is not None,
                           quality_scores.efficiency is not None]):
                        # Use sample_id as the key (matches submission_id in dataset)
                        scores_dict[sample_id] = quality_scores
                    else:
                        logging.warning(f"{model_name}/{source}.jsonl line {line_num}: Incomplete scores")

                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logging.warning(f"{model_name}/{source}.jsonl line {line_num}: Error parsing: {e}")
                    continue

        if scores_dict:
            model_scores[model_name] = scores_dict
            logging.info(f"Loaded {len(scores_dict)} scores from model: {model_name}")
        else:
            logging.warning(f"No valid scores found for model: {model_name}")

    return model_scores


def update_dataset_with_llm_scores(
    source: str,
    split: str,
    model_scores: Dict[str, Dict[str, QualityScores]]
) -> Dict[str, Any]:
    """
    Update dataset split with LLM scores.

    Args:
        source: Source dataset name (e.g., 'codeeval')
        split: Dataset split ('train', 'valid', or 'test')
        model_scores: Dictionary mapping model_name -> {submission_id -> QualityScores}

    Returns:
        Dictionary with update statistics
    """
    # Path to integrated data
    integrated_path = Path("generated") / "integrated" / source
    split_file = integrated_path / f"{split}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    logging.info(f"Updating {split} set with LLM scores: {split_file}")

    # Load existing samples
    samples = []
    with open(split_file, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = CodeSample.from_dict(sample_dict)
                samples.append(sample)

    logging.info(f"Loaded {len(samples)} samples from {split} set")

    # Update samples with LLM scores
    samples_updated = 0
    models_integrated = list(model_scores.keys())

    for sample in samples:
        # Match by submission_id
        submission_id = sample.submission_id

        # Initialize llm_scores list if not present
        if not hasattr(sample, 'llm_scores') or sample.llm_scores is None:
            sample.llm_scores = []

        # Track if this sample was updated
        sample_had_scores = len(sample.llm_scores) > 0

        # Process each model's scores
        for model_name, scores_dict in model_scores.items():
            if submission_id in scores_dict:
                # Remove existing scores for this model (overwrite)
                sample.llm_scores = [
                    score for score in sample.llm_scores
                    if score.get('model_name') != model_name
                ]

                # Add new scores
                llm_score_entry = {
                    'model_name': model_name,
                    'scores': scores_dict[submission_id].to_dict()
                }
                sample.llm_scores.append(llm_score_entry)

        # Check if sample was updated (has any llm_scores)
        if sample.llm_scores and not sample_had_scores:
            samples_updated += 1

    # Create backup
    backup_file = split_file.with_suffix(f'.backup_{split}.jsonl')
    logging.info(f"Creating backup: {backup_file}")
    shutil.copy2(split_file, backup_file)

    # Write updated samples
    with open(split_file, 'w') as f:
        for sample in samples:
            f.write(sample.to_jsonl_line() + '\n')

    results = {
        'source': source,
        'split': split,
        'updated_file': str(split_file),
        'backup_file': str(backup_file),
        'total_samples': len(samples),
        'samples_updated': samples_updated,
        'models': models_integrated,
        'coverage_percentage': (samples_updated / len(samples) * 100) if samples else 0
    }

    logging.info(f"LLM score update completed:")
    logging.info(f"  Total samples: {results['total_samples']}")
    logging.info(f"  Samples updated: {results['samples_updated']}")
    logging.info(f"  Models integrated: {', '.join(results['models'])}")
    logging.info(f"  Coverage: {results['coverage_percentage']:.1f}%")
    logging.info(f"  Backup created: {backup_file}")

    return results


def add_llm_scores_to_datasets(
    input_path: str,
    split: str,
    sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main function to add LLM scores to integrated datasets.

    Args:
        input_path: Path to LLM scores session directory
        split: Dataset split to update ('train', 'valid', or 'test')
        sources: List of sources to process (if None, auto-detect from input_path)

    Returns:
        Dictionary with operation results
    """
    input_dir = Path(input_path)

    # Auto-detect sources if not specified
    if not sources:
        sources = set()
        for model_dir in input_dir.iterdir():
            if not model_dir.is_dir():
                continue
            # Find all JSONL files in model directory
            for jsonl_file in model_dir.glob("*.jsonl"):
                source_name = jsonl_file.stem
                sources.add(source_name)
        sources = sorted(list(sources))
        logging.info(f"Auto-detected sources: {', '.join(sources)}")

    # Process each source
    source_results = []
    total_models = 0
    total_samples_updated = 0

    for source in sources:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing source: {source}")
        logging.info(f"{'='*60}")

        # Check if integrated data exists
        integrated_path = Path("generated") / "integrated" / source
        if not integrated_path.exists():
            logging.warning(f"Integrated data not found for source: {source}")
            logging.warning(f"Run integration first: python main.py integrate --source {source}")
            continue

        # Load LLM scores for this source
        model_scores = load_llm_scores_from_session(input_path, source)

        if not model_scores:
            logging.warning(f"No LLM scores found for source: {source}")
            continue

        # Update dataset
        try:
            results = update_dataset_with_llm_scores(source, split, model_scores)
            source_results.append(results)
            total_models = max(total_models, len(results['models']))
            total_samples_updated += results['samples_updated']
        except Exception as e:
            logging.error(f"Failed to update {source}: {e}")
            continue

    return {
        'sources_processed': len(source_results),
        'total_models': total_models,
        'total_samples_updated': total_samples_updated,
        'source_results': source_results
    }

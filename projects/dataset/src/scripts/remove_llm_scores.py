#!/usr/bin/env python3
"""
Remove LLM assessment scores from integrated datasets.

This script removes LLM assessment scores from the specified split (train/valid/test).
It can remove scores from all models or specific models.

Usage:
    python main.py remove-llm-scores --split test
    python main.py remove-llm-scores --split test --models gpt-4 claude-3-sonnet
    python main.py remove-llm-scores --split test --sources codeeval mbpp
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..processors.unified_schema import CodeSample


def get_available_sources() -> List[str]:
    """Get list of available integrated sources."""
    integrated_path = Path("generated") / "integrated"
    if not integrated_path.exists():
        return []

    sources = []
    for source_dir in integrated_path.iterdir():
        if source_dir.is_dir() and not source_dir.name.startswith('.'):
            # Check if it has any split files
            if any(source_dir.glob("*.jsonl")):
                sources.append(source_dir.name)

    return sorted(sources)


def get_models_in_dataset(source: str, split: str) -> List[str]:
    """Get list of models that have scores in the dataset."""
    integrated_path = Path("generated") / "integrated" / source
    split_file = integrated_path / f"{split}.jsonl"

    if not split_file.exists():
        return []

    models = set()
    with open(split_file, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                llm_scores = sample_dict.get('llm_scores', [])
                if llm_scores:
                    for score_entry in llm_scores:
                        model_name = score_entry.get('model_name')
                        if model_name:
                            models.add(model_name)

    return sorted(list(models))


def remove_llm_scores_from_dataset(
    source: str,
    split: str,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Remove LLM scores from a dataset split.

    Args:
        source: Source dataset name (e.g., 'codeeval')
        split: Dataset split ('train', 'valid', or 'test')
        models: List of model names to remove. If None, removes all LLM scores.

    Returns:
        Dictionary with removal statistics
    """
    # Path to integrated data
    integrated_path = Path("generated") / "integrated" / source
    split_file = integrated_path / f"{split}.jsonl"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    logging.info(f"Removing LLM scores from {split} set: {split_file}")
    if models:
        logging.info(f"Target models: {', '.join(models)}")
    else:
        logging.info("Removing ALL LLM scores")

    # Load existing samples
    samples = []
    with open(split_file, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = CodeSample.from_dict(sample_dict)
                samples.append(sample)

    logging.info(f"Loaded {len(samples)} samples from {split} set")

    # Track statistics
    samples_modified = 0
    models_removed = set()
    scores_removed = 0

    for sample in samples:
        # Skip if no llm_scores
        if not hasattr(sample, 'llm_scores') or sample.llm_scores is None:
            continue

        original_count = len(sample.llm_scores)

        if models:
            # Remove only specified models
            new_scores = []
            for score_entry in sample.llm_scores:
                model_name = score_entry.get('model_name')
                if model_name in models:
                    models_removed.add(model_name)
                else:
                    new_scores.append(score_entry)

            if len(new_scores) < original_count:
                sample.llm_scores = new_scores if new_scores else None
                samples_modified += 1
                scores_removed += original_count - len(new_scores)
        else:
            # Remove all LLM scores
            if sample.llm_scores:
                for score_entry in sample.llm_scores:
                    model_name = score_entry.get('model_name')
                    if model_name:
                        models_removed.add(model_name)
                scores_removed += len(sample.llm_scores)
                sample.llm_scores = None
                samples_modified += 1

    # Create backup
    backup_file = split_file.with_suffix(f'.backup_before_remove.jsonl')
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
        'samples_modified': samples_modified,
        'scores_removed': scores_removed,
        'models_removed': sorted(list(models_removed)),
        'models_requested': models if models else 'all'
    }

    logging.info(f"LLM score removal completed:")
    logging.info(f"  Total samples: {results['total_samples']}")
    logging.info(f"  Samples modified: {results['samples_modified']}")
    logging.info(f"  Scores removed: {results['scores_removed']}")
    logging.info(f"  Models removed: {', '.join(results['models_removed']) if results['models_removed'] else 'none'}")
    logging.info(f"  Backup created: {backup_file}")

    return results


def remove_llm_scores_from_datasets(
    split: str,
    sources: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main function to remove LLM scores from integrated datasets.

    Args:
        split: Dataset split to update ('train', 'valid', or 'test')
        sources: List of sources to process (if None, auto-detect from integrated directory)
        models: List of model names to remove (if None, removes all LLM scores)

    Returns:
        Dictionary with operation results
    """
    # Auto-detect sources if not specified
    if not sources:
        sources = get_available_sources()
        if not sources:
            logging.error("No integrated datasets found in generated/integrated/")
            return {
                'sources_processed': 0,
                'total_samples_modified': 0,
                'total_scores_removed': 0,
                'source_results': []
            }
        logging.info(f"Auto-detected sources: {', '.join(sources)}")

    # Process each source
    source_results = []
    total_samples_modified = 0
    total_scores_removed = 0
    all_models_removed = set()

    for source in sources:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing source: {source}")
        logging.info(f"{'='*60}")

        # Check if integrated data exists
        integrated_path = Path("generated") / "integrated" / source
        if not integrated_path.exists():
            logging.warning(f"Integrated data not found for source: {source}")
            continue

        # Check if split file exists
        split_file = integrated_path / f"{split}.jsonl"
        if not split_file.exists():
            logging.warning(f"Split file not found: {split_file}")
            continue

        # Remove LLM scores
        try:
            results = remove_llm_scores_from_dataset(source, split, models)
            source_results.append(results)
            total_samples_modified += results['samples_modified']
            total_scores_removed += results['scores_removed']
            all_models_removed.update(results['models_removed'])
        except Exception as e:
            logging.error(f"Failed to process {source}: {e}")
            continue

    return {
        'sources_processed': len(source_results),
        'total_samples_modified': total_samples_modified,
        'total_scores_removed': total_scores_removed,
        'all_models_removed': sorted(list(all_models_removed)),
        'source_results': source_results
    }

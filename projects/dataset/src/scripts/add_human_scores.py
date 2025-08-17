#!/usr/bin/env python3
"""
Add human assessment scores to integrated test set.

This script reads human assessment scores from a JSONL file and updates the test.jsonl
in the integrated dataset with the corresponding quality scores.

Usage:
    python scripts/add_human_scores.py --source codenet --input-path human_scores_detailed.jsonl --method mean
"""

import json
import csv
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple

from ..processors.unified_schema import CodeSample, QualityScores


def load_human_scores(input_file: str, aggregation_method: str = 'mean') -> Dict[Tuple[str, str], QualityScores]:
    """
    Load human assessment scores from JSONL file with individual annotator scores.
    
    Args:
        input_file: Path to JSONL with format:
                   {"problem_id": "...", "submission_id": "...", "scores": [{"annotator_id": "...", "quality_scores": {...}}]}
        aggregation_method: How to aggregate multiple annotator scores ('mean' or 'median')
    
    Returns:
        Dictionary mapping (problem_id, submission_id) to QualityScores
    """
    scores_dict = {}
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Human scores file not found: {input_file}")
    
    logging.info(f"Loading human assessment scores from {input_path} with {aggregation_method} aggregation")
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                problem_id = data['problem_id']
                submission_id = data['submission_id']
                annotator_scores = data['scores']
                
                if not annotator_scores:
                    logging.warning(f"Line {line_num}: No annotator scores found for {problem_id}, {submission_id}")
                    continue
                
                # Extract scores by dimension
                dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
                aggregated_scores = {}
                
                for dim in dimensions:
                    dim_scores = []
                    for annotator in annotator_scores:
                        if 'quality_scores' in annotator and dim in annotator['quality_scores']:
                            score = annotator['quality_scores'][dim]
                            if score is not None:
                                dim_scores.append(float(score))
                    
                    if dim_scores:
                        if aggregation_method == 'mean':
                            aggregated_scores[dim] = statistics.mean(dim_scores)
                        elif aggregation_method == 'median':
                            aggregated_scores[dim] = statistics.median(dim_scores)
                        else:
                            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
                    else:
                        logging.warning(f"Line {line_num}: No valid scores for dimension '{dim}' in {problem_id}, {submission_id}")
                        aggregated_scores[dim] = None
                
                # Only create QualityScores if we have all dimensions
                if all(score is not None for score in aggregated_scores.values()):
                    quality_scores = QualityScores(
                        functionality=aggregated_scores['functionality'],
                        readability=aggregated_scores['readability'], 
                        idiomatic=aggregated_scores['idiomatic'],
                        error_handling=aggregated_scores['error_handling'],
                        efficiency=aggregated_scores['efficiency']
                    )
                    
                    key = (problem_id, submission_id)
                    scores_dict[key] = quality_scores
                else:
                    logging.warning(f"Line {line_num}: Incomplete scores for {problem_id}, {submission_id}")
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(f"Line {line_num}: Skipping invalid line: {e}")
                continue
    
    logging.info(f"Loaded {len(scores_dict)} human assessment scores using {aggregation_method} aggregation")
    return scores_dict


def update_test_set_with_human_scores(source: str, human_scores: Dict[Tuple[str, str], QualityScores]) -> Dict[str, Any]:
    """
    Update the test.jsonl file with human assessment scores.
    Removes samples without human annotations to ensure 100% coverage.
    
    Args:
        source: Source dataset name (e.g., 'codenet', 'codeeval')
        human_scores: Dictionary mapping (problem_id, submission_id) to QualityScores
    
    Returns:
        Dictionary with update statistics
    """
    # Path to integrated data
    integrated_path = Path("generated") / "integrated" / source
    test_file = integrated_path / "test.jsonl"
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    logging.info(f"Updating test set with human scores: {test_file}")
    
    # Load existing test samples
    original_samples = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                sample_dict = json.loads(line)
                sample = CodeSample.from_dict(sample_dict)
                original_samples.append(sample)
    
    logging.info(f"Loaded {len(original_samples)} samples from test set")
    
    # Separate samples into annotated and unannotated
    annotated_samples = []
    removed_samples = []
    
    for sample in original_samples:
        key = (sample.problem_id, sample.submission_id)
        
        if key in human_scores:
            # Update with human scores
            sample.quality_scores = human_scores[key]
            # Update metadata to indicate human assessment
            if sample.metadata is None:
                sample.metadata = {}
            sample.metadata['assessment_method'] = 'human_annotation'
            sample.metadata['human_annotated'] = True
            annotated_samples.append(sample)
        else:
            # Track removed samples for metadata
            removed_samples.append({
                'problem_id': sample.problem_id,
                'submission_id': sample.submission_id,
                'reason': 'no_human_annotation'
            })
    
    logging.info(f"Samples with human scores: {len(annotated_samples)}")
    logging.info(f"Samples removed (no annotation): {len(removed_samples)}")
    
    if removed_samples:
        logging.warning(f"Removing {len(removed_samples)} samples without human annotations:")
        for removed in removed_samples:
            logging.warning(f"  - {removed['problem_id']}, {removed['submission_id']}")
    
    # Only keep annotated samples (100% human coverage)
    final_samples = annotated_samples
    
    # Save updated test set
    backup_file = test_file.with_suffix('.backup.jsonl')
    logging.info(f"Creating backup: {backup_file}")
    
    # Create backup
    import shutil
    shutil.copy2(test_file, backup_file)
    
    # Write updated test set (only annotated samples)
    with open(test_file, 'w') as f:
        for sample in final_samples:
            f.write(sample.to_jsonl_line() + '\n')
    
    # Update metadata file
    metadata_file = integrated_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update split counts (test set size changed)
        original_test_count = metadata['splits']['test']
        metadata['splits']['test'] = len(final_samples)
        metadata['total_samples'] = metadata['splits']['train'] + metadata['splits']['valid'] + len(final_samples)
        
        # Add human assessment info
        metadata['human_assessment'] = {
            'enabled': True,
            'test_set_updated': True,
            'samples_with_human_scores': len(annotated_samples),
            'samples_removed_no_annotation': len(removed_samples),
            'coverage_percentage': 100.0,  # Always 100% after cleanup
            'original_test_size': original_test_count,
            'final_test_size': len(final_samples)
        }
        
        # Add details about removed samples
        if removed_samples:
            metadata['human_assessment']['removed_samples'] = removed_samples
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    results = {
        'source': source,
        'test_file': str(test_file),
        'backup_file': str(backup_file),
        'original_test_samples': len(original_samples),
        'final_test_samples': len(final_samples),
        'updated_samples': len(annotated_samples),
        'removed_samples': len(removed_samples),
        'coverage_percentage': 100.0,  # Always 100% after cleanup
        'human_scores_available': len(human_scores),
        'removed_sample_details': removed_samples
    }
    
    logging.info(f"Human score update completed:")
    logging.info(f"  Original test samples: {results['original_test_samples']}")
    logging.info(f"  Final test samples: {results['final_test_samples']}")
    logging.info(f"  Updated with human scores: {results['updated_samples']}")
    logging.info(f"  Removed (no annotation): {results['removed_samples']}")
    logging.info(f"  Coverage: {results['coverage_percentage']:.1f}%")
    logging.info(f"  Backup created: {backup_file}")
    
    return results


def add_human_scores_to_dataset(source: str, input_path: str, aggregation_method: str = 'mean') -> Dict[str, Any]:
    """
    Main function to add human scores to integrated dataset.
    
    Args:
        source: Source dataset name
        input_path: Path to human scores JSONL file
        aggregation_method: How to aggregate multiple annotator scores ('mean' or 'median')
        
    Returns:
        Dictionary with operation results
    """
    logging.info(f"Adding human scores to {source} dataset using {aggregation_method} aggregation")
    
    # Load human scores
    human_scores = load_human_scores(input_path, aggregation_method)
    
    if not human_scores:
        raise ValueError("No valid human scores found in input file")
    
    # Update test set
    results = update_test_set_with_human_scores(source, human_scores)
    results['aggregation_method'] = aggregation_method
    
    return results


def validate_human_assessment_setup(source: str, input_path: str) -> bool:
    """Validate that human assessment can be performed."""
    # Check if integrated data exists
    integrated_path = Path("generated") / "integrated" / source
    if not integrated_path.exists():
        logging.error(f"Integrated data not found for source: {source}")
        logging.error(f"Run integration first: python main.py integrate --source {source}")
        return False
    
    test_file = integrated_path / "test.jsonl"
    if not test_file.exists():
        logging.error(f"Test file not found: {test_file}")
        return False
    
    # Check input JSONL
    if not Path(input_path).exists():
        logging.error(f"Human scores JSONL not found: {input_path}")
        return False
    
    logging.info("Human assessment setup validation passed")
    return True
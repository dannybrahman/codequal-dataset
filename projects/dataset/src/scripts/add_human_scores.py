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
from pathlib import Path
from typing import Dict, List, Any, Tuple

from ..processors.unified_schema import CodeSample, QualityScores


def parse_human_score_ids(human_problem_id: str, human_submission_id: str) -> Tuple[str, str]:
    """
    Convert human score IDs to test set IDs.

    Human scores use format:
    - problem_id: "codeeval", "mbpp", "codesearchnet", "humaneval"
    - submission_id: "{problem_info}_{submission_info}"

    Test sets use format:
    - problem_id: "{source}_{problem_info}"
    - submission_id: "{source}_sub_{number}" or "{source}_{variant}_{number}"

    Examples:
        ("codeeval", "composition_composition_6_codeeval_sub_00416")
        -> ("codeeval_composition_composition_6", "codeeval_sub_00416")

        ("mbpp", "889_mbpp_sub_00888")
        -> ("mbpp_889", "mbpp_sub_00888")

        ("humaneval", "x_147_humaneval_x_python_147")
        -> ("humaneval_x_147", "humaneval_x_python_147")
    """
    parts = human_submission_id.split('_')

    try:
        # Special handling for humaneval (format: x_147_humaneval_x_python_147)
        if human_problem_id == 'humaneval':
            # Extract problem number from "x_{num}_humaneval_x_..."
            if len(parts) >= 4 and parts[0] == 'x':
                problem_num = parts[1]
                # Submission ID is everything from "humaneval" onwards
                humaneval_idx = parts.index('humaneval')
                test_submission_id = '_'.join(parts[humaneval_idx:])
                test_problem_id = f"humaneval_x_{problem_num}"
                return (test_problem_id, test_submission_id)

        # Standard handling for sources with "sub" pattern
        if 'sub' in parts:
            sub_idx = parts.index('sub')
            # Submission ID is source_sub_number
            test_submission_id = f"{human_problem_id}_sub_{parts[sub_idx + 1]}"

            # Problem ID is everything before the submission part
            problem_info = '_'.join(parts[:sub_idx-1])  # Skip the source name before 'sub'
            test_problem_id = f"{human_problem_id}_{problem_info}"

            return (test_problem_id, test_submission_id)

        # Fallback: return as-is
        logging.warning(f"Unrecognized ID format: {human_problem_id}, {human_submission_id}")
        return (human_problem_id, human_submission_id)

    except (ValueError, IndexError) as e:
        logging.warning(f"Failed to parse IDs: {human_problem_id}, {human_submission_id}: {e}")
        return (human_problem_id, human_submission_id)


def load_human_scores(input_file: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Load human assessment scores from JSONL file with individual annotator scores.

    Args:
        input_file: Path to JSONL with format:
                   {"problem_id": "...", "submission_id": "...", "scores": [{"annotator_id": "...", "quality_scores": {...}}]}

    Returns:
        Dictionary mapping (problem_id, submission_id) to list of annotator scores
        Format: [{'annotator_id': str, 'scores': {...}}]
    """
    scores_dict = {}
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Human scores file not found: {input_file}")

    logging.info(f"Loading human assessment scores from {input_path}")
    
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

                # Convert human score IDs to test set IDs
                test_problem_id, test_submission_id = parse_human_score_ids(problem_id, submission_id)
                key = (test_problem_id, test_submission_id)

                # Store individual annotator scores (no aggregation)
                annotator_score_list = []

                for annotator in annotator_scores:
                    if 'annotator_id' not in annotator or 'quality_scores' not in annotator:
                        logging.warning(f"Line {line_num}: Missing annotator_id or quality_scores")
                        continue

                    qs = annotator['quality_scores']

                    # Validate all dimensions are present
                    dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
                    if not all(dim in qs and qs[dim] is not None for dim in dimensions):
                        logging.warning(f"Line {line_num}: Incomplete scores for annotator {annotator['annotator_id']}")
                        continue

                    # Create annotator score entry
                    annotator_entry = {
                        'annotator_id': annotator['annotator_id'],
                        'scores': {
                            'functionality': float(qs['functionality']),
                            'readability': float(qs['readability']),
                            'idiomatic': float(qs['idiomatic']),
                            'error_handling': float(qs['error_handling']),
                            'efficiency': float(qs['efficiency'])
                        }
                    }
                    annotator_score_list.append(annotator_entry)

                if annotator_score_list:
                    scores_dict[key] = annotator_score_list
                else:
                    logging.warning(f"Line {line_num}: No valid annotator scores for {problem_id}, {submission_id}")
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(f"Line {line_num}: Skipping invalid line: {e}")
                continue
    
    total_annotators = sum(len(annotators) for annotators in scores_dict.values())
    logging.info(f"Loaded {len(scores_dict)} samples with {total_annotators} total annotator scores")
    return scores_dict


def update_test_set_with_human_scores(source: str, human_scores: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Update the test.jsonl file with human assessment scores.
    Removes samples without human annotations to ensure 100% coverage.

    Args:
        source: Source dataset name (e.g., 'codenet', 'codeeval')
        human_scores: Dictionary mapping (problem_id, submission_id) to list of annotator scores

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
            # Initialize human_scores list if not present
            if not hasattr(sample, 'human_scores') or sample.human_scores is None:
                sample.human_scores = []

            # Get list of annotator scores for this sample
            annotator_scores = human_scores[key]

            # Process each annotator's scores
            for annotator_entry in annotator_scores:
                # Remove existing scores for this annotator (overwrite)
                sample.human_scores = [
                    score for score in sample.human_scores
                    if score.get('annotator_id') != annotator_entry['annotator_id']
                ]

                # Add new scores
                sample.human_scores.append(annotator_entry)

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
        # Handle both metadata formats: 'test' or 'test_size'
        if 'test' in metadata['splits']:
            original_test_count = metadata['splits']['test']
            metadata['splits']['test'] = len(final_samples)
            train_count = metadata['splits'].get('train', 0)
            valid_count = metadata['splits'].get('valid', 0)
        elif 'test_size' in metadata['splits']:
            original_test_count = metadata['splits']['test_size']
            metadata['splits']['test_size'] = len(final_samples)
            train_count = metadata['splits'].get('train_size', 0)
            valid_count = metadata['splits'].get('valid_size', 0)
        else:
            logging.warning(f"Unknown metadata format, skipping total_samples update")
            original_test_count = len(original_samples)
            train_count = 0
            valid_count = 0

        if train_count or valid_count:
            metadata['total_samples'] = train_count + valid_count + len(final_samples)
        
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


def add_human_scores_to_dataset(source: str, input_path: str) -> Dict[str, Any]:
    """
    Main function to add human scores to integrated dataset.

    Args:
        source: Source dataset name
        input_path: Path to human scores JSONL file

    Returns:
        Dictionary with operation results
    """
    logging.info(f"Adding human scores to {source} dataset")

    # Load human scores (individual annotator scores, no aggregation)
    human_scores = load_human_scores(input_path)

    if not human_scores:
        raise ValueError("No valid human scores found in input file")

    # Update test set
    results = update_test_set_with_human_scores(source, human_scores)

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
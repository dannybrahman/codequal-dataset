#!/usr/bin/env python3
"""
Convert Qualtrics human assessment CSV to CodeQual unified schema format.

This script reads a Qualtrics survey export CSV containing human quality assessments
and converts it to the CodeQual unified schema with 5-dimensional quality scores.

Usage:
    python convert_qualtrics_to_codequal.py --input codenet_test_set_human_assessment.csv --samples sample_human_main_v1.csv --output human_scores_detailed.jsonl
"""

import csv
import json
import re
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HumanAssessment:
    """Human assessment scores for a single sample."""
    problem_id: str
    submission_id: str
    functionality: float
    readability: float
    idiomatic: float  # pythonic in survey
    error_handling: float
    efficiency: float
    num_annotators: int
    raw_scores: List[List[float]]  # [annotator][dimension]


class QualtricsConverter:
    """Converts Qualtrics survey data to CodeQual unified schema."""
    
    def __init__(self, qualtrics_file: str, survey_config: dict):
        """
        Initialize converter.
        
        Args:
            qualtrics_file: Path to Qualtrics CSV export
            survey_config: Survey configuration dict with question mapping
        """
        self.qualtrics_file = Path(qualtrics_file)
        self.survey_config = survey_config
        self.question_to_sample_map: Dict[int, Tuple[str, str]] = {}
        self.human_assessments: List[HumanAssessment] = []
        
    def load_sample_mapping(self):
        """Load mapping from question numbers to (problem_id, submission_id)."""
        print("Loading sample mapping from survey configuration")
        questions = self.survey_config['survey_structure']['questions']
        for i, question in enumerate(questions, 1):  # Q1, Q2, Q3, ...
            # Extract problem_id and submission_id from sample_id
            sample_id = question['sample_id']
            if '_' in sample_id:
                problem_id, submission_id = sample_id.split('_', 1)
                self.question_to_sample_map[i] = (problem_id, submission_id)
            else:
                print(f"Warning: Invalid sample_id format: {sample_id}")
        
        print(f"Loaded {len(self.question_to_sample_map)} sample mappings from survey config")
    
    def parse_score(self, score_str: str) -> Optional[float]:
        """Parse score from various Qualtrics formats."""
        if not score_str or score_str.strip() == '':
            return None
        
        # Extract numeric part (handles "1 (Poor)", "5 (Excellent)", etc.)
        match = re.match(r'(\d+)', score_str.strip())
        if match:
            score = int(match.group(1))
            # Validate range (1-5)
            if 1 <= score <= 5:
                return float(score)
        
        return None
    
    def load_qualtrics_data(self) -> List[HumanAssessment]:
        """Load and process Qualtrics CSV data."""
        print(f"Loading Qualtrics data from {self.qualtrics_file}")
        
        with open(self.qualtrics_file, 'r') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
        
        if len(all_rows) < 6:
            raise ValueError("Expected at least 6 rows in Qualtrics CSV (header + metadata + scores)")
        
        header = all_rows[0]
        
        # Find score rows (skip header and metadata rows)
        # Based on analysis: rows 5-8 contain actual scores
        score_rows = []
        for i in range(5, len(all_rows)):
            row = all_rows[i]
            # Check if this row has actual scores (not metadata)
            if self._has_real_scores(header, row):
                score_rows.append(row)
        
        print(f"Found {len(score_rows)} annotator rows with scores")
        
        # Process each question
        assessments = []
        for q_num in range(1, 226):  # Q1 to Q225
            if q_num not in self.question_to_sample_map:
                print(f"Warning: No sample mapping for Q{q_num}")
                continue
            
            problem_id, submission_id = self.question_to_sample_map[q_num]
            
            # Collect scores from all annotators for this question
            annotator_scores = []
            for row in score_rows:
                scores_for_annotator = []
                for dim in range(1, 6):  # 5 dimensions
                    col_name = f'Q{q_num}_{dim}'
                    if col_name in header:
                        col_idx = header.index(col_name)
                        raw_score = row[col_idx] if col_idx < len(row) else ''
                        parsed_score = self.parse_score(raw_score)
                        scores_for_annotator.append(parsed_score)
                    else:
                        scores_for_annotator.append(None)
                
                # Only include if annotator provided at least some scores
                if any(s is not None for s in scores_for_annotator):
                    annotator_scores.append(scores_for_annotator)
            
            if not annotator_scores:
                print(f"Warning: No scores found for Q{q_num} ({problem_id}, {submission_id})")
                continue
            
            # Aggregate scores across annotators (using mean)
            aggregated_scores = []
            for dim in range(5):  # 5 dimensions
                dim_scores = [scores[dim] for scores in annotator_scores if scores[dim] is not None]
                if dim_scores:
                    aggregated_scores.append(statistics.mean(dim_scores))
                else:
                    aggregated_scores.append(None)
            
            # Only create assessment if we have scores for all dimensions
            if all(s is not None for s in aggregated_scores):
                assessment = HumanAssessment(
                    problem_id=problem_id,
                    submission_id=submission_id,
                    functionality=round(aggregated_scores[0], 2),
                    readability=round(aggregated_scores[1], 2),
                    idiomatic=round(aggregated_scores[2], 2),  # pythonic -> idiomatic
                    error_handling=round(aggregated_scores[3], 2),
                    efficiency=round(aggregated_scores[4], 2),
                    num_annotators=len(annotator_scores),
                    raw_scores=annotator_scores
                )
                assessments.append(assessment)
            else:
                print(f"Warning: Incomplete scores for Q{q_num} ({problem_id}, {submission_id})")
        
        print(f"Processed {len(assessments)} complete assessments")
        return assessments
    
    def _has_real_scores(self, header: List[str], row: List[str]) -> bool:
        """Check if a row contains actual numeric scores (not metadata)."""
        # Check Q1_1 as a sample
        if 'Q1_1' in header:
            col_idx = header.index('Q1_1')
            if col_idx < len(row):
                score = row[col_idx].strip()
                # Real scores are numeric or contain numbers, not JSON metadata
                return bool(score and not score.startswith('{') and re.search(r'\d', score))
        return False
    
    
    def save_detailed_jsonl(self, output_file: str):
        """Save detailed JSONL with individual annotator scores."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving detailed assessment data to {output_path}")
        
        with open(output_path, 'w') as f:
            for assessment in self.human_assessments:
                # Create scores array with individual annotator data
                scores_array = []
                
                for ann_idx, annotator_scores in enumerate(assessment.raw_scores):
                    if all(s is not None for s in annotator_scores):  # Valid scores
                        annotator_data = {
                            "annotator_id": f"annotator_{ann_idx + 1}",
                            "quality_scores": {
                                "functionality": annotator_scores[0],
                                "readability": annotator_scores[1], 
                                "idiomatic": annotator_scores[2],
                                "error_handling": annotator_scores[3],
                                "efficiency": annotator_scores[4]
                            }
                        }
                        scores_array.append(annotator_data)
                
                # Create sample record
                sample_record = {
                    "problem_id": assessment.problem_id,
                    "submission_id": assessment.submission_id,
                    "scores": scores_array
                }
                
                # Write as JSONL
                f.write(json.dumps(sample_record) + '\n')
        
        print(f"Saved detailed assessment data to {output_path}")
    
    def print_statistics(self):
        """Print summary statistics of the assessments."""
        if not self.human_assessments:
            print("No assessments to analyze")
            return
        
        print(f"\n=== Human Assessment Statistics ===")
        print(f"Total assessments: {len(self.human_assessments)}")
        
        # Score statistics by dimension
        dimensions = ['functionality', 'readability', 'idiomatic', 'error_handling', 'efficiency']
        
        for dim in dimensions:
            scores = [getattr(a, dim) for a in self.human_assessments]
            print(f"{dim.title()}:")
            print(f"  Mean: {statistics.mean(scores):.2f}")
            print(f"  Std:  {statistics.stdev(scores):.2f}")
            print(f"  Range: {min(scores):.1f} - {max(scores):.1f}")
        
        # Annotator coverage
        annotator_counts = [a.num_annotators for a in self.human_assessments]
        print(f"\nAnnotator coverage:")
        print(f"  Average annotators per sample: {statistics.mean(annotator_counts):.1f}")
        print(f"  Min-Max annotators: {min(annotator_counts)} - {max(annotator_counts)}")
    
    def convert(self):
        """Run the full conversion process."""
        print("=== Qualtrics to CodeQual Converter ===")
        
        # Load sample mapping
        self.load_sample_mapping()
        
        # Load and process Qualtrics data
        self.human_assessments = self.load_qualtrics_data()
        
        # Print statistics
        self.print_statistics()
        
        return self.human_assessments


def main():
    parser = argparse.ArgumentParser(description='Convert Qualtrics CSV to CodeQual format (standalone mode)')
    parser.add_argument('--input', required=True, 
                      help='Path to Qualtrics CSV file')
    parser.add_argument('--survey-config', required=True,
                      help='Path to survey configuration JSON file')
    parser.add_argument('--output', default='human_assessments_detailed.jsonl',
                      help='Output JSONL file path (individual annotator scores)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not Path(args.survey_config).exists():
        print(f"Error: Survey config file not found: {args.survey_config}")
        return 1
    
    # Load survey configuration
    with open(args.survey_config, 'r') as f:
        survey_config = json.load(f)
    
    # Run conversion
    converter = QualtricsConverter(args.input, survey_config)
    assessments = converter.convert()
    
    if not assessments:
        print("Error: No assessments were successfully converted")
        return 1
    
    # Save results
    converter.save_detailed_jsonl(args.output)
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📊 {len(assessments)} human assessments converted")
    print(f"📁 Detailed JSONL: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
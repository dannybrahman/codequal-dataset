#!/usr/bin/env python3
"""
CodeQual Dataset - Main CLI Entry Point

Enhanced CodeQual dataset creation toolkit with 5-dimensional continuous quality scoring.
Handles dataset integration, analysis, and human annotation integration.

Examples:
  # Dataset Integration
  python main.py integrate --source codenet --input-path /path/to/codenet/data
  python main.py integrate --source codeeval --input-path /path/to/codeeval/data
  python main.py integrate --source codesearchnet --input-path /path/to/codesearchnet/data
  
  # Dataset Analysis (for sampling strategy design)
  python main.py analyze --source codesearchnet
  
  # Human Annotation Integration (adds human scores to datasets)
  python main.py add-human-scores --source codenet --input-path human_scores_detailed.jsonl --method median
  python main.py add-human-scores --source codeeval --input-path codeeval_human_scores.jsonl --method mean
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.scripts import (
    run_integration, validate_source_data, get_available_sources
)
from src.scripts.add_human_scores import add_human_scores_to_dataset, validate_human_assessment_setup


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CodeQual Dataset - 5D Quality Assessment Toolkit"
    )
    
    # Global arguments
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    subparsers = parser.add_subparsers(dest='workflow', help='Available workflows')
    
    # Data integration workflow
    integrate_parser = subparsers.add_parser(
        'integrate',
        help='Integrate source datasets into unified format'
    )
    integrate_parser.add_argument('--source', required=True,
                                 help='Source dataset name (e.g., codeeval, humaneval-x, mbpp, codecontests)')
    integrate_parser.add_argument('--input-path', required=True,
                                 help='Path to source dataset directory')
    integrate_parser.add_argument('--test-samples',
                                 help='Path to file with test sample IDs (format: problem_id,submission_id per line)')
    integrate_parser.add_argument('--train-ratio', type=float, default=0.8,
                                 help='Training set ratio (default: 0.8)')
    integrate_parser.add_argument('--valid-ratio', type=float, default=0.1,
                                 help='Validation set ratio (default: 0.1)')
    
    # Filtering options for CodeSearchNet
    integrate_parser.add_argument('--min-description-length', type=int,
                                 help='Minimum description length filter (CodeSearchNet only)')
    integrate_parser.add_argument('--max-description-length', type=int,
                                 help='Maximum description length filter (CodeSearchNet only)')
    integrate_parser.add_argument('--min-lines-of-code', type=int,
                                 help='Minimum lines of code filter (CodeSearchNet only)')
    integrate_parser.add_argument('--max-lines-of-code', type=int,
                                 help='Maximum lines of code filter (CodeSearchNet only)')
    integrate_parser.add_argument('--random-sample', type=int,
                                 help='Randomly sample N items from each language in the dataset (CodeSearchNet only)')
    integrate_parser.add_argument('--skip-flagged', action='store_true',
                                 help='Skip samples flagged in dataset-viewer (CodeSearchNet only)')
    integrate_parser.add_argument('--languages', nargs='+',
                                 help='Languages to include (CodeSearchNet/HumanEval-X only, e.g., --languages python javascript)')
    
    integrate_parser.set_defaults(workflow='integrate')
    
    # Human annotation integration workflow
    human_parser = subparsers.add_parser(
        'add-human-scores',
        help='Add human annotation scores to integrated datasets'
    )
    human_parser.add_argument('--source', required=True,
                             help='Source dataset name (must match integrated data)')
    human_parser.add_argument('--input-path', required=True,
                             help='Path to JSONL file with human scores')
    human_parser.add_argument('--method', default='mean', choices=['mean', 'median'],
                             help='Aggregation method for multiple annotator scores (default: mean)')
    human_parser.add_argument('--split', default='all',
                             choices=['train', 'valid', 'test', 'all'],
                             help='Which split to process (default: all)')
    human_parser.set_defaults(workflow='add-human-scores')
    
    # Dataset analysis workflow
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze dataset characteristics for sampling strategy design'
    )
    analyze_parser.add_argument('--source', required=True,
                               help='Source dataset name (e.g., codesearchnet)')
    analyze_parser.add_argument('--dataset-path',
                               help='Path to converted dataset (default: generated/integrated/{source})')
    analyze_parser.add_argument('--output-dir',
                               help='Output directory for analysis results (default: generated/analysis/{source})')
    analyze_parser.set_defaults(workflow='analyze')
    
    args = parser.parse_args()
    
    if not args.workflow:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Route to appropriate workflow
        if args.workflow == 'integrate':
            return run_integration_workflow(args)
        elif args.workflow == 'add-human-scores':
            return add_human_scores_workflow(args)
        elif args.workflow == 'analyze':
            return analyze_dataset_workflow(args)
        else:
            logger.error(f"Unknown workflow: {args.workflow}")
            return 1
            
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        return 1


def run_integration_workflow(args):
    """Run data integration workflow."""
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not validate_source_data(args.source, args.input_path):
        logger.error("Source data validation failed")
        return 1
    
    available_sources = get_available_sources()
    if args.source not in available_sources:
        logger.error(f"Unknown source: {args.source}")
        logger.error(f"Available sources: {available_sources}")
        return 1
    
    # Prepare filtering options for CodeSearchNet and HumanEval-X
    filtering_options = {}
    if args.source == 'codesearchnet':
        if args.min_description_length is not None:
            filtering_options['min_description_length'] = args.min_description_length
        if args.max_description_length is not None:
            filtering_options['max_description_length'] = args.max_description_length
        if args.min_lines_of_code is not None:
            filtering_options['min_lines_of_code'] = args.min_lines_of_code
        if args.max_lines_of_code is not None:
            filtering_options['max_lines_of_code'] = args.max_lines_of_code
        if args.random_sample is not None:
            filtering_options['random_sample'] = args.random_sample
        if args.skip_flagged:
            filtering_options['skip_flagged'] = True
        if args.languages:
            filtering_options['languages'] = args.languages
    elif args.source == 'humaneval-x':
        if args.languages:
            filtering_options['languages'] = args.languages
    
    # Run integration
    logger.info(f"Starting integration workflow: {args.source}")
    results = run_integration(
        source=args.source,
        input_path=args.input_path,
        test_samples_file=args.test_samples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        **filtering_options
    )
    
    logger.info("Integration workflow completed successfully!")
    logger.info(f"Output directory: {results['output_path']}")
    logger.info(f"Samples converted: {results['samples_converted']}")
    
    return 0


def add_human_scores_workflow(args):
    """Run human assessment workflow - add human scores to test set."""
    logger = logging.getLogger(__name__)
    
    # Validate required arguments
    if not args.input_path:
        logger.error("--input-path is required for human assessment")
        return 1
    
    # Validate setup
    if not validate_human_assessment_setup(args.source, args.input_path):
        return 1
    
    # Run human assessment
    logger.info(f"Starting human assessment workflow for {args.source}")
    logger.info(f"Loading human scores from: {args.input_path}")
    logger.info(f"Using aggregation method: {args.method}")
    
    try:
        results = add_human_scores_to_dataset(args.source, args.input_path, args.method)
        
        logger.info("Human assessment workflow completed!")
        logger.info(f"Aggregation method: {results['aggregation_method']}")
        logger.info(f"Updated test set: {results['test_file']}")
        logger.info(f"Original test samples: {results['original_test_samples']}")
        logger.info(f"Final test samples: {results['final_test_samples']}")
        logger.info(f"Updated with human scores: {results['updated_samples']}")
        logger.info(f"Coverage: {results['coverage_percentage']:.1f}%")
        logger.info(f"Backup created: {results['backup_file']}")
        
        if results['removed_samples'] > 0:
            logger.warning(f"Removed {results['removed_samples']} samples without human annotations")
            logger.info("Test set now has 100% human annotation coverage")
        
        return 0
        
    except Exception as e:
        logger.error(f"Human assessment workflow failed: {e}")
        return 1



def analyze_dataset_workflow(args):
    """Run dataset analysis workflow."""
    logger = logging.getLogger(__name__)
    
    # Set default paths
    if not args.dataset_path:
        args.dataset_path = f"generated/integrated/{args.source}"
    
    if not args.output_dir:
        args.output_dir = f"generated/analysis/{args.source}"
    
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    
    # Validate dataset path
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        logger.error(f"Make sure to run integration first: python main.py integrate --source {args.source}")
        return 1
    
    # Import and create analyzer based on source
    if args.source == 'codesearchnet':
        from src.analyzers import CodeSearchNetAnalyzer
        analyzer = CodeSearchNetAnalyzer(dataset_path)
    else:
        logger.error(f"Analysis not yet implemented for source: {args.source}")
        logger.error("Currently supported: codesearchnet")
        return 1
    
    try:
        # Run analysis
        logger.info(f"Starting analysis of {args.source} dataset")
        results = analyzer.analyze(output_dir)
        
        logger.info("Dataset analysis completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("Files generated:")
        logger.info(f"  - {output_dir}/analysis_summary.md (comprehensive report)")
        logger.info(f"  - {output_dir}/full_analysis.json (raw data)")
        logger.info(f"  - {output_dir}/*.png (visualizations)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
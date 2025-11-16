#!/usr/bin/env python3
"""
CodeQual Dataset - Main CLI Entry Point

Enhanced CodeQual dataset creation toolkit with 5-dimensional continuous quality scoring.
Handles dataset integration, analysis, human annotation integration, and LLM assessment integration.

Examples:
  # Dataset Integration
  python main.py integrate --source codenet --input-path /path/to/codenet/data
  python main.py integrate --source codeeval --input-path /path/to/codeeval/data
  python main.py integrate --source codesearchnet --input-path /path/to/codesearchnet/data

  # Dataset Analysis (for sampling strategy design)
  python main.py analyze --source codesearchnet

  # Human Annotation Integration (stores individual annotator scores)
  python main.py add-human-scores --source codenet --input-path human_scores.jsonl
  python main.py add-human-scores --source codeeval --source mbpp --input-path multi_source_scores.jsonl

  # LLM Assessment Integration (stores individual model scores)
  python main.py add-llm-scores --input-path projects/llm_benchmarks/generated/session_XXX --split test
  python main.py add-llm-scores --input-path session_dir --split test --sources codeeval mbpp
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
from src.scripts.add_llm_scores import add_llm_scores_to_datasets, validate_llm_scores_setup


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
    human_parser.add_argument('--source', action='append', dest='sources',
                             help='Source dataset name (can specify multiple times)')
    human_parser.add_argument('--input-path', required=True,
                             help='Path to JSONL file with human scores')
    human_parser.add_argument('--split', default='all',
                             choices=['train', 'valid', 'test', 'all'],
                             help='Which split to process (default: all)')
    human_parser.set_defaults(workflow='add-human-scores')

    # LLM scores integration workflow
    llm_parser = subparsers.add_parser(
        'add-llm-scores',
        help='Add LLM assessment scores to integrated datasets'
    )
    llm_parser.add_argument('--input-path', required=True,
                           help='Path to LLM scores directory (e.g., projects/llm_benchmarks/generated/session_XXX)')
    llm_parser.add_argument('--split', required=True,
                           choices=['train', 'valid', 'test'],
                           help='Which dataset split to update')
    llm_parser.add_argument('--sources', nargs='+',
                           help='Specific sources to process (if not specified, processes all found)')
    llm_parser.set_defaults(workflow='add-llm-scores')

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
        elif args.workflow == 'add-llm-scores':
            return add_llm_scores_workflow(args)
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

    if not args.sources:
        logger.error("--source is required (can specify multiple times)")
        return 1

    # Process each source
    all_results = []
    failed_sources = []

    for source in args.sources:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing source: {source}")
        logger.info(f"{'='*60}")

        # Validate setup
        if not validate_human_assessment_setup(source, args.input_path):
            logger.error(f"Validation failed for source: {source}")
            failed_sources.append(source)
            continue

        # Run human assessment
        logger.info(f"Starting human assessment workflow for {source}")
        logger.info(f"Loading human scores from: {args.input_path}")

        try:
            results = add_human_scores_to_dataset(source, args.input_path)
            all_results.append(results)

            logger.info(f"Human assessment workflow completed for {source}!")
            logger.info(f"Updated test set: {results['test_file']}")
            logger.info(f"Original test samples: {results['original_test_samples']}")
            logger.info(f"Final test samples: {results['final_test_samples']}")
            logger.info(f"Updated with human scores: {results['updated_samples']}")
            logger.info(f"Coverage: {results['coverage_percentage']:.1f}%")
            logger.info(f"Backup created: {results['backup_file']}")

            if results['samples_without_scores'] > 0:
                logger.info(f"{results['samples_without_scores']} samples without human scores (kept in test set)")
                logger.info(f"Coverage: {results['coverage_percentage']:.1f}%")

        except Exception as e:
            logger.error(f"Human assessment workflow failed for {source}: {e}")
            failed_sources.append(source)
            continue

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: Processed {len(args.sources)} sources")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {len(all_results)}")
    if failed_sources:
        logger.error(f"Failed: {len(failed_sources)} - {', '.join(failed_sources)}")

    for result in all_results:
        logger.info(f"\n{result['source']}:")
        logger.info(f"  Updated samples: {result['updated_samples']}")
        logger.info(f"  Samples without scores: {result['samples_without_scores']}")
        logger.info(f"  Final test samples: {result['final_test_samples']}")

    return 1 if failed_sources else 0


def add_llm_scores_workflow(args):
    """Run LLM scores integration workflow - add LLM assessment scores to dataset split."""
    logger = logging.getLogger(__name__)

    # Validate required arguments
    if not args.input_path:
        logger.error("--input-path is required for LLM scores integration")
        return 1

    # Validate setup
    if not validate_llm_scores_setup(args.input_path):
        return 1

    logger.info(f"Starting LLM scores integration workflow")
    logger.info(f"Input directory: {args.input_path}")
    logger.info(f"Target split: {args.split}")
    if args.sources:
        logger.info(f"Processing sources: {', '.join(args.sources)}")

    try:
        results = add_llm_scores_to_datasets(
            input_path=args.input_path,
            split=args.split,
            sources=args.sources
        )

        logger.info("\n" + "="*60)
        logger.info("LLM SCORES INTEGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sources processed: {results['sources_processed']}")
        logger.info(f"Total models integrated: {results['total_models']}")
        logger.info(f"Total samples updated: {results['total_samples_updated']}")

        for source_result in results['source_results']:
            logger.info(f"\n{source_result['source']}:")
            logger.info(f"  Models: {', '.join(source_result['models'])}")
            logger.info(f"  Samples updated: {source_result['samples_updated']}")
            logger.info(f"  Coverage: {source_result['coverage_percentage']:.1f}%")
            logger.info(f"  Updated file: {source_result['updated_file']}")

        return 0

    except Exception as e:
        logger.error(f"LLM scores integration workflow failed: {e}")
        import traceback
        traceback.print_exc()
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
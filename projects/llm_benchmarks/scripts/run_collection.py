#!/usr/bin/env python3
"""
Main script for running LLM solution collection

Usage:
    cd projects/llm_benchmarks
    python scripts/run_collection.py --models gpt-4 claude-3-sonnet-20240229 --categories typing inheritance
    python scripts/run_collection.py --all-models --quick-test
    python scripts/run_collection.py --resume session_1234567890
"""

import sys
import os
import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (override existing env vars)
load_dotenv(override=True)

# Add src to path so we can import our modules  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_benchmarks.config import ModelConfigs, CollectionSettings
from llm_benchmarks.runners.collection_runner import SolutionCollectionRunner

def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

def progress_callback(completed: int, total: int, is_final: bool):
    """Progress callback for displaying collection progress"""
    percent = (completed / max(1, total)) * 100
    status = "COMPLETE" if is_final else "PROGRESS"
    print(f"[{status}] {completed}/{total} tasks completed ({percent:.1f}%)")

async def main():
    parser = argparse.ArgumentParser(description="Run LLM solution collection")
    
    # Model selection
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--all-models", action="store_true", help="Use all available models")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    
    # Dataset options
    parser.add_argument("--data-sources", type=str,
                       help="Specific data sources to evaluate. Comma-separated: codeeval,codenet,codesearchnet,humaneval-x,mbpp")
    parser.add_argument("--sets", default="test",
                       help="Dataset splits to process (default: test). Comma-separated: train,valid,test")
    parser.add_argument("--dataset-path", default="../dataset/generated/integrated",
                       help="Path to integrated dataset directory")
    parser.add_argument("--sample-ids-file", type=str,
                       help="File containing sample IDs to process (one per line, format: source,submission_id)")
    
    # Execution options
    parser.add_argument("--output-dir", default="generated", help="Output directory")
    
    # Session management
    parser.add_argument("--resume", help="Resume from session ID")
    parser.add_argument("--session-name", help="Custom session name")
    
    # Test/debug options
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with small batch")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running collection")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--log-file", help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # List models and exit if requested
    if args.list_models:
        print("Available models:")
        for model_name in ModelConfigs.get_available_models():
            config = ModelConfigs.get_model_config(model_name)
            api_key_available = bool(os.getenv(config.api_key_env))
            status = "✓" if api_key_available else "✗ (no API key)"
            print(f"  {model_name} ({config.provider}) {status}")
        
        print("\nAPI Key Status:")
        key_status = ModelConfigs.validate_api_keys()
        for key, available in key_status.items():
            status = "✓" if available else "✗"
            print(f"  {key}: {status}")
        
        return
    
    # Determine which models to use
    if args.all_models:
        model_names = None  # Use all enabled models
    elif args.models:
        model_names = args.models
    else:
        # Default to a few common models if none specified
        model_names = ["gpt_3_5_turbo", "claude-3-haiku", "command", "gemini_2_0_flash", "grok_2_vision", "Qwen3_4b", "llama3_2_1b", "mistral_12b", "deepseek_r1_distill_llama_8b"]
        logger.info(f"No models specified, using defaults: {model_names}")
    
    # Create settings
    if args.quick_test:
        settings = CollectionSettings.quick_test()
    else:
        settings = CollectionSettings()
    
    # Override settings with command line arguments
    settings.data_sources = getattr(args, 'data_sources', None)
    settings.sets = args.sets
    settings.session_name = args.session_name
    settings.log_level = args.log_level
    settings.log_file = args.log_file

    # Load sample IDs filter if provided
    sample_ids_filter = None
    if args.sample_ids_file:
        sample_ids_filter = {}
        with open(args.sample_ids_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    source, submission_id = parts[0].strip(), parts[1].strip()
                    if source not in sample_ids_filter:
                        sample_ids_filter[source] = set()
                    sample_ids_filter[source].add(submission_id)
        logger.info(f"Loaded sample ID filter: {sum(len(v) for v in sample_ids_filter.values())} samples from {len(sample_ids_filter)} sources")
    
    # Validate settings
    warnings = settings.validate()
    if warnings:
        logger.warning("Settings validation warnings:")
        for warning in warnings:
            logger.warning(f"  {warning}")
    
    try:
        # Create model instances
        logger.info("Initializing models...")
        models = ModelConfigs.create_models_from_config(model_names)
        
        if not models:
            logger.error("No models could be initialized. Check API keys.")
            return 1
        
        logger.info(f"Initialized {len(models)} models: {[m.model_name for m in models]}")
        
        # Get extraction API key
        extraction_key = ModelConfigs.get_extraction_config()
        if not extraction_key:
            logger.warning("No extraction API key found. Using fallback extraction method.")
        
        # Create collection runner
        runner = SolutionCollectionRunner(
            models=models,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            extraction_api_key=extraction_key,
            batch_size=settings.batch_size,
            max_concurrent_per_model=settings.max_concurrent_per_model
        )
        
        # Load problems to check dataset
        logger.info("Loading dataset...")
        num_problems = runner.load_problems(settings.get_data_sources_list(), settings.get_sets_list(), sample_ids_filter)
        logger.info(f"Loaded {num_problems} problems")
        
        if num_problems == 0:
            logger.error("No problems found in dataset")
            return 1
        
        total_tasks = num_problems * len(models)
        logger.info(f"Total tasks to complete: {total_tasks}")
        
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0
        
        # Run collection
        logger.info("Starting solution collection...")
        results = await runner.collect_all_solutions(
            data_sources=settings.get_data_sources_list(),
            sets=settings.get_sets_list(),
            progress_callback=progress_callback,
            resume_session=args.resume,
            quick_test=args.quick_test
        )
        
        # Print summary
        print("\nCollection Summary:")
        print(f"Session ID: {results['session_id']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Problems processed: {results['problems_processed']}")
        print(f"Models evaluated: {len(results['models_evaluated'])}")
        print(f"Success: {results['success']}")
        
        print("\nOutput files:")
        for file_type, path in results['output_files'].items():
            print(f"  {file_type}: {path}")
        
        if results['success']:
            logger.info("Collection completed successfully!")
            return 0
        else:
            logger.error("Collection failed")
            return 1
            
    except Exception as e:
        logger.exception(f"Collection failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
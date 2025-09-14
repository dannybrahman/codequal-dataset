#!/usr/bin/env python3
"""
Script for resuming interrupted LLM solution collections

Usage:
    cd projects/llm_benchmarks
    python scripts/resume_collection.py session_1234567890
    python scripts/resume_collection.py --list-sessions
    python scripts/resume_collection.py session_1234567890 --status

Note: Resume automatically uses the original session parameters (models, data_sources, 
quick_test mode, etc.) from when the session was first created.
"""

import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (override existing env vars)
load_dotenv(override=True)

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_benchmarks.storage.progress_tracker import ProgressTracker
from llm_benchmarks.storage.assessment_store import AssessmentStore
from llm_benchmarks.config import ModelConfigs
from llm_benchmarks.runners.collection_runner import SolutionCollectionRunner

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def progress_callback(completed: int, total: int, is_final: bool):
    """Progress callback for displaying collection progress"""
    percent = (completed / max(1, total)) * 100
    status = "COMPLETE" if is_final else "PROGRESS"
    print(f"[{status}] {completed}/{total} tasks completed ({percent:.1f}%)")

def list_sessions(output_dir: str):
    """List available sessions"""
    tracker = ProgressTracker(output_dir)
    sessions = tracker.get_available_sessions()
    
    if not sessions:
        print("No sessions found")
        return
    
    print(f"Available sessions in {output_dir}:")
    
    for session_id in sessions:
        session_dir = Path(output_dir) / session_id
        progress_file = session_dir / "progress.json"
        
        # Try to get basic info
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                session_status = data.get("status", "unknown").upper()
                progress = data.get("current_progress", {})
                completed = progress.get("completed_tasks", 0)
                total = progress.get("total_tasks", 0)
                
            percent = (completed / max(1, total)) * 100
            resumable = "(RESUMABLE)" if session_status in ["FAILED", "IN_PROGRESS"] else ""
            print(f"  {session_id}: {session_status} ({completed}/{total} = {percent:.1f}%) {resumable}")
            
        except Exception:
            print(f"  {session_id}: UNKNOWN (unable to read progress)")

def show_session_status(session_id: str, output_dir: str):
    """Show detailed status for a specific session"""
    tracker = ProgressTracker(output_dir)
    
    if not tracker.load_previous_session(session_id):
        print(f"Session {session_id} not found")
        return
    
    report = tracker.get_progress_report()
    
    print(f"Session Status: {session_id}")
    print("=" * 50)
    
    # Progress overview
    progress = report["progress"]
    print(f"Overall Progress: {progress['completed']}/{progress['total']} ({progress['progress_percent']:.1f}%)")
    print(f"Successful: {progress['successful']} ({progress['success_rate']:.1f}%)")
    print(f"Failed: {progress['failed']}")
    
    # Timing info
    timing = report["timing"]
    elapsed_hours = timing["elapsed_time"] / 3600
    print(f"Elapsed Time: {elapsed_hours:.2f} hours")
    
    if timing["estimated_remaining"]:
        remaining_hours = timing["estimated_remaining"] / 3600
        print(f"Estimated Remaining: {remaining_hours:.2f} hours")
    
    # Model progress
    print("\nModel Progress:")
    for model_name, model_progress in report["model_progress"].items():
        completed = model_progress["completed"]
        successful = model_progress["successful"]
        failed = model_progress["failed"]
        success_rate = (successful / max(1, completed)) * 100
        avg_time = model_progress["avg_time"]
        
        print(f"  {model_name}:")
        print(f"    Completed: {completed} (Success: {success_rate:.1f}%)")
        print(f"    Failed: {failed}")
        print(f"    Avg Time: {avg_time:.2f}s")
        
        if model_progress.get("last_error"):
            print(f"    Last Error: {model_progress['last_error']}")

async def resume_session(session_id: str):
    """Resume a collection session using original parameters"""
    logger = logging.getLogger(__name__)
    
    # First, try to find the session in default location to get output_dir
    default_output_dir = "generated"
    
    # Load session parameters first to get the actual output_dir
    logger.info("Loading original session parameters...")
    session_params = AssessmentStore.load_session_parameters(session_id, default_output_dir)
    
    if not session_params:
        print(f"Session {session_id} not found in {default_output_dir}")
        return 1
    
    # Get the actual output directory from session parameters
    output_dir = session_params.get("output_dir", default_output_dir)
    
    # Check if session exists
    tracker = ProgressTracker(output_dir)
    available_sessions = tracker.get_available_sessions()
    
    if session_id not in available_sessions:
        print(f"Session {session_id} not found")
        print(f"Available sessions: {available_sessions}")
        return 1
    
    # Check if session is actually completed (not just interrupted)
    session_dir = Path(output_dir) / session_id
    progress_file = session_dir / "progress.json"
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        session_status = progress_data.get("status", "unknown")
        
        if session_status == "completed":
            print(f"Session {session_id} is already completed")
            return 0
        elif session_status in ["failed", "in_progress"]:
            logger.info(f"Session {session_id} has status '{session_status}', resuming...")
        else:
            logger.warning(f"Session {session_id} has unknown status '{session_status}', attempting to resume...")
            
    except Exception as e:
        logger.warning(f"Could not read session status: {e}, attempting to resume...")
    
    try:
        # Extract original parameters
        original_models = session_params.get("models", [])
        original_data_sources = session_params.get("datasets")
        original_sets = session_params.get("sets", ["test"])  # Default to test if not specified
        original_quick_test = session_params.get("quick_test", False)
        original_dataset_path = session_params.get("dataset_path", "../dataset/generated")
        original_batch_size = session_params.get("batch_size", 10)
        original_max_concurrent = session_params.get("max_concurrent_per_model", 3)
        
        logger.info(f"Resuming with original parameters:")
        logger.info(f"  Models: {original_models}")
        logger.info(f"  Data sources: {original_data_sources}")
        logger.info(f"  Sets: {original_sets}")
        logger.info(f"  Quick test: {original_quick_test}")
        logger.info(f"  Dataset path: {original_dataset_path}")
        
        # Recreate the original models
        all_available_models = ModelConfigs.create_models_from_config()
        
        if not all_available_models:
            logger.error("No models could be initialized. Check API keys.")
            return 1
        
        # Filter to only the models that were used in the original session
        models = [model for model in all_available_models if model.model_name in original_models]
        
        if not models:
            logger.error(f"None of the original models {original_models} could be initialized.")
            logger.error("Available models: " + ", ".join([m.model_name for m in all_available_models]))
            return 1
        
        logger.info(f"Initialized {len(models)} models: {[m.model_name for m in models]}")
        
        # Get extraction API key
        extraction_key = ModelConfigs.get_extraction_config()
        
        # Create collection runner with original parameters
        runner = SolutionCollectionRunner(
            models=models,
            dataset_path=original_dataset_path,
            output_dir=output_dir,
            extraction_api_key=extraction_key,
            batch_size=original_batch_size,
            max_concurrent_per_model=original_max_concurrent
        )
        
        # Load problems using original data sources and sets
        logger.info("Loading dataset...")
        num_problems = runner.load_problems(original_data_sources, original_sets)
        logger.info(f"Loaded {num_problems} problems")
        
        # Resume collection with original parameters
        logger.info(f"Resuming collection session {session_id}...")
        results = await runner.collect_all_solutions(
            data_sources=original_data_sources,
            sets=original_sets,
            progress_callback=progress_callback,
            resume_session=session_id,
            quick_test=original_quick_test
        )
        
        # Print summary
        print("\nResume Summary:")
        print(f"Session ID: {results['session_id']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Success: {results['success']}")
        
        if results['success']:
            logger.info("Session resumed and completed successfully!")
            return 0
        else:
            logger.error("Resumed session failed")
            return 1
            
    except Exception as e:
        logger.exception(f"Failed to resume session: {e}")
        return 1

async def main():
    parser = argparse.ArgumentParser(description="Resume LLM solution collection")
    
    parser.add_argument("session_id", nargs="?", help="Session ID to resume")
    parser.add_argument("--list-sessions", action="store_true", help="List available sessions")
    parser.add_argument("--status", action="store_true", help="Show status of specified session")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.list_sessions:
        list_sessions("generated")  # Use default output directory for listing
        return 0
    
    if not args.session_id:
        print("Error: session_id is required unless using --list-sessions")
        parser.print_help()
        return 1
    
    if args.status:
        show_session_status(args.session_id, "generated")  # Use default for status
        return 0
    
    # Resume the session
    return await resume_session(args.session_id)

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
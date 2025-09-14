"""Main orchestrator for LLM solution collection"""

import asyncio
import json
import time
import signal
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
import logging

from ..models.base_model import BaseLLMModel
from ..processors.quality_extractor import QualityExtractor
from ..processors.response_validator import ResponseValidator
from ..storage.assessment_store import AssessmentStore
from ..storage.progress_tracker import ProgressTracker
from ..prompts.prompt_template import PromptTemplate
from .parallel_executor import ParallelExecutor, ExecutionTask
from .rate_limiter import AdaptiveRateLimiter

class SolutionCollectionRunner:
    """Main orchestrator for collecting solutions from multiple LLM providers"""
    
    def __init__(self, 
                 models: List[BaseLLMModel],
                 dataset_path: str,
                 output_dir: str,
                 extraction_api_key: Optional[str] = None,
                 batch_size: int = 10,
                 max_concurrent_per_model: int = 3):
        """
        Initialize the collection runner
        
        Args:
            models: List of LLM models to evaluate
            dataset_path: Path to the dataset directory containing JSONL files
            output_dir: Directory to store collected solutions
            extraction_api_key: API key for solution extraction (if using agentic approach)
            batch_size: Number of problems to process in each batch
            max_concurrent_per_model: Max concurrent requests per model
        """
        self.models = models
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.max_concurrent_per_model = max_concurrent_per_model
        
        # Initialize components (session_id will be set later in collect_all_solutions)
        self.assessment_store = None  # Will be initialized with session_id later
        self.progress_tracker = None  # Will be initialized with session_id later
        self.output_dir = output_dir
        self.executor = ParallelExecutor(
            max_concurrent_per_model=max_concurrent_per_model,
            max_total_concurrent=len(models) * max_concurrent_per_model
        )
        self.rate_limiter = AdaptiveRateLimiter()
        
        # Initialize quality extractor
        if extraction_api_key:
            self.quality_extractor = QualityExtractor(extraction_api_key)
        else:
            # Fallback to basic extraction if no API key provided
            self.quality_extractor = None
            
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register models with components
        for model in self.models:
            self.rate_limiter.register_model(model.model_name, model.rate_limit)
            self.executor.register_model(model.model_name, max_concurrent_per_model)
        
        # Collection state
        self.problems: List[Dict] = []
        self.session_id: Optional[str] = None
    
    def load_problems(self, data_sources: Optional[List[str]] = None, sets: Optional[List[str]] = None) -> int:
        """
        Load samples from integrated dataset files
        
        Args:
            data_sources: List of data sources to include (None for all: codeeval, codenet, codesearchnet, humaneval-x, mbpp)
            sets: List of dataset splits to load (default: ["test"])
            
        Returns:
            Number of samples loaded
        """
        self.problems = []
        
        # Default data sources if none specified
        available_data_sources = ['codeeval', 'codenet', 'codesearchnet', 'humaneval-x', 'mbpp']
        if data_sources is None:
            data_sources = available_data_sources
        
        # Default sets if none specified
        if sets is None:
            sets = ["test"]
        
        # Load from integrated dataset files
        base_path = Path("../dataset/generated/integrated")
        
        for data_source in data_sources:
            for dataset_set in sets:
                dataset_file = base_path / data_source / f"{dataset_set}.jsonl"
                
                # Skip if dataset file doesn't exist
                if not dataset_file.exists():
                    self.logger.warning(f"Dataset file not found: {dataset_file}")
                    continue
                
                try:
                    with open(dataset_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                try:
                                    sample = json.loads(line)
                                    sample['data_source'] = data_source
                                    sample['dataset_set'] = dataset_set
                                    sample['source_file'] = str(dataset_file)
                                    sample['line_number'] = line_num
                                    self.problems.append(sample)
                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Invalid JSON in {dataset_file}:{line_num}: {e}")
                                    
                except Exception as e:
                    self.logger.error(f"Failed to load {dataset_file}: {e}")
        
        sets_str = ", ".join(sets)
        self.logger.info(f"Loaded {len(self.problems)} samples from {len(data_sources)} data sources ({sets_str} sets)")
        return len(self.problems)
    
    def _filter_for_quick_test(self, data_sources: Optional[List[str]] = None) -> List[Dict]:
        """
        Filter problems for quick test: 1 problem per data source
        
        Args:
            data_sources: List of data sources to include (None for all data sources)
            
        Returns:
            Filtered list of problems (1 per data source)
        """
        if not self.problems:
            return []
        
        # Group problems by data source
        problems_by_data_source = {}
        for problem in self.problems:
            data_source = problem.get('data_source', 'unknown')
            if data_source not in problems_by_data_source:
                problems_by_data_source[data_source] = []
            problems_by_data_source[data_source].append(problem)
        
        # Filter to specified data sources if provided
        if data_sources:
            problems_by_data_source = {ds: probs for ds, probs in problems_by_data_source.items() 
                                     if ds in data_sources}
        
        # Take first problem from each data source
        quick_test_problems = []
        for data_source, source_problems in problems_by_data_source.items():
            if source_problems:
                quick_test_problems.append(source_problems[0])
                self.logger.info(f"Quick test: selected sample {source_problems[0].get('submission_id', 'unknown')} from data source '{data_source}'")
        
        return quick_test_problems
    
    def _get_successful_task_ids_set(self, task_ids_to_skip: List[str]) -> set:
        """
        Convert list of successful task IDs to a set for efficient lookup
        
        Args:
            task_ids_to_skip: List of successful task IDs (format: "problem_id_model_name")
            
        Returns:
            Set of successful task IDs for O(1) lookup during task creation
        """
        return set(task_ids_to_skip) if task_ids_to_skip else set()
    
    async def collect_all_solutions(self, 
                                  data_sources: Optional[List[str]] = None,
                                  sets: Optional[List[str]] = None,
                                  progress_callback: Optional[Callable] = None,
                                  resume_session: Optional[str] = None,
                                  quick_test: bool = False) -> Dict:
        """
        Main method to collect solutions from all models for all problems
        
        Args:
            data_sources: List of data sources to include
            sets: List of dataset splits to process (train, valid, test)
            progress_callback: Optional callback for progress updates
            resume_session: Session ID to resume from previous run
            quick_test: If True, only process 1 problem per category
            
        Returns:
            Collection results summary
        """
        start_time = time.time()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, saving solutions and exiting...")
            # Save solutions immediately
            try:
                self.assessment_store.save_assessments_jsonl()
                self.assessment_store.save_assessments_csv()
                self.assessment_store.save_session_summary()
                self.progress_tracker.finalize_session(False)
            except Exception as e:
                self.logger.error(f"Error saving solutions during signal handler: {e}")
            raise KeyboardInterrupt()
        
        # Register signal handler for SIGINT (Ctrl+C)
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        # Load problems
        if not self.problems:
            num_problems = self.load_problems(data_sources, sets)
            if num_problems == 0:
                raise ValueError("No problems loaded")
        
        # Initialize session ID and create session directory
        if resume_session:
            self.session_id = resume_session
            self.logger.info(f"Attempting to resume session {resume_session}")
            
            # For resume: Session directory should already exist
            session_dir = Path(self.output_dir) / self.session_id
            if not session_dir.exists():
                raise FileNotFoundError(f"Resume session directory not found: {session_dir}")
            
            # Initialize components for existing session
            if self.progress_tracker is None:
                self.progress_tracker = ProgressTracker(self.output_dir, self.session_id, is_resume=True)
            if self.assessment_store is None:
                self.assessment_store = AssessmentStore(self.output_dir, self.session_id, is_resume=True)
                
            # Load previous session data
            if not self.progress_tracker.load_previous_session(self.session_id):
                self.logger.error(f"Failed to load previous session data for {resume_session}")
                raise RuntimeError(f"Could not load session data from {session_dir}")
                
            self.logger.info(f"Successfully loaded existing session {resume_session}")
        else:
            # NEW SESSION: Create new session ID and session directory
            self.session_id = f"session_{int(start_time)}"
            session_dir = Path(self.output_dir) / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created new session directory: {session_dir}")
            
            # Initialize components with session_id
            if self.assessment_store is None:
                self.assessment_store = AssessmentStore(self.output_dir, self.session_id)
            if self.progress_tracker is None:
                self.progress_tracker = ProgressTracker(self.output_dir, self.session_id)
            
        # Handle session type: new vs resume  
        if resume_session:
            # For resume: Apply original quick-test filtering FIRST if it was a quick-test session
            if quick_test:
                original_count = len(self.problems)
                self.problems = self._filter_for_quick_test(data_sources)
                self.logger.info(f"Resume quick test: restored original selection of {len(self.problems)} problems (1 per data source)")
            
            # Get successful task IDs for task-level filtering during batch processing
            # We no longer filter problems here - we filter individual tasks later
            successful_task_ids = self.progress_tracker.get_successful_task_ids()
            self.successful_task_ids_set = self._get_successful_task_ids_set(successful_task_ids)
            
            # Calculate remaining tasks for logging
            total_possible_tasks = len(self.problems) * len(self.models)
            tasks_to_skip = len(successful_task_ids)
            remaining_tasks = total_possible_tasks - tasks_to_skip
            
            self.logger.info(f"Resume: Original session had {len(self.problems)} problems × {len(self.models)} models = {total_possible_tasks} total tasks")
            self.logger.info(f"Resume: {tasks_to_skip} tasks already completed successfully, {remaining_tasks} tasks remaining")
        else:
            # NEW SESSION: Apply quick test filtering if requested
            if quick_test:
                original_count = len(self.problems)
                self.problems = self._filter_for_quick_test(data_sources)
                self.logger.info(f"Quick test: reduced from {original_count} problems to {len(self.problems)} problems (1 per data source)")
            
            # No successful tasks to skip in new session
            self.successful_task_ids_set = set()
            
        # Initialize progress tracking
        model_names = [model.model_name for model in self.models]
        category_names = list(set(p.get('category', 'unknown') for p in self.problems))
        
        # Calculate total tasks (accounting for already successful tasks in resume sessions)
        total_possible_tasks = len(self.problems) * len(self.models)
        if hasattr(self, 'successful_task_ids_set'):
            # For resume: total tasks = possible tasks - already successful tasks
            total_tasks = total_possible_tasks - len(self.successful_task_ids_set)
        else:
            # For new session: all tasks need to be completed
            total_tasks = total_possible_tasks
        
        self.progress_tracker.initialize_session(total_tasks, model_names, category_names)
        
        # Store session parameters in solution store for resume functionality
        self.assessment_store.set_session_parameters(
            models=model_names,
            datasets=data_sources,
            sets=sets or ["test"],
            quick_test=quick_test,
            dataset_path=str(self.dataset_path),
            batch_size=self.batch_size,
            max_concurrent_per_model=self.max_concurrent_per_model,
            output_dir=self.output_dir
        )
        
        # Set total problems count in metadata
        self.assessment_store.set_total_samples(len(self.problems))
        
        self.logger.info(f"Starting collection session {self.session_id}")
        if hasattr(self, 'successful_task_ids_set') and self.successful_task_ids_set:
            # Resume session: show remaining tasks
            self.logger.info(f"Tasks to process: {total_tasks} (resuming from previous session)")
        else:
            # New session: show total tasks
            self.logger.info(f"Total tasks: {total_tasks} ({len(self.problems)} problems × {len(self.models)} models)")
        
        try:
            # Process in batches
            await self._process_batches(progress_callback, resume_session=resume_session)
            
            # Finalize collection
            collection_summary = self._finalize_collection(start_time, True)
            
            return collection_summary
            
        except Exception as e:
            self.logger.error(f"Collection failed: {e}")
            self._finalize_collection(start_time, False)
            raise
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
    
    async def _process_batches(self, progress_callback: Optional[Callable] = None, resume_session: Optional[str] = None) -> List[Dict]:
        """Process problems in batches"""
        all_results = []
        
        # Step 1: Generate all possible task IDs and their associated data
        all_tasks = []
        for problem in self.problems:
            for model in self.models:
                task_id = f"{problem.get('submission_id', problem.get('problem_id', 'unknown'))}_{model.model_name}"
                all_tasks.append({
                    'task_id': task_id,
                    'model': model,
                    'problem': problem
                })
        
        # Step 2: For resume sessions, filter out successful tasks
        if resume_session and hasattr(self, 'successful_task_ids_set') and self.successful_task_ids_set:
            tasks_to_process = [t for t in all_tasks if t['task_id'] not in self.successful_task_ids_set]
            # Log message moved to collect_all_solutions to avoid duplication
        else:
            tasks_to_process = all_tasks
        
        # Step 3: Process tasks in batches based on tasks_to_process
        for batch_num, batch_start in enumerate(range(0, len(tasks_to_process), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(tasks_to_process))
            batch_tasks = tasks_to_process[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_num + 1}: tasks {batch_start + 1}-{batch_end} of {len(tasks_to_process)}")
            
            # Create ExecutionTask objects for this batch
            tasks = []
            for task_data in batch_tasks:
                task = ExecutionTask(
                    task_id=task_data['task_id'],
                    model_name=task_data['model'].model_name,
                    problem_data=task_data['problem'],
                    execute_func=self._create_execution_function(task_data['model']),
                    priority=task_data['problem'].get('complexity', 1)
                )
                tasks.append(task)
            
            # Submit tasks to executor
            for task in tasks:
                self.executor.submit_task(task)
            
            # Execute batch
            batch_results = await self.executor.execute_all(
                progress_callback=self._create_progress_callback(progress_callback)
            )
            
            all_results.extend(batch_results)
            
            # Process results and update storage
            await self._process_batch_results(batch_results)
            
            # Reset executor for next batch
            self.executor.reset()
            
            self.logger.info(f"Completed batch {batch_num + 1}")
        
        return all_results
    
    def _create_execution_function(self, model: BaseLLMModel):
        """Create execution function for a specific model"""
        async def execute_problem(problem_data: Dict, task_id: str) -> Dict:
            # Rate limiting
            await self.rate_limiter.acquire(model.model_name)
            
            # Generate prompt
            prompt_data = PromptTemplate.format_for_chat_model(problem_data)
            
            # Generate solution
            start_time = time.time()
            try:
                response = await model.generate_solution(prompt_data["user"], task_id)
                execution_time = time.time() - start_time
                
                # Record response for adaptive rate limiting
                self.rate_limiter.record_response(model.model_name, execution_time, response["success"])
                
                if not response["success"]:
                    return {
                        "task_id": task_id,
                        "model": model.model_name,
                        "llm_call_success": False,
                        "error": response.get("error", "Unknown error"),
                        "execution_time": execution_time
                    }
                
                # Extract quality scores from assessment response
                raw_response = response["content"]
                quality_scores = {}
                
                if self.quality_extractor:
                    quality_scores = await self.quality_extractor.extract_quality_scores(
                        raw_response, task_id
                    )
                else:
                    # Basic extraction fallback for quality scores
                    quality_scores = self._basic_quality_extract(raw_response)
                
                # No validation needed for assessments - we're not generating code
                
                return {
                    "sample_id": problem_data.get('submission_id', problem_data.get('problem_id', 'unknown')),
                    "model": model.model_name,
                    "problem": problem_data.get("problem", ""),
                    "submission": problem_data.get("submission", ""),
                    "dataset": problem_data.get("source", "unknown"),
                    "quality_scores": quality_scores,
                    "raw_response": raw_response,
                    "execution_time": execution_time,
                    "llm_call_success": True,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.rate_limiter.record_response(model.model_name, execution_time, False)
                
                return {
                    "task_id": task_id,
                    "model": model.model_name,
                    "llm_call_success": False,
                    "error": str(e),
                    "execution_time": execution_time
                }
        
        return execute_problem
    
    def _basic_quality_extract(self, response: str) -> Dict:
        """Basic fallback extraction of quality scores from text"""
        # Try to find numeric scores in the response
        import re
        
        score_patterns = {
            'functionality': r'functionality[:\s]*(\d+)',
            'readability': r'readability[:\s]*(\d+)',
            'idiomatic': r'idiomatic[:\s]*(\d+)',
            'error_handling': r'error[_\s]handling[:\s]*(\d+)',
            'efficiency': r'efficiency[:\s]*(\d+)'
        }
        
        scores = {}
        for dimension, pattern in score_patterns.items():
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = int(match.group(1))
                    # Ensure score is in valid range
                    if 1 <= score <= 5:
                        scores[dimension] = score
                except ValueError:
                    pass
        
        # If we couldn't extract all scores, return defaults
        default_scores = {
            'functionality': 3,
            'readability': 3, 
            'idiomatic': 3,
            'error_handling': 3,
            'efficiency': 3
        }
        
        for dimension in default_scores:
            if dimension not in scores:
                scores[dimension] = default_scores[dimension]
        
        return scores

    def _basic_extract(self, response: str) -> str:
        """Basic fallback extraction if no extractor API key provided"""
        import re
        
        # Try to find code blocks
        pattern = r'```(?:python|py)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the response as-is
        return response.strip()
    
    def _create_progress_callback(self, user_callback: Optional[Callable]):
        """Create progress callback that updates tracking and calls user callback"""
        def progress_callback(completed: int, total: int, is_final: bool):
            # Update progress tracker would happen in _process_batch_results
            
            # Call user callback if provided
            if user_callback:
                user_callback(completed, total, is_final)
        
        return progress_callback
    
    async def _process_batch_results(self, results: List[Any]):
        """Process batch results and update storage/tracking"""
        for result in results:
            if hasattr(result, 'success') and hasattr(result, 'result'):
                # This is an ExecutionResult object
                execution_result = result.result if result.success else {}
                task_id = execution_result.get("task_id", result.task_id)
                model = execution_result.get("model", result.model_name)
                success = result.success and execution_result.get("llm_call_success", False)
                error = result.error or execution_result.get("error")
                
                # Update progress tracker
                self.progress_tracker.update_progress(
                    task_id=task_id,
                    model=model,
                    success=success,
                    execution_time=result.execution_time,
                    error=error
                )
                
                # Add to solution store
                if success and execution_result:
                    self.assessment_store.add_single_assessment(
                        sample_id=task_id,
                        model_name=model,
                        assessment_data=execution_result
                    )
    
    def _finalize_collection(self, start_time: float, success: bool) -> Dict:
        """Finalize the collection process"""
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save solutions
        jsonl_path = self.assessment_store.save_assessments_jsonl()
        csv_path = self.assessment_store.save_assessments_csv()
        summary_path = self.assessment_store.save_session_summary()
        
        # Finalize progress tracking
        final_report = self.progress_tracker.finalize_session(success)
        
        # Get rate limiter stats
        rate_stats = self.rate_limiter.get_adaptive_stats()
        
        collection_summary = {
            "session_id": self.session_id,
            "success": success,
            "duration": total_time,
            "problems_processed": len(self.problems),
            "models_evaluated": [m.model_name for m in self.models],
            "output_files": {
                "solutions_jsonl": str(jsonl_path),
                "solutions_csv": str(csv_path),
                "analysis_summary": str(summary_path)
            },
            "progress_report": final_report,
            "rate_limiting_stats": rate_stats
        }
        
        self.logger.info(f"Collection completed in {total_time:.2f} seconds")
        self.logger.info(f"Results saved to {jsonl_path}")
        
        return collection_summary
"""Parallel execution manager for coordinating multiple LLM requests"""

import asyncio
import time
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class ExecutionTask:
    """Represents a single execution task"""
    task_id: str
    model_name: str 
    problem_data: Dict
    execute_func: Callable
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass 
class ExecutionResult:
    """Represents the result of an execution task"""
    task_id: str
    model_name: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    completed_at: float = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = time.time()

class ParallelExecutor:
    """
    Manages parallel execution of LLM requests with proper concurrency control,
    progress tracking, and error handling
    """
    
    def __init__(self, max_concurrent_per_model: int = 5, max_total_concurrent: int = 20):
        """
        Initialize the parallel executor
        
        Args:
            max_concurrent_per_model: Max concurrent requests per model
            max_total_concurrent: Max total concurrent requests across all models
        """
        self.max_concurrent_per_model = max_concurrent_per_model
        self.max_total_concurrent = max_total_concurrent
        
        # Semaphores for concurrency control
        self.total_semaphore = asyncio.Semaphore(max_total_concurrent)
        self.model_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Task tracking
        self.pending_tasks: List[ExecutionTask] = []
        self.running_tasks: Dict[str, ExecutionTask] = {}  # task_id -> task
        self.completed_results: List[ExecutionResult] = []
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "start_time": None,
            "model_stats": {}
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model_name: str, max_concurrent: Optional[int] = None):
        """Register a model for execution tracking"""
        concurrent_limit = max_concurrent or self.max_concurrent_per_model
        self.model_semaphores[model_name] = asyncio.Semaphore(concurrent_limit)
        self.stats["model_stats"][model_name] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "total_time": 0.0
        }
        self.logger.info(f"Registered {model_name} with max concurrent: {concurrent_limit}")
    
    def submit_task(self, task: ExecutionTask):
        """Submit a task for execution"""
        if task.model_name not in self.model_semaphores:
            self.register_model(task.model_name)
        
        self.pending_tasks.append(task)
        self.stats["tasks_submitted"] += 1
        self.stats["model_stats"][task.model_name]["submitted"] += 1
        
        self.logger.debug(f"Submitted task {task.task_id} for model {task.model_name}")
    
    async def execute_all(self, progress_callback: Optional[Callable] = None) -> List[ExecutionResult]:
        """
        Execute all submitted tasks in parallel
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of execution results
        """
        if not self.pending_tasks:
            self.logger.info("No tasks to execute")
            return []
        
        self.stats["start_time"] = time.time()
        total_tasks = len(self.pending_tasks)
        
        self.logger.info(f"Starting execution of {total_tasks} tasks")
        
        # Create coroutines for all tasks
        coroutines = []
        for task in self.pending_tasks:
            coro = self._execute_single_task(task, progress_callback)
            coroutines.append(coro)
        
        # Clear pending tasks since we're now executing them
        self.pending_tasks.clear()
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results (some might be exceptions)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed task
                error_result = ExecutionResult(
                    task_id=f"unknown_{i}",
                    model_name="unknown",
                    success=False,
                    error=str(result)
                )
                self.completed_results.append(error_result)
            elif isinstance(result, ExecutionResult):
                self.completed_results.append(result)
        
        # Final progress callback
        if progress_callback:
            progress_callback(len(self.completed_results), total_tasks, True)
        
        end_time = time.time()
        self.stats["total_execution_time"] = end_time - self.stats["start_time"]
        
        self.logger.info(f"Completed execution of {len(self.completed_results)} tasks in {self.stats['total_execution_time']:.2f}s")
        
        return self.completed_results.copy()
    
    async def _execute_single_task(self, task: ExecutionTask, progress_callback: Optional[Callable] = None) -> ExecutionResult:
        """Execute a single task with proper concurrency control"""
        
        # Acquire semaphores (both total and model-specific)
        async with self.total_semaphore:
            async with self.model_semaphores[task.model_name]:
                
                # Track running task
                self.running_tasks[task.task_id] = task
                
                start_time = time.time()
                
                try:
                    # Execute the actual task
                    result = await task.execute_func(task.problem_data, task.task_id)
                    
                    execution_time = time.time() - start_time
                    
                    # Create success result
                    exec_result = ExecutionResult(
                        task_id=task.task_id,
                        model_name=task.model_name,
                        success=True,
                        result=result,
                        execution_time=execution_time
                    )
                    
                    # Update stats
                    self.stats["tasks_completed"] += 1
                    self.stats["model_stats"][task.model_name]["completed"] += 1
                    self.stats["model_stats"][task.model_name]["total_time"] += execution_time
                    
                    self.logger.debug(f"Completed task {task.task_id} in {execution_time:.2f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Create error result
                    exec_result = ExecutionResult(
                        task_id=task.task_id,
                        model_name=task.model_name,
                        success=False,
                        error=str(e),
                        execution_time=execution_time
                    )
                    
                    # Update stats
                    self.stats["tasks_failed"] += 1
                    self.stats["model_stats"][task.model_name]["failed"] += 1
                    
                    self.logger.error(f"Failed task {task.task_id}: {e}")
                
                finally:
                    # Remove from running tasks
                    self.running_tasks.pop(task.task_id, None)
                    
                    # Progress callback
                    if progress_callback:
                        completed_count = self.stats["tasks_completed"] + self.stats["tasks_failed"]
                        total_count = self.stats["tasks_submitted"]
                        progress_callback(completed_count, total_count, False)
                
                return exec_result
    
    def get_progress(self) -> Dict:
        """Get current execution progress"""
        completed = self.stats["tasks_completed"] + self.stats["tasks_failed"]
        total = self.stats["tasks_submitted"]
        
        progress = {
            "total_tasks": total,
            "completed_tasks": completed,
            "successful_tasks": self.stats["tasks_completed"],
            "failed_tasks": self.stats["tasks_failed"],
            "running_tasks": len(self.running_tasks),
            "pending_tasks": len(self.pending_tasks),
            "progress_percent": (completed / max(1, total)) * 100,
            "elapsed_time": time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        }
        
        # Add per-model progress
        for model_name, model_stats in self.stats["model_stats"].items():
            model_completed = model_stats["completed"] + model_stats["failed"]
            model_total = model_stats["submitted"]
            
            progress[f"{model_name}_progress"] = {
                "completed": model_completed,
                "total": model_total,
                "success_rate": model_stats["completed"] / max(1, model_completed) * 100,
                "avg_execution_time": model_stats["total_time"] / max(1, model_stats["completed"])
            }
        
        return progress
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed execution statistics"""
        progress = self.get_progress()
        
        # Add result analysis
        if self.completed_results:
            execution_times = [r.execution_time for r in self.completed_results if r.success]
            
            if execution_times:
                progress.update({
                    "min_execution_time": min(execution_times),
                    "max_execution_time": max(execution_times),
                    "avg_execution_time": sum(execution_times) / len(execution_times),
                    "total_execution_time": self.stats["total_execution_time"]
                })
        
        return progress
    
    def reset(self):
        """Reset the executor for a new batch of tasks"""
        self.pending_tasks.clear()
        self.running_tasks.clear()
        self.completed_results.clear()
        
        # Reset stats
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "start_time": None,
            "model_stats": {model: {"submitted": 0, "completed": 0, "failed": 0, "total_time": 0.0} 
                          for model in self.model_semaphores.keys()}
        }
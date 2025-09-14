"""Progress tracking for long-running LLM evaluation collections"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

@dataclass
class ProgressCheckpoint:
    """Represents a progress checkpoint for a single run (original or resume)"""
    is_resume: bool
    run_type: str  # "original" or "resume"
    timestamp: float
    completed_tasks: int
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    model_progress: Dict[str, Dict[str, Any]]
    successful_task_ids: List[str] = None  # List of successful task_ids
    failed_task_ids: List[str] = None  # List of failed task_ids
    current_batch: Optional[int] = None
    estimated_completion: Optional[float] = None

class ProgressTracker:
    """Tracks and persists progress of LLM evaluation collection"""
    
    def __init__(self, output_dir: str, session_id: Optional[str] = None, is_resume: bool = False):
        """
        Initialize progress tracker
        
        Args:
            output_dir: Directory to store progress files
            session_id: Unique session identifier (auto-generated if None)
            is_resume: Whether this is a resume session (affects progress file naming)
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        
        self.session_id = session_id or f"session_{int(time.time())}"
        self.is_resume = is_resume
        
        # Set session directory path (but don't create it here)
        self.session_dir = self.base_output_dir / self.session_id
        
        # Both original and resume sessions use the same progress.json file
        self.progress_file = self.session_dir / "progress.json"
        
        # Progress state
        self.start_time = time.time()
        self.last_update = self.start_time
        self.successful_task_ids: List[str] = []  # Track successful task_ids
        self.failed_task_ids: List[str] = []  # Track failed task_ids
        
        # Progress history - list of all runs (original + resumes)
        self.progresses: List[ProgressCheckpoint] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Auto-save settings
        self.auto_save_interval = 30  # seconds
        self.last_auto_save = time.time()
    
    
    def initialize_session(self, total_tasks: int, models: List[str], categories: List[str]):
        """
        Initialize a new collection session or add a new run to existing session
        
        Args:
            total_tasks: Total number of tasks to complete in this run
            models: List of model names being evaluated
            categories: List of problem categories
        """
        # with self._lock:  # TODO: Replace with asyncio.Lock for async compatibility
        
        # Create new progress entry for this run
        new_progress = ProgressCheckpoint(
            is_resume=self.is_resume,
            run_type="resume" if self.is_resume else "original",
            timestamp=self.start_time,
            completed_tasks=0,
            total_tasks=total_tasks,
            successful_tasks=0,
            failed_tasks=0,
            model_progress={
                model: {
                    "completed": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_time": 0.0,
                    "last_error": None
                } for model in models
            },
            successful_task_ids=[],
            failed_task_ids=[]
        )
        
        # Add to progresses list
        self.progresses.append(new_progress)
        
        # Update session metadata (only if new session or updating existing)
        if not hasattr(self, 'session_metadata') or not self.session_metadata:
            # New session
            self.session_metadata = {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "models": models,
                "categories": categories,
                "status": "in_progress"
            }
        
        # Always save current state
        self._save_progress()
    
    def update_progress(self, task_id: str, model: str, success: bool, 
                       execution_time: float = 0.0, error: Optional[str] = None):
        """
        Update progress for a completed task
        
        Args:
            task_id: Completed task identifier
            model: Model that completed the task
            success: Whether the task was successful
            execution_time: Time taken to execute the task
            error: Error message if task failed
        """
        # with self._lock:  # TODO: Replace with asyncio.Lock for async compatibility
        current_time = time.time()
        
        # Get current progress (last entry in progresses list)
        if not self.progresses:
            raise RuntimeError("No progress entries found. Call initialize_session first.")
        
        current_progress = self.progresses[-1]
        
        # Track successful/failed tasks separately
        if success:
            if task_id not in self.successful_task_ids:
                self.successful_task_ids.append(task_id)
                current_progress.successful_task_ids = self.successful_task_ids.copy()
            # Remove from failed if it was previously failed (retry succeeded)
            if task_id in self.failed_task_ids:
                self.failed_task_ids.remove(task_id)
                current_progress.failed_task_ids = self.failed_task_ids.copy()
        else:
            if task_id not in self.failed_task_ids:
                self.failed_task_ids.append(task_id)
                current_progress.failed_task_ids = self.failed_task_ids.copy()
            # Remove from successful if it was previously successful (retry failed)
            if task_id in self.successful_task_ids:
                self.successful_task_ids.remove(task_id)
                current_progress.successful_task_ids = self.successful_task_ids.copy()
        
        # Update overall progress
        current_progress.completed_tasks = len(self.successful_task_ids) + len(self.failed_task_ids)
        current_progress.successful_tasks = len(self.successful_task_ids)
        current_progress.failed_tasks = len(self.failed_task_ids)
        
        # Update model-specific progress
        if model in current_progress.model_progress:
            model_stats = current_progress.model_progress[model]
            model_stats["completed"] += 1
            
            if success:
                model_stats["successful"] += 1
            else:
                model_stats["failed"] += 1
                model_stats["last_error"] = error
            
            # Update average execution time
            current_avg = model_stats["avg_time"]
            completed = model_stats["completed"]
            model_stats["avg_time"] = (current_avg * (completed - 1) + execution_time) / completed
        
        # Update timestamp
        current_progress.timestamp = current_time
        self.last_update = current_time
        
        # Estimate completion time
        if current_progress.completed_tasks > 0:
            elapsed = current_time - self.start_time
            rate = current_progress.completed_tasks / elapsed
            remaining_tasks = current_progress.total_tasks - current_progress.completed_tasks
            if rate > 0:
                current_progress.estimated_completion = current_time + (remaining_tasks / rate)
        
        # Auto-save if enough time has passed
        if current_time - self.last_auto_save > self.auto_save_interval:
            self._save_progress()
            self.last_auto_save = current_time
    
    
    def get_progress_report(self) -> Dict:
        """Get current progress report"""
        # with self._lock:  # TODO: Replace with asyncio.Lock for async compatibility
        if not self.progresses:
            return {"error": "No progress data available"}
        
        current_progress = self.progresses[-1]
        elapsed_time = time.time() - self.start_time
        
        progress_percent = 0
        if current_progress.total_tasks > 0:
            progress_percent = (current_progress.completed_tasks / 
                             current_progress.total_tasks) * 100
        
        # Calculate overall success rate
        success_rate = 0
        if current_progress.completed_tasks > 0:
            success_rate = (current_progress.successful_tasks / 
                          current_progress.completed_tasks) * 100
        
        # Estimate remaining time
        remaining_time = None
        if current_progress.estimated_completion:
            remaining_time = max(0, current_progress.estimated_completion - time.time())
        
        report = {
            "session_id": self.session_id,
            "current_run": {
                "run_type": current_progress.run_type,
                "is_resume": current_progress.is_resume,
                "completed": current_progress.completed_tasks,
                "total": current_progress.total_tasks,
                "successful": current_progress.successful_tasks,
                "failed": current_progress.failed_tasks,
                "progress_percent": progress_percent,
                "success_rate": success_rate
            },
            "timing": {
                "elapsed_time": elapsed_time,
                "estimated_remaining": remaining_time,
                "estimated_total": elapsed_time + (remaining_time or 0),
                "last_update": self.last_update
            },
            "model_progress": current_progress.model_progress.copy(),
            "run_history": {
                "total_runs": len(self.progresses),
                "original_run": asdict(self.progresses[0]) if self.progresses else None,
                "current_run_index": len(self.progresses) - 1
            }
        }
        
        return report
    
    def _save_progress(self):
        """Save current progress to file"""
        if not self.progresses:
            return
        
        current_progress = self.progresses[-1]
        
        # Update status based on completion
        if (current_progress.completed_tasks == current_progress.total_tasks and 
            current_progress.total_tasks > 0):
            # Set status to "failed" if there are any failed tasks, otherwise "completed"
            if current_progress.failed_tasks > 0:
                self.session_metadata["status"] = "failed"
            else:
                self.session_metadata["status"] = "completed"
        
        progress_data = {
            **self.session_metadata,
            "progresses": [asdict(progress) for progress in self.progresses],
            "last_saved": time.time()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    
    def load_previous_session(self, session_id: str) -> bool:
        """
        Load progress from a previous session
        
        Args:
            session_id: Session to load
            
        Returns:
            True if session was loaded successfully
        """
        session_dir = self.base_output_dir / session_id
        progress_file = session_dir / "progress.json"
        
        try:
            # Load progress
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                self.session_id = session_id
                
                # Update session directory and progress file to point to the loaded session
                self.session_dir = self.base_output_dir / session_id
                self.progress_file = self.session_dir / "progress.json"
                
                # Load progresses list (handle both new and old format)
                if "progresses" in progress_data:
                    # New format with progresses list
                    self.progresses = [ProgressCheckpoint(**progress) for progress in progress_data["progresses"]]
                elif "current_progress" in progress_data:
                    # Old format - convert to new format
                    old_progress = progress_data["current_progress"]
                    # Add missing fields for backward compatibility
                    if "is_resume" not in old_progress:
                        old_progress["is_resume"] = False
                    if "run_type" not in old_progress:
                        old_progress["run_type"] = "original"
                    self.progresses = [ProgressCheckpoint(**old_progress)]
                else:
                    return False
                
                # Restore session metadata
                self.session_metadata = {k: v for k, v in progress_data.items() 
                                       if k not in ["progresses", "current_progress", "last_saved"]}
                
                # Restore successful/failed task IDs from the latest progress
                if self.progresses:
                    latest_progress = self.progresses[-1]
                    self.successful_task_ids = latest_progress.successful_task_ids or []
                    self.failed_task_ids = latest_progress.failed_task_ids or []
                
                return True
            else:
                return False
            
        except Exception as e:
            print(f"Failed to load session {session_id}: {e}")
            return False
    
    def finalize_session(self, success: bool = True):
        """Mark session as completed"""
        # with self._lock:  # TODO: Replace with asyncio.Lock for async compatibility
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Update session status
        self.session_metadata["status"] = "completed" if success else "failed"
        self._save_progress()  # Save final progress with updated status
        
        final_report = self.get_progress_report()
        final_report.update({
            "status": "completed" if success else "failed",
            "end_time": end_time,
            "total_duration": total_time,
            "final_checkpoint": asdict(self.progresses[-1]) if self.progresses else None,
            "session_summary": {
                "total_runs": len(self.progresses),
                "all_runs": [asdict(progress) for progress in self.progresses]
            }
        })
        
        # Save final report in session directory
        final_file = self.session_dir / "final_report.json"
        with open(final_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        return final_report
    
    def get_successful_task_ids(self) -> List[str]:
        """Get list of successful task IDs for resume functionality"""
        return self.successful_task_ids.copy()
    
    def get_failed_task_ids(self) -> List[str]:
        """Get list of failed task IDs for resume functionality"""
        return self.failed_task_ids.copy()
    
    def get_available_sessions(self) -> List[str]:
        """Get list of available session IDs"""
        sessions = []
        
        # Check for session directories with progress.json
        for session_dir in self.base_output_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                progress_file = session_dir / "progress.json"
                if progress_file.exists():
                    sessions.append(session_dir.name)
        
        return sorted(sessions)

"""Storage system for collected LLM quality assessments"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd

class AssessmentStore:
    """Handles storage and retrieval of LLM quality assessments"""
    
    def __init__(self, output_dir: str, session_id: str = None, is_resume: bool = False):
        """
        Initialize the assessment store
        
        Args:
            output_dir: Directory to store collected assessments
            session_id: Session identifier for organizing model-specific folders
            is_resume: Whether this is a resume session (affects how assessments are saved)
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.is_resume = is_resume
        
        # Set session directory path (but don't create it here)
        self.output_dir = self.base_output_dir / self.session_id
        
        # Assessments grouped by model and dataset
        self.assessments_by_model_dataset: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.metadata = {
            "collection_start": None,
            "collection_end": None,
            "total_samples": 0,
            "models_evaluated": [],
            "datasets_included": [],
            "session_id": self.session_id,
            "session_parameters": {
                "models": [],
                "datasets": None,
                "sets": None,
                "quick_test": False,
                "dataset_path": None,
                "human_annotations_path": None,
                "batch_size": None,
                "max_concurrent_per_model": None,
                "output_dir": None
            }
        }
    
    def set_session_parameters(self, models: List[str], datasets: Optional[List[str]] = None,
                              sets: Optional[List[str]] = None,
                              quick_test: bool = False, dataset_path: str = None,
                              human_annotations_path: str = None,
                              batch_size: int = None, max_concurrent_per_model: int = None,
                              output_dir: str = None):
        """
        Store the original session parameters for resume functionality
        
        Args:
            models: List of model names used in the session
            datasets: List of datasets processed (None for all)
            sets: List of dataset splits processed (train, valid, test)
            quick_test: Whether this was a quick test session
            dataset_path: Path to the dataset directory
            human_annotations_path: Path to human annotations
            batch_size: Batch size used
            max_concurrent_per_model: Concurrency limit used
            output_dir: Output directory where session is stored
        """
        self.metadata["session_parameters"].update({
            "models": models,
            "datasets": datasets,
            "sets": sets,
            "quick_test": quick_test,
            "dataset_path": dataset_path,
            "human_annotations_path": human_annotations_path,
            "batch_size": batch_size,
            "max_concurrent_per_model": max_concurrent_per_model,
            "output_dir": output_dir
        })
        
        # Also update the legacy fields for compatibility
        self.metadata["models_evaluated"] = models
        self.metadata["datasets_included"] = datasets or []
        
        # Set collection start time
        if self.metadata["collection_start"] is None:
            self.metadata["collection_start"] = time.time()
    
    def set_total_samples(self, total_samples: int):
        """Set the total number of samples being processed"""
        self.metadata["total_samples"] = total_samples

    def add_assessments(self, assessments: List[Dict]):
        """Add a batch of assessments to the store"""
        for assessment in assessments:
            model_name = assessment.get("model", "unknown")
            dataset = assessment.get("dataset", "unknown")
            self.assessments_by_model_dataset[model_name][dataset].append(assessment)
    
    def add_single_assessment(self, sample_id: str, model_name: str, assessment_data: Dict):
        """
        Add a single assessment result
        
        Args:
            sample_id: Sample identifier
            model_name: Model that generated the assessment
            assessment_data: Assessment data including quality scores, justifications, etc.
        """
        assessment_record = {
            "sample_id": sample_id,
            "model": model_name,
            "timestamp": time.time(),
            **assessment_data
        }
        dataset = assessment_data.get("dataset", "unknown")
        self.assessments_by_model_dataset[model_name][dataset].append(assessment_record)
    
    def _create_model_directory(self, model_name: str) -> Path:
        """Create directory for model-specific assessments"""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        return model_dir
    
    def save_assessments_jsonl(self) -> Dict[str, Dict[str, Path]]:
        """
        Save assessments in JSONL format, one file per dataset per model
        
        Returns:
            Dictionary mapping model names to dictionaries of dataset -> file path
        """
        output_paths = {}
        
        for model_name, datasets in self.assessments_by_model_dataset.items():
            if not datasets:
                continue
                
            model_dir = self._create_model_directory(model_name)
            model_paths = {}
            
            for dataset, assessments in datasets.items():
                if not assessments:
                    continue
                    
                filename = f"{dataset}.jsonl"
                output_path = model_dir / filename
                
                # For resume sessions, load existing assessments first
                if self.is_resume and output_path.exists():
                    existing_assessments = self.load_assessments_from_jsonl(model_name, dataset)
                    
                    # Create a set of existing sample_ids to avoid duplicates
                    existing_sample_ids = {ass.get('sample_id') for ass in existing_assessments}
                    
                    # Filter out solutions that already exist
                    new_assessments = [ass for ass in assessments 
                                     if ass.get('sample_id') not in existing_sample_ids]
                    
                    # Combine existing with new
                    all_assessments = existing_assessments + new_assessments
                else:
                    all_assessments = assessments
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for assessment in all_assessments:
                        f.write(json.dumps(assessment) + '\n')
                
                model_paths[dataset] = output_path
                print(f"Saved {len(all_assessments)} assessments to {output_path}")
            
            output_paths[model_name] = model_paths
        
        return output_paths
    
    def save_assessments_csv(self) -> Dict[str, Dict[str, Path]]:
        """
        Save assessments in CSV format, one file per dataset per model
        
        Returns:
            Dictionary mapping model names to dictionaries of dataset -> file path
        """
        output_paths = {}
        
        for model_name, datasets in self.assessments_by_model_dataset.items():
            if not datasets:
                continue
                
            model_dir = self._create_model_directory(model_name)
            model_paths = {}
            
            for dataset, assessments in datasets.items():
                if not assessments:
                    continue
                
                filename = f"{dataset}.csv"
                output_path = model_dir / filename
                
                # Convert to DataFrame for easier CSV export
                df = pd.DataFrame(assessments)
                
                # Flatten quality_scores if it exists
                if 'quality_scores' in df.columns:
                    quality_df = pd.json_normalize(df['quality_scores'])
                    quality_df.columns = [f'quality_{col}' for col in quality_df.columns]
                    df = pd.concat([df.drop('quality_scores', axis=1), quality_df], axis=1)
                
                # Flatten justifications if it exists
                if 'justifications' in df.columns:
                    justifications_df = pd.json_normalize(df['justifications'])
                    justifications_df.columns = [f'justification_{col}' for col in justifications_df.columns]
                    df = pd.concat([df.drop('justifications', axis=1), justifications_df], axis=1)
                
                df.to_csv(output_path, index=False)
                model_paths[dataset] = output_path
                print(f"Saved {len(assessments)} assessments to {output_path}")
            
            output_paths[model_name] = model_paths
        
        return output_paths
    
    def load_assessments_from_jsonl(self, model_name: str, dataset: str) -> List[Dict]:
        """
        Load assessments from a JSONL file
        
        Args:
            model_name: Name of the model
            dataset: Dataset name
            
        Returns:
            List of assessment dictionaries
        """
        file_path = self.output_dir / model_name / f"{dataset}.jsonl"
        
        if not file_path.exists():
            return []
        
        assessments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    assessments.append(json.loads(line.strip()))
        
        return assessments
    
    def get_dataset_files(self, model_name: str) -> List[str]:
        """
        Get list of available dataset files for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of dataset names (without .jsonl extension)
        """
        model_dir = self.output_dir / model_name
        
        if not model_dir.exists():
            return []
        
        datasets = []
        for file_path in model_dir.glob("*.jsonl"):
            datasets.append(file_path.stem)
        
        return sorted(datasets)
    
    def save_session_summary(self) -> Path:
        """
        Save session-level summary with metadata and model performance
        
        Returns:
            Path to the saved summary file
        """
        # Set collection end time
        self.metadata["collection_end"] = time.time()
        
        # Create session directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Calculate model summaries
        model_summaries = {}
        
        for model_name, datasets in self.assessments_by_model_dataset.items():
            total_assessments = sum(len(assessments) for assessments in datasets.values())
            successful_assessments = 0
            total_execution_time = 0
            
            dataset_performance = {}
            for dataset, assessments in datasets.items():
                valid_assessments = [a for a in assessments if a.get('llm_call_success', True)]
                successful_assessments += len(valid_assessments)
                total_execution_time += sum(a.get('execution_time', 0) for a in assessments)
                
                dataset_performance[dataset] = {
                    "total_assessments": len(assessments),
                    "successful_assessments": len(valid_assessments),
                    "success_rate": len(valid_assessments) / max(1, len(assessments)) * 100
                }
            
            model_summaries[model_name] = {
                "model_name": model_name,
                "overview": {
                    "total_samples": total_assessments,
                    "total_assessments": total_assessments,
                    "datasets_processed": list(datasets.keys())
                },
                "overall_performance": {
                    "success_rate": successful_assessments / max(1, total_assessments) * 100,
                    "avg_execution_time": total_execution_time / max(1, total_assessments)
                },
                "dataset_performance": dataset_performance
            }
        
        summary_data = {
            "session_id": self.session_id,
            "collection_metadata": self.metadata,
            "model_summaries": model_summaries,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Session summary saved to {summary_path}")
        return summary_path
    
    def save_cross_model_comparison(self) -> Path:
        """
        Save cross-model comparison data
        
        Returns:
            Path to the saved comparison file
        """
        # Create session directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Collect all samples across models for comparison
        all_samples = {}
        models_evaluated = list(self.assessments_by_model_dataset.keys())
        datasets_processed = set()
        
        for model_name, datasets in self.assessments_by_model_dataset.items():
            for dataset, assessments in datasets.items():
                datasets_processed.add(dataset)
                for assessment in assessments:
                    sample_id = assessment.get('sample_id')
                    if sample_id not in all_samples:
                        all_samples[sample_id] = {
                            'sample_id': sample_id,
                            'problem': assessment.get('problem', ''),
                            'dataset': dataset
                        }
                    
                    # Add model-specific data
                    quality_scores = assessment.get('quality_scores', {})
                    all_samples[sample_id][f'{model_name}_functionality'] = quality_scores.get('functionality')
                    all_samples[sample_id][f'{model_name}_readability'] = quality_scores.get('readability') 
                    all_samples[sample_id][f'{model_name}_idiomatic'] = quality_scores.get('idiomatic')
                    all_samples[sample_id][f'{model_name}_error_handling'] = quality_scores.get('error_handling')
                    all_samples[sample_id][f'{model_name}_efficiency'] = quality_scores.get('efficiency')
                    all_samples[sample_id][f'{model_name}_time'] = assessment.get('execution_time')
        
        comparison_data = {
            "summary": {
                "session_id": self.session_id,
                "models_evaluated": models_evaluated,
                "datasets_processed": sorted(datasets_processed),
                "total_samples": len(all_samples)
            },
            "samples": list(all_samples.values())
        }
        
        comparison_path = self.output_dir / "cross_model_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Cross-model comparison saved to {comparison_path}")
        return comparison_path
    
    @staticmethod
    def load_session_parameters(session_id: str, output_dir: str) -> Dict:
        """
        Load session parameters from a previous session for resume functionality
        
        Args:
            session_id: Session identifier to load
            output_dir: Base output directory containing sessions
            
        Returns:
            Dictionary of session parameters or None if not found
        """
        session_dir = Path(output_dir) / session_id
        summary_path = session_dir / "summary.json"
        
        if not summary_path.exists():
            return None
        
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                return summary_data.get("collection_metadata", {}).get("session_parameters", {})
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading session parameters: {e}")
            return None
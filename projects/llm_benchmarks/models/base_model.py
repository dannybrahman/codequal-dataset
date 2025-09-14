"""Base abstract class for LLM model implementations"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import time
import asyncio

class BaseLLMModel(ABC):
    """Abstract base class for LLM model implementations"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 60, timeout: int = 30):
        """
        Initialize LLM model
        
        Args:
            model_name: Name/ID of the model
            api_key: API key for authentication
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.request_count = 0
        self.last_request_time = 0
        
    @abstractmethod
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """
        Generate solution for a given problem
        
        Args:
            problem: Problem description
            task_id: Unique identifier for the problem
            
        Returns:
            Dict containing response content, timestamp, and metadata
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Return model metadata
        
        Returns:
            Dict with model information (name, provider, version, etc.)
        """
        pass
    
    def _create_response(self, content: str, task_id: str, success: bool = True, error: str = None) -> Dict:
        """Helper method to create standardized response format"""
        return {
            "content": content,
            "task_id": task_id,
            "model": self.model_name,
            "timestamp": time.time(),
            "success": success,
            "error": error,
            "request_count": self.request_count
        }
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.rate_limit  # seconds between requests
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
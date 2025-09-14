"""Advanced rate limiting for multiple LLM providers with different limits"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional
import logging

class RateLimiter:
    """
    Advanced rate limiter that handles different rate limits per model/provider
    with sliding window approach for precise rate limiting
    """
    
    def __init__(self):
        # Track request timestamps per model in sliding windows
        self.request_windows: Dict[str, deque] = defaultdict(deque)
        
        # Model-specific rate limits (requests per minute)
        self.rate_limits: Dict[str, int] = {}
        
        # Locks for thread-safe access
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Statistics tracking
        self.stats = {
            "requests_made": defaultdict(int),
            "requests_blocked": defaultdict(int),
            "total_wait_time": defaultdict(float)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model_name: str, rate_limit: int):
        """
        Register a model with its rate limit
        
        Args:
            model_name: Unique identifier for the model
            rate_limit: Requests per minute allowed
        """
        self.rate_limits[model_name] = rate_limit
        self.logger.info(f"Registered {model_name} with rate limit {rate_limit}/min")
    
    async def acquire(self, model_name: str) -> float:
        """
        Acquire permission to make a request for the given model
        
        Args:
            model_name: Model to make request for
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        if model_name not in self.rate_limits:
            self.logger.warning(f"Model {model_name} not registered, allowing request")
            return 0.0
        
        async with self.locks[model_name]:
            wait_time = await self._calculate_wait_time(model_name)
            
            if wait_time > 0:
                self.stats["requests_blocked"][model_name] += 1
                self.stats["total_wait_time"][model_name] += wait_time
                self.logger.debug(f"Rate limiting {model_name}: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            # Record this request
            current_time = time.time()
            self.request_windows[model_name].append(current_time)
            self.stats["requests_made"][model_name] += 1
            
            return wait_time
    
    async def _calculate_wait_time(self, model_name: str) -> float:
        """Calculate how long to wait before making next request"""
        current_time = time.time()
        rate_limit = self.rate_limits[model_name]
        window = self.request_windows[model_name]
        
        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60.0
        while window and window[0] < cutoff_time:
            window.popleft()
        
        # Check if we're under the rate limit
        if len(window) < rate_limit:
            return 0.0
        
        # Calculate wait time based on oldest request in window
        oldest_request = window[0]
        wait_time = 60.0 - (current_time - oldest_request)
        
        return max(0.0, wait_time)
    
    def get_stats(self) -> Dict:
        """Get statistics about rate limiting performance"""
        stats = dict(self.stats)
        
        # Add derived statistics
        for model_name in self.rate_limits:
            requests_made = stats["requests_made"][model_name]
            requests_blocked = stats["requests_blocked"][model_name]
            total_wait = stats["total_wait_time"][model_name]
            
            stats[f"{model_name}_block_rate"] = (
                requests_blocked / max(1, requests_made + requests_blocked) * 100
            )
            stats[f"{model_name}_avg_wait"] = (
                total_wait / max(1, requests_blocked)
            )
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "requests_made": defaultdict(int),
            "requests_blocked": defaultdict(int), 
            "total_wait_time": defaultdict(float)
        }

class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts to actual API response times and errors
    to optimize throughput while respecting limits
    """
    
    def __init__(self):
        super().__init__()
        
        # Track response times to optimize batching
        self.response_times: Dict[str, deque] = defaultdict(deque)
        
        # Track error rates to back off when needed
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # Adaptive rate multipliers (1.0 = full rate, 0.5 = half rate, etc.)
        self.rate_multipliers: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def record_response(self, model_name: str, response_time: float, success: bool):
        """
        Record the result of an API call to adapt rate limiting
        
        Args:
            model_name: Model that was called
            response_time: Time taken for the request in seconds
            success: Whether the request was successful
        """
        current_time = time.time()
        
        # Record response time
        self.response_times[model_name].append((current_time, response_time))
        
        # Clean old response times (keep last 50 or last 5 minutes)
        window = self.response_times[model_name]
        cutoff_time = current_time - 300  # 5 minutes
        while window and (len(window) > 50 or window[0][0] < cutoff_time):
            window.popleft()
        
        # Record success/failure
        if success:
            self.success_counts[model_name] += 1
        else:
            self.error_counts[model_name] += 1
        
        # Adapt rate multiplier based on error rate
        self._adapt_rate_multiplier(model_name)
    
    def _adapt_rate_multiplier(self, model_name: str):
        """Adapt the rate multiplier based on recent error rates"""
        total_requests = self.success_counts[model_name] + self.error_counts[model_name]
        
        if total_requests < 10:  # Not enough data yet
            return
        
        error_rate = self.error_counts[model_name] / total_requests
        
        # Adjust rate multiplier based on error rate
        if error_rate > 0.1:  # More than 10% errors
            self.rate_multipliers[model_name] *= 0.8  # Slow down
        elif error_rate < 0.02:  # Less than 2% errors
            self.rate_multipliers[model_name] = min(1.0, self.rate_multipliers[model_name] * 1.1)  # Speed up
        
        # Keep multiplier in reasonable bounds
        self.rate_multipliers[model_name] = max(0.1, min(1.0, self.rate_multipliers[model_name]))
    
    async def _calculate_wait_time(self, model_name: str) -> float:
        """Calculate wait time with adaptive rate limiting"""
        base_wait = await super()._calculate_wait_time(model_name)
        
        # Apply rate multiplier
        multiplier = self.rate_multipliers[model_name]
        if multiplier < 1.0:
            # Slow down by increasing wait time
            additional_wait = base_wait * (1.0 / multiplier - 1.0)
            return base_wait + additional_wait
        
        return base_wait
    
    def get_adaptive_stats(self) -> Dict:
        """Get statistics including adaptive behavior"""
        stats = self.get_stats()
        
        for model_name in self.rate_limits:
            total_requests = self.success_counts[model_name] + self.error_counts[model_name]
            error_rate = self.error_counts[model_name] / max(1, total_requests)
            
            # Average response time
            times = [rt for _, rt in self.response_times[model_name]]
            avg_response_time = sum(times) / len(times) if times else 0
            
            stats[f"{model_name}_error_rate"] = error_rate * 100
            stats[f"{model_name}_rate_multiplier"] = self.rate_multipliers[model_name]
            stats[f"{model_name}_avg_response_time"] = avg_response_time
        
        return stats
"""Cohere model implementation using aiohttp"""

import asyncio
from typing import Dict
import aiohttp
from .base_model import BaseLLMModel


class CohereModel(BaseLLMModel):
    """Cohere model implementation using their API"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 10, timeout: int = 120):
        """
        Initialize Cohere model
        
        Args:
            model_name: Model name (e.g., "command-r-plus", "command-r", "command", "command-light")
            api_key: Cohere API key
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = "https://api.cohere.com/v2/chat"
        self.headers = {
            "Authorization": f"bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """
        Generate solution using Cohere API
        
        Args:
            problem: Problem description
            task_id: Unique identifier for the problem
            
        Returns:
            Dict containing response content, timestamp, and metadata
        """
        await self._enforce_rate_limit()
        
        # Format messages for chat API
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programmer. Provide complete, working solutions to programming problems. Return only the Python code without any explanation or markdown formatting."
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPlease provide a complete Python solution:"
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,
            "p": 0.95,  # Top-p sampling
            "stop_sequences": ["```"]
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract content from chat response
                        if "message" in result and "content" in result["message"]:
                            if isinstance(result["message"]["content"], list):
                                # Handle list of content blocks
                                content = ""
                                for block in result["message"]["content"]:
                                    if block.get("type") == "text":
                                        content += block.get("text", "")
                            else:
                                # Handle string content
                                content = result["message"]["content"]
                        else:
                            content = ""
                        
                        # Clean up the response
                        content = content.strip()
                        
                        # Remove markdown code blocks if present
                        import re
                        code_pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
                        matches = re.findall(code_pattern, content, re.DOTALL)
                        if matches:
                            content = matches[0].strip()
                        
                        return self._create_response(
                            content=content,
                            task_id=task_id,
                            success=True
                        )
                        
                    elif response.status == 429:
                        # Rate limit exceeded
                        retry_after = response.headers.get("X-Ratelimit-Reset", 60)
                        await asyncio.sleep(int(retry_after))
                        return await self.generate_solution(problem, task_id)
                        
                    elif response.status == 401:
                        error_msg = "Invalid API key"
                        return self._create_response(
                            content="",
                            task_id=task_id,
                            success=False,
                            error=error_msg
                        )
                        
                    else:
                        error_text = await response.text()
                        error_msg = f"API error {response.status}: {error_text}"
                        return self._create_response(
                            content="",
                            task_id=task_id,
                            success=False,
                            error=error_msg
                        )
                
        except Exception as e:
            return self._create_response(
                content="",
                task_id=task_id,
                success=False,
                error=f"Request failed: {str(e)}"
            )
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        return {
            "name": self.model_name,
            "provider": "cohere",
            "type": "api",
            "rate_limit": self.rate_limit,
            "timeout": self.timeout
        }
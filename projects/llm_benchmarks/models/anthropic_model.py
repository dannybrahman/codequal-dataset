"""Anthropic Claude model implementation"""

import asyncio
import aiohttp
import json
from typing import Dict
from .base_model import BaseLLMModel

class AnthropicModel(BaseLLMModel):
    """Anthropic Claude model implementation"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 50, timeout: int = 60):
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """Generate solution using Anthropic API"""
        await self._enforce_rate_limit()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": 2000,
            "temperature": 0.1,
            "system": "You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code.",
            "messages": [
                {
                    "role": "user",
                    "content": problem
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["content"][0]["text"].strip()
                        return self._create_response(content, task_id, success=True)
                    else:
                        error_text = await response.text()
                        return self._create_response("", task_id, success=False,
                                                   error=f"HTTP {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            return self._create_response("", task_id, success=False, error="Request timeout")
        except Exception as e:
            return self._create_response("", task_id, success=False, error=str(e))
    
    def get_model_info(self) -> Dict:
        """Return Anthropic model information"""
        return {
            "provider": "Anthropic", 
            "model_name": self.model_name,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "api_base": self.base_url
        }
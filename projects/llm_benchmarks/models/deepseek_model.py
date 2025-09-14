"""DeepSeek model implementation"""

import asyncio
import aiohttp
import json
from typing import Dict
from .base_model import BaseLLMModel

class DeepSeekModel(BaseLLMModel):
    """DeepSeek model implementation"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 20, timeout: int = 60):
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """Generate solution using DeepSeek API"""
        await self._enforce_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code."
                },
                {
                    "role": "user",
                    "content": problem
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
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
        """Return DeepSeek model information"""
        return {
            "provider": "DeepSeek",
            "model_name": self.model_name,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "api_base": self.base_url
        }
"""OpenAI model implementation (GPT-4, GPT-3.5, etc.)"""

import asyncio
import aiohttp
import json
import os
from typing import Dict
from .base_model import BaseLLMModel

class OpenAIModel(BaseLLMModel):
    """OpenAI GPT model implementation"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 50, timeout: int = 60):
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """Generate solution using OpenAI API"""
        await self._enforce_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add organization and project headers if available
        org_id = os.getenv("OPENAI_ORG_ID")
        project_id = os.getenv("OPENAI_PROJECT_ID")
        
        if org_id:
            headers["OpenAI-Organization"] = org_id
        if project_id:
            headers["OpenAI-Project"] = project_id
        
        # Determine parameters based on model type
        # Newer models (o1, o3, o4 series) have different requirements
        if any(prefix in self.model_name for prefix in ["o1-", "o3-", "o4-"]):
            token_param = "max_completion_tokens"
            # Handle temperature based on specific model series
            if "o1-" in self.model_name or "o3-" in self.model_name:
                # o1/o3 models don't support temperature parameter at all
                temperature_value = None
            elif "o4-" in self.model_name:
                # o4 models only support temperature=1 (default)
                temperature_value = 1
            else:
                temperature_value = 0.1
        else:
            token_param = "max_tokens"
            temperature_value = 0.1
        
        # Build payload
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
            token_param: 2000
        }
        
        # Add temperature only if supported
        if temperature_value is not None:
            payload["temperature"] = temperature_value
        
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
        """Return OpenAI model information"""
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "api_base": self.base_url
        }
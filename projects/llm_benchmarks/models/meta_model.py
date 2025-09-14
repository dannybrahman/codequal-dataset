"""Meta Llama model implementation (via API providers like Together, Replicate, Hugging Face, etc.)"""

import asyncio
import aiohttp
import json
from typing import Dict
from .base_model import BaseLLMModel

class MetaModel(BaseLLMModel):
    """Meta Llama model implementation via API providers"""
    
    def __init__(self, model_name: str, api_key: str, provider: str = "together", rate_limit: int = 40, timeout: int = 60):
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.provider = provider
        
        # Set base URL based on provider
        if provider == "together":
            self.base_url = "https://api.together.xyz/v1/chat/completions"
        elif provider == "replicate":
            self.base_url = "https://api.replicate.com/v1/predictions"
        elif provider == "huggingface":
            self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """Generate solution using Meta Llama via API provider"""
        await self._enforce_rate_limit()
        
        if self.provider == "together":
            return await self._generate_via_together(problem, task_id)
        elif self.provider == "replicate":
            return await self._generate_via_replicate(problem, task_id)
        elif self.provider == "huggingface":
            return await self._generate_via_huggingface(problem, task_id)
    
    async def _generate_via_together(self, problem: str, task_id: str) -> Dict:
        """Generate via Together AI"""
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
    
    async def _generate_via_huggingface(self, problem: str, task_id: str) -> Dict:
        """Generate via Hugging Face Inference API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format prompt for Llama models
        system_prompt = "You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code."
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.1,
                "max_new_tokens": 2000,
                "return_full_text": False
            }
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            content = data[0]["generated_text"].strip()
                        else:
                            content = data.get("generated_text", "").strip()
                        return self._create_response(content, task_id, success=True)
                    else:
                        error_text = await response.text()
                        return self._create_response("", task_id, success=False,
                                                   error=f"HTTP {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            return self._create_response("", task_id, success=False, error="Request timeout")
        except Exception as e:
            return self._create_response("", task_id, success=False, error=str(e))
    
    async def _generate_via_replicate(self, problem: str, task_id: str) -> Dict:
        """Generate via Replicate (implementation would depend on their specific API)"""
        # Placeholder for Replicate implementation
        return self._create_response("", task_id, success=False, 
                                   error="Replicate implementation not yet available")
    
    def get_model_info(self) -> Dict:
        """Return Meta model information"""
        return {
            "provider": f"Meta (via {self.provider})",
            "model_name": self.model_name,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "api_base": self.base_url
        }
"""Google Gemini model implementation"""

import asyncio
import aiohttp
import json
from typing import Dict
from .base_model import BaseLLMModel

class GoogleModel(BaseLLMModel):
    """Google Gemini model implementation"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 60, timeout: int = 60):
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """Generate solution using Google Gemini API"""
        await self._enforce_rate_limit()
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": f"api_key"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code.\n\nProblem: {problem}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2000
            }
        }
        
        url = f"{self.base_url}?key={self.api_key}"
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "candidates" in data and len(data["candidates"]) > 0:
                            content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            return self._create_response(content, task_id, success=True)
                        else:
                            return self._create_response("", task_id, success=False, 
                                                       error="No candidates in response")
                    else:
                        error_text = await response.text()
                        return self._create_response("", task_id, success=False,
                                                   error=f"HTTP {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            return self._create_response("", task_id, success=False, error="Request timeout")
        except Exception as e:
            return self._create_response("", task_id, success=False, error=str(e))
    
    def get_model_info(self) -> Dict:
        """Return Google model information"""
        return {
            "provider": "Google",
            "model_name": self.model_name,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "api_base": self.base_url
        }
"""HuggingFace model implementations using Router API and local transformers"""

import asyncio
import time
from typing import Dict
import aiohttp
import re
from .base_model import BaseLLMModel


class HuggingFaceRouterInferenceModel(BaseLLMModel):
    """HuggingFace model using the Router API (OpenAI-compatible)"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 30, timeout: int = 90):
        """
        Initialize HuggingFace Router API model
        
        Args:
            model_name: Model ID with optional provider (e.g., "microsoft/phi-2" or "deepseek-ai/DeepSeek-V3:novita")
            api_key: HuggingFace API token
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """
        Generate solution using HuggingFace Router API (OpenAI-compatible)
        
        Args:
            problem: Problem description
            task_id: Unique identifier for the problem
            
        Returns:
            Dict containing response content, timestamp, and metadata
        """
        await self._enforce_rate_limit()
        
        # Format as OpenAI-compatible chat messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert Python programmer. Provide complete, working solutions to programming problems."
            },
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPlease provide a complete Python solution:"
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.95,
            "stop": ["```\n\n", "\n\n\n"]
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
                        
                        # Extract content from OpenAI-compatible response
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                        else:
                            content = str(result)
                        
                        # Extract code from the response
                        code = self._extract_code_from_response(content)
                        
                        return self._create_response(
                            content=code,
                            task_id=task_id,
                            success=True
                        )
                        
                    elif response.status == 429:
                        # Rate limit - wait and retry
                        await asyncio.sleep(10)
                        return await self.generate_solution(problem, task_id)
                        
                    elif response.status == 503:
                        # Service unavailable - model might be loading
                        await asyncio.sleep(20)
                        return await self.generate_solution(problem, task_id)
                        
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
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from the model response"""
        # Look for ```python or ``` blocks
        pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, check if the entire response looks like code
        lines = response.strip().split('\n')
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__']
        
        if any(line.strip().startswith(indicator) for line in lines for indicator in code_indicators):
            return response.strip()
        
        # Otherwise return the whole response
        return response
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        # Parse provider from model name if specified
        if ":" in self.model_name:
            model_id, provider = self.model_name.split(":", 1)
        else:
            model_id = self.model_name
            provider = "auto"  # HF router auto-selects
            
        return {
            "name": self.model_name,
            "model_id": model_id,
            "provider": "huggingface_router",
            "router_provider": provider,
            "type": "router_api",
            "rate_limit": self.rate_limit,
            "timeout": self.timeout
        }


class HuggingFaceDirectInferenceModel(BaseLLMModel):
    """HuggingFace model using the direct Inference API"""
    
    def __init__(self, model_name: str, api_key: str, rate_limit: int = 30, timeout: int = 90):
        """
        Initialize HuggingFace Direct Inference API model
        
        Args:
            model_name: Model ID (e.g., "mistralai/Mistral-Nemo-Instruct-2407")
            api_key: HuggingFace API token
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, api_key, rate_limit, timeout)
        self.base_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """
        Generate solution using HuggingFace Direct Inference API
        
        Args:
            problem: Problem description
            task_id: Unique identifier for the problem
            
        Returns:
            Dict containing response content, timestamp, and metadata
        """
        await self._enforce_rate_limit()
        
        # Format the prompt for instruct models
        prompt = f"""<s>[INST] You are an expert Python programmer. Solve the following programming problem.

Problem: {problem}

Provide only the Python code solution without any explanation or markdown formatting.
[/INST]"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.95,
                "do_sample": True,
                "return_full_text": False
            }
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
                        
                        # Extract generated text
                        if isinstance(result, list) and len(result) > 0:
                            content = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            content = result.get("generated_text", "")
                        else:
                            content = str(result)
                        
                        # Clean up the response - remove any markdown if present
                        content = self._extract_code_from_response(content)
                        
                        return self._create_response(
                            content=content,
                            task_id=task_id,
                            success=True
                        )
                        
                    elif response.status == 503:
                        # Model is loading
                        error_data = await response.json()
                        wait_time = error_data.get("estimated_time", 20)
                        await asyncio.sleep(wait_time)
                        return await self.generate_solution(problem, task_id)
                        
                    elif response.status == 429:
                        # Rate limit exceeded
                        await asyncio.sleep(60)
                        return await self.generate_solution(problem, task_id)
                        
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
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from the model response"""
        # Look for ```python or ``` blocks
        pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, check if the entire response looks like code
        lines = response.strip().split('\n')
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__']
        
        if any(line.strip().startswith(indicator) for line in lines for indicator in code_indicators):
            return response.strip()
        
        # Otherwise return the whole response
        return response
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        return {
            "name": self.model_name,
            "provider": "huggingface_direct",
            "type": "direct_inference_api",
            "rate_limit": self.rate_limit,
            "timeout": self.timeout
        }


class HuggingFaceLocalModel(BaseLLMModel):
    """HuggingFace model using local transformers library"""
    
    def __init__(self, model_name: str, api_key: str = "", rate_limit: int = 60, timeout: int = 120):
        """
        Initialize local HuggingFace model
        
        Args:
            model_name: HuggingFace model ID
            api_key: Not used for local models (kept for interface compatibility)
            rate_limit: Requests per minute limit
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, api_key or "local", rate_limit, timeout)
        self.tokenizer = None
        self.model = None
        self._initialized = False
        
    async def _initialize_model(self):
        """Initialize the model and tokenizer (lazy loading)"""
        if self._initialized:
            return
            
        try:
            # Import here to avoid dependency issues if transformers not installed
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Determine the best available device
            if torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Check for Mac MPS (Metal Performance Shaders) support
                device = "mps"
                # Use float32 for MPS to avoid numerical instability issues
                torch_dtype = torch.float32
            else:
                device = "cpu"
                torch_dtype = torch.float32
            
            self.device = device
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate device settings
            if device == "cuda":
                # Use device_map for multi-GPU setups
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto"
                )
            else:
                # For MPS and CPU, load to CPU first then move to device
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(device)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self._initialized = True
            print(f"Model {self.model_name} loaded on {device}")
            
        except ImportError:
            raise ImportError("transformers library is required for local models. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {str(e)}")
    
    async def generate_solution(self, problem: str, task_id: str) -> Dict:
        """
        Generate solution using local transformers model
        
        Args:
            problem: Problem description
            task_id: Unique identifier for the problem
            
        Returns:
            Dict containing response content, timestamp, and metadata
        """
        await self._enforce_rate_limit()
        
        try:
            await self._initialize_model()
            
            prompt = self._format_prompt(problem)
            
            # Tokenize input with proper attention mask
            encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = encoded.input_ids
            attention_mask = encoded.attention_mask
            
            # Import torch here to avoid issues
            import torch
            
            # Move inputs to the same device as the model
            inputs = inputs.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            start_time = time.time()
            
            # Generate response
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs,
                        attention_mask=attention_mask,  # Pass attention mask
                        max_new_tokens=512,
                        temperature=0.7,  # Increased from 0.1 to avoid numerical issues
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        # Add repetition penalty to improve generation quality
                        repetition_penalty=1.1,
                        # Ensure we don't get NaN/Inf issues
                        bad_words_ids=None,
                        force_words_ids=None
                    )
                except RuntimeError as e:
                    if "nan" in str(e).lower() or "inf" in str(e).lower():
                        # Fallback to greedy decoding if sampling fails
                        outputs = self.model.generate(
                            inputs,
                            attention_mask=attention_mask,  # Pass attention mask
                            max_new_tokens=512,
                            do_sample=False,  # Use greedy decoding
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            generated_text = generated_text[len(prompt):].strip()
            
            return self._create_response(
                content=generated_text,
                task_id=task_id,
                success=True
            )
            
        except Exception as e:
            return self._create_response(
                content="",
                task_id=task_id,
                success=False,
                error=f"Local generation failed: {str(e)}"
            )
    
    def _format_prompt(self, problem: str) -> str:
        """Format the problem as a code generation prompt"""
        return f"""Below is a Python programming problem. Write a complete, working Python solution.

Problem: {problem}

Solution:
```python
"""
    
    def get_model_info(self) -> Dict:
        """Return model metadata"""
        info = {
            "name": self.model_name,
            "provider": "huggingface",
            "type": "local",
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "initialized": self._initialized
        }
        
        # Add device info if model is initialized
        if self._initialized and hasattr(self, 'device'):
            info["device"] = self.device
            
        return info
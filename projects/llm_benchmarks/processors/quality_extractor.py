"""Extract quality scores from LLM assessment responses"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, Optional

class QualityExtractor:
    """Extracts quality scores from LLM assessment responses using an agentic approach"""
    
    def __init__(self, extraction_api_key: str = None, extraction_model: str = "gpt-3.5-turbo"):
        """
        Initialize the quality extractor with an LLM for score extraction
        
        Args:
            extraction_api_key: API key for the extraction model (OpenAI)
            extraction_model: Model to use for extraction (fast/cheap model recommended)
        """
        self.extraction_api_key = extraction_api_key
        self.extraction_model = extraction_model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Extraction prompt template
        self.extraction_prompt = """You are a JSON extraction specialist. Your job is to extract quality scores from code assessment text.

Extract the 5 quality dimension scores (1-5 scale) and return them as valid JSON.

If the response contains scores, extract them. If not, try to infer scores from the descriptions.

TEXT TO EXTRACT FROM:
{response_text}

Return ONLY this JSON structure (no other text):
{{
    "functionality": <score 1-5>,
    "readability": <score 1-5>,
    "idiomatic": <score 1-5>,
    "error_handling": <score 1-5>,
    "efficiency": <score 1-5>
}}"""

    async def extract_quality_scores(self, response: str, sample_id: str = "") -> Dict:
        """
        Extract quality scores from an LLM assessment response
        
        Args:
            response: Raw response text from LLM
            sample_id: Optional sample ID for logging
            
        Returns:
            Dict with quality scores or None if extraction fails
        """
        if not response or not response.strip():
            return None
        
        # First try direct JSON extraction
        json_match = self._extract_json_directly(response)
        if json_match and self._validate_scores(json_match):
            return json_match
        
        # If direct extraction fails and we have an API key, use agent
        if self.extraction_api_key:
            try:
                extracted_scores = await self._extract_with_agent(response)
                if extracted_scores and self._validate_scores(extracted_scores):
                    return extracted_scores
            except Exception as e:
                print(f"Agent extraction failed for sample {sample_id}: {e}")
        
        # Final fallback: try pattern matching
        return self._extract_with_patterns(response)
    
    async def _extract_with_agent(self, response_text: str) -> Optional[Dict]:
        """Use LLM agent to extract quality scores"""
        if not self.extraction_api_key:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.extraction_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = self.extraction_prompt.format(response_text=response_text)
        
        payload = {
            "model": self.extraction_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,  # Deterministic extraction
            "max_tokens": 200
        }
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        return self._extract_json_directly(content)
                    else:
                        raise Exception(f"Extraction API error: {response.status}")
        except Exception as e:
            print(f"Agent extraction error: {e}")
            return None
    
    def _extract_json_directly(self, text: str) -> Optional[Dict]:
        """Try to extract JSON object directly from text"""
        # Look for JSON-like structure
        json_patterns = [
            r'\{[^{}]*"functionality"[^{}]*\}',  # Simple single-level JSON
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested JSON
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    # Check if it has our expected fields
                    if all(key in data for key in ["functionality", "readability", "idiomatic", "error_handling", "efficiency"]):
                        # Extract just the scores (ignore justifications if present)
                        scores = {
                            "functionality": data.get("functionality"),
                            "readability": data.get("readability"),
                            "idiomatic": data.get("idiomatic"),
                            "error_handling": data.get("error_handling"),
                            "efficiency": data.get("efficiency")
                        }
                        return scores
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_with_patterns(self, text: str) -> Optional[Dict]:
        """Fallback pattern matching for score extraction"""
        scores = {}
        
        # Pattern to match "dimension: score" or "dimension = score" formats
        patterns = [
            (r'functionality[:\s=]+(\d)', 'functionality'),
            (r'readability[:\s=]+(\d)', 'readability'),
            (r'idiomatic[:\s=]+(\d)', 'idiomatic'),
            (r'error[\s_-]?handling[:\s=]+(\d)', 'error_handling'),
            (r'efficiency[:\s=]+(\d)', 'efficiency'),
        ]
        
        for pattern, key in patterns:
            match = re.search(pattern, text.lower())
            if match:
                score = int(match.group(1))
                if 1 <= score <= 5:
                    scores[key] = score
        
        # Only return if we found all 5 dimensions
        if len(scores) == 5:
            return scores
        
        return None
    
    def _validate_scores(self, scores: Dict) -> bool:
        """Validate that scores are complete and in valid range"""
        if not scores:
            return False
        
        required_keys = ["functionality", "readability", "idiomatic", "error_handling", "efficiency"]
        
        # Check all keys present
        if not all(key in scores for key in required_keys):
            return False
        
        # Check all scores are valid integers 1-5
        for key in required_keys:
            score = scores.get(key)
            if not isinstance(score, (int, float)):
                return False
            if not (1 <= score <= 5):
                return False
        
        return True
    
    def _normalize_scores(self, scores: Dict) -> Dict:
        """Ensure scores are integers"""
        normalized = {}
        for key, value in scores.items():
            if isinstance(value, float):
                normalized[key] = int(value)
            else:
                normalized[key] = value
        return normalized
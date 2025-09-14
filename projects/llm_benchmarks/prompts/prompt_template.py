"""Standardized prompt formatting for code quality assessment"""

from typing import Dict

class PromptTemplate:
    """Handles standardized prompt formatting for code quality assessment across different LLM providers"""
    
    @staticmethod
    def get_quality_rubric() -> str:
        """Get the quality assessment rubric - matches exactly with human annotator rubric"""
        return """Quality Assessment Rubric (1-5 scale):

**Functionality**:
5 - Perfect: Solves completely and correctly with all edge cases
4 - Good: Works correctly for main cases
3 - Adequate: Works for typical cases with some limitations
2 - Poor: Significant bugs or only simple cases
1 - Broken: Doesn't work or has fundamental flaws

**Readability**:
5 - Excellent: Crystal clear with meaningful names and structure
4 - Good: Clear and well-organized
3 - Adequate: Reasonably readable
2 - Poor: Hard to understand
1 - Unreadable: Extremely difficult to understand

**Idiomatic**:
5 - Expert: Follows all best practices optimally
4 - Proficient: Generally follows conventions
3 - Acceptable: Follows basic conventions
2 - Poor: Ignores many conventions
1 - Non-idiomatic: Completely ignores conventions

**Error Handling**:
5 - Comprehensive: Handles all edge cases with informative errors
4 - Good: Handles most error cases
3 - Basic: Handles some error cases
2 - Minimal: Little error handling
1 - None: No error handling

**Efficiency**:
5 - Optimal: Best algorithm with excellent complexity
4 - Good: Efficient with reasonable performance
3 - Acceptable: Works but could be optimized
2 - Poor: Inefficient approach
1 - Very Poor: Extremely inefficient"""
    
    @staticmethod
    def format_assessment_prompt(sample_data: Dict) -> str:
        """
        Format a code sample for quality assessment
        
        Args:
            sample_data: Dict containing 'problem', 'submission', and metadata
            
        Returns:
            Formatted prompt string for quality assessment
        """
        problem = sample_data.get("problem", "")
        code = sample_data.get("submission", "")
        
        prompt = f"""Assess the quality of the following Python code solution.

**Problem Description:**
{problem}

**Code Solution:**
```python
{code}
```

{PromptTemplate.get_quality_rubric()}

Provide your assessment in the following JSON format:
{{
    "functionality": <score 1-5>,
    "readability": <score 1-5>,
    "idiomatic": <score 1-5>,
    "error_handling": <score 1-5>,
    "efficiency": <score 1-5>,
    "justification": {{
        "functionality": "<brief explanation>",
        "readability": "<brief explanation>",
        "idiomatic": "<brief explanation>",
        "error_handling": "<brief explanation>",
        "efficiency": "<brief explanation>"
    }}
}}

Return ONLY the JSON object, no additional text."""
        
        return prompt
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the standard system prompt for code quality assessment"""
        return """You are an expert code reviewer specializing in Python code quality assessment. You evaluate code across 5 dimensions: Functionality, Readability, Idiomatic Style, Error Handling, and Efficiency. You provide objective, consistent scores on a 1-5 scale with brief justifications for each dimension."""
    
    @staticmethod
    def format_for_chat_model(sample_data: Dict) -> Dict:
        """
        Format for chat-based models (OpenAI, Anthropic, etc.)
        
        Returns:
            Dict with 'system' and 'user' messages
        """
        return {
            "system": PromptTemplate.get_system_prompt(),
            "user": PromptTemplate.format_assessment_prompt(sample_data)
        }
    
    @staticmethod
    def format_for_completion_model(sample_data: Dict) -> str:
        """
        Format for completion-based models (some Hugging Face models)
        
        Returns:
            Single prompt string with system and user content combined
        """
        system_prompt = PromptTemplate.get_system_prompt()
        user_prompt = PromptTemplate.format_assessment_prompt(sample_data)
        
        return f"{system_prompt}\n\n{user_prompt}\n\nAssessment:"
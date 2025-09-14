"""System prompt variations for different use cases"""

class SystemPrompts:
    """Collection of system prompts for different scenarios"""
    
    # Standard system prompt for code generation
    STANDARD = "You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code."
    
    # More explicit version emphasizing code-only output
    STRICT_CODE_ONLY = """You are a Python programming expert. You will be given programming problems.

CRITICAL REQUIREMENTS:
- Respond with ONLY executable Python code
- NO explanations before or after the code
- NO comments in the code
- NO markdown formatting (no ```python blocks)
- NO additional text whatsoever
- Just the raw Python code that solves the problem"""
    
    # Version that emphasizes function/class naming
    EXACT_NAMING = """You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution.

REQUIREMENTS:
- Use the EXACT function/class name specified in the problem
- Return only executable Python code
- No explanations, comments, or markdown formatting
- No additional text"""
    
    # Version for models that tend to be verbose
    MINIMAL_OUTPUT = """Respond with only the Python code. No explanations. No formatting. Just code."""
    
    @classmethod
    def get_prompt_for_model(cls, model_name: str) -> str:
        """Get the most appropriate system prompt for a specific model"""
        model_lower = model_name.lower()
        
        # Some models are known to be more verbose, use stricter prompts
        if any(verbose_model in model_lower for verbose_model in ['gpt-4', 'claude', 'gemini']):
            return cls.STRICT_CODE_ONLY
        
        # For coding-specific models, use the exact naming version
        if any(coding_model in model_lower for coding_model in ['deepseek', 'codellama', 'starcoder']):
            return cls.EXACT_NAMING
            
        # Default to standard prompt
        return cls.STANDARD
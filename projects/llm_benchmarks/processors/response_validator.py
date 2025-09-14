"""Validate extracted Python code for syntax and basic structure"""

import ast
import sys
from typing import Dict, Tuple, Optional

class ResponseValidator:
    """Validates extracted Python code for syntax and structure"""
    
    @staticmethod
    def validate_python_syntax(code: str) -> Dict:
        """
        Validate if the extracted code has valid Python syntax
        
        Args:
            code: Python code string to validate
            
        Returns:
            Dict with validation results
        """
        if not code or not code.strip():
            return {
                "is_valid": False,
                "error": "Empty code",
                "error_type": "empty",
                "line_number": None
            }
        
        try:
            # Parse the code to check syntax
            ast.parse(code)
            return {
                "is_valid": True,
                "error": None,
                "error_type": None,
                "line_number": None
            }
            
        except SyntaxError as e:
            return {
                "is_valid": False,
                "error": str(e),
                "error_type": "syntax_error",
                "line_number": e.lineno
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": str(e),
                "error_type": "parse_error", 
                "line_number": None
            }
    
    @staticmethod
    def validate_function_structure(code: str, expected_name: str) -> Dict:
        """
        Validate that code contains the expected function/class name
        
        Args:
            code: Python code string
            expected_name: Expected function or class name
            
        Returns:
            Dict with structure validation results
        """
        syntax_result = ResponseValidator.validate_python_syntax(code)
        if not syntax_result["is_valid"]:
            return {
                "has_expected_name": False,
                "found_names": [],
                "syntax_valid": False,
                **syntax_result
            }
        
        try:
            tree = ast.parse(code)
            
            # Find all function and class definitions
            found_names = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    found_names.append(node.name)
            
            has_expected = expected_name in found_names
            
            return {
                "has_expected_name": has_expected,
                "found_names": found_names,
                "syntax_valid": True,
                "is_valid": has_expected,
                "error": None if has_expected else f"Expected '{expected_name}' not found. Found: {found_names}",
                "error_type": None if has_expected else "missing_expected_name"
            }
            
        except Exception as e:
            return {
                "has_expected_name": False,
                "found_names": [],
                "syntax_valid": False,
                "is_valid": False,
                "error": str(e),
                "error_type": "structure_analysis_error"
            }
    
    @staticmethod
    def validate_imports(code: str) -> Dict:
        """
        Check what modules the code imports
        
        Args:
            code: Python code string
            
        Returns:
            Dict with import information
        """
        try:
            tree = ast.parse(code)
            imports = []
            from_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        from_imports.append(f"{module}.{alias.name}")
            
            return {
                "imports": imports,
                "from_imports": from_imports,
                "has_imports": len(imports) > 0 or len(from_imports) > 0
            }
            
        except Exception as e:
            return {
                "imports": [],
                "from_imports": [],
                "has_imports": False,
                "error": str(e)
            }
    
    @staticmethod
    def comprehensive_validation(code: str, expected_name: str) -> Dict:
        """
        Run all validation checks and return comprehensive results
        
        Args:
            code: Python code string
            expected_name: Expected function/class name
            
        Returns:
            Dict with all validation results
        """
        syntax_result = ResponseValidator.validate_python_syntax(code)
        structure_result = ResponseValidator.validate_function_structure(code, expected_name)
        import_result = ResponseValidator.validate_imports(code)
        
        # Determine overall validity
        is_fully_valid = (
            syntax_result["is_valid"] and 
            structure_result["has_expected_name"]
        )
        
        return {
            "is_fully_valid": is_fully_valid,
            "syntax": syntax_result,
            "structure": structure_result,
            "imports": import_result,
            "code_length": len(code),
            "line_count": len(code.split('\n')) if code else 0
        }
    
    @staticmethod
    def get_validation_summary(validation_result: Dict) -> str:
        """
        Get a human-readable summary of validation results
        
        Args:
            validation_result: Result from comprehensive_validation
            
        Returns:
            Summary string
        """
        if validation_result["is_fully_valid"]:
            return "✅ Valid: Correct syntax and expected function/class found"
        
        issues = []
        
        if not validation_result["syntax"]["is_valid"]:
            error = validation_result["syntax"]["error"]
            line = validation_result["syntax"]["line_number"]
            issues.append(f"Syntax error{f' (line {line})' if line else ''}: {error}")
        
        if not validation_result["structure"]["has_expected_name"]:
            found = validation_result["structure"]["found_names"]
            issues.append(f"Missing expected name. Found: {found}")
        
        return "❌ Invalid: " + "; ".join(issues)
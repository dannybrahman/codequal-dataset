#!/usr/bin/env python3
"""
Qualtrics API Configuration Management

Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from .. import ROOT


class QualtricsConfig:
    """Configuration class for Qualtrics API settings."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize Qualtrics configuration.
        
        Args:
            env_file: Optional path to .env file (defaults to .env in project root)
        """
        # Load .env file if it exists
        if env_file:
            env_path = Path(env_file)
        else:
            # Use global ROOT variable from main.py
            env_path = ROOT / ".env"
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from: {env_path}")
        else:
            print(f"No .env file found at: {env_path}")
            print("Using system environment variables only")
        
        # Load configuration from environment
        self.api_token = os.getenv('QUALTRICS_API_TOKEN')
        self.base_url = os.getenv('QUALTRICS_BASE_URL')
        self.datacenter = os.getenv('QUALTRICS_DATACENTER')
        self.directory_id = os.getenv('QUALTRICS_DIRECTORY_ID')  # Optional
        
        # Validate required settings
        self.validate_config()
    
    def validate_config(self):
        """Validate that required configuration is present."""
        missing_config = []
        
        if not self.api_token:
            missing_config.append("QUALTRICS_API_TOKEN")
        
        if not self.base_url:
            missing_config.append("QUALTRICS_BASE_URL")
        
        if not self.datacenter:
            missing_config.append("QUALTRICS_DATACENTER")
        
        if missing_config:
            raise ValueError(
                f"Missing required Qualtrics configuration: {', '.join(missing_config)}\n"
                f"Please set these environment variables or add them to your .env file.\n"
                f"See .env.example for the required format."
            )
    
    def get_api_headers(self) -> dict:
        """Get HTTP headers for Qualtrics API requests."""
        return {
            'X-API-TOKEN': self.api_token,
            'Content-Type': 'application/json'
        }
    
    def get_api_base_url(self) -> str:
        """Get the base URL for API requests."""
        # Remove trailing slash if present
        base = self.base_url.rstrip('/')
        return f"{base}/API/v3"
    
    def __str__(self) -> str:
        """String representation (safe - doesn't expose token)."""
        return (
            f"QualtricsConfig("
            f"base_url='{self.base_url}', "
            f"datacenter='{self.datacenter}', "
            f"directory_id='{self.directory_id}', "
            f"token={'***' if self.api_token else 'None'}"
            f")"
        )


def load_qualtrics_config(env_file: Optional[str] = None) -> QualtricsConfig:
    """
    Load Qualtrics configuration from environment.
    
    Args:
        env_file: Optional path to .env file
        
    Returns:
        QualtricsConfig instance
        
    Raises:
        ValueError: If required configuration is missing
    """
    return QualtricsConfig(env_file)


# Example usage
if __name__ == '__main__':
    try:
        config = load_qualtrics_config()
        print("✅ Qualtrics configuration loaded successfully!")
        print(f"   {config}")
        print(f"   API Base URL: {config.get_api_base_url()}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
#!/usr/bin/env python3
"""
CodeQual Dataset Viewer - Startup Script

A simple startup script for the Flask dataset viewer application.
"""

import os
import sys
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv()

# Import and create the Flask app
from app import create_app

if __name__ == '__main__':
    app = create_app()
    
    # Get configuration from environment
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🔍 CodeQual Dataset Viewer")
    print(f"📊 Dataset: {os.getenv('DATASET_PATH', 'Not configured')}")
    print(f"🌐 Server: http://{host}:{port}")
    print("=" * 50)
    
    app.run(host=host, port=port, debug=debug)
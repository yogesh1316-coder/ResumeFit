#!/usr/bin/env python3
"""
Startup script for the AI-Powered Resume Analyzer
This ensures the Flask app starts from the correct directory
"""

import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Change to the script directory
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# Import and run the Flask app
try:
    from app import app
    print("Flask app imported successfully")
    
    if __name__ == '__main__':
        print("Starting Flask application...")
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        debug = os.environ.get('FLASK_ENV') != 'production'
        
        print(f"Starting Flask app on {host}:{port} (debug={debug})")
        app.run(host=host, port=port, debug=debug)
        
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
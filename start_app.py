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
        app.run(debug=True, host='127.0.0.1', port=5000)
        
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
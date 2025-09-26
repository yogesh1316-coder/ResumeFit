#!/usr/bin/env python
"""
WSGI entry point for production deployment
This file is used by Gunicorn to serve the Flask application
"""

import os
from app import app

# Configure production environment
os.environ['FLASK_ENV'] = 'production'

if __name__ == "__main__":
    # This is used when running locally
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # This is used by Gunicorn
    application = app
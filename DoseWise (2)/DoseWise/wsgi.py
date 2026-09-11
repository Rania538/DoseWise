"""
WSGI entrypoint for DoseWise deployment (e.g. PythonAnywhere, Gunicorn).
Exposes the Flask application object `app`.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.server import app

if __name__ == '__main__':
    app.run()

# server/app.py — OpenEnv entry point
# This file re-exports the FastAPI app from the root server module
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

__all__ = ["app"]

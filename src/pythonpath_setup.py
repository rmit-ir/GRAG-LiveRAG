"""
This module adds the src directory to Python's import path.
Import this at the beginning of scripts to allow direct imports from services.
"""
import os
import sys

# Add the src directory to the Python path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

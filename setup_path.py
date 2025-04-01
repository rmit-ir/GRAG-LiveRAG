"""
Helper module to add the src directory to Python path.
Import this at the beginning of scripts to enable direct imports from services.
"""
import os
import sys

# Get the directory containing this file (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the src directory to the path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Print for confirmation
print(f"Added to Python path: {project_root}")
print(f"Added to Python path: {src_path}")

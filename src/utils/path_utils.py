"""
Utility functions for path handling in LiveRAG.
"""
import os
from utils.logging_utils import get_logger

log = get_logger("path_utils")

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Uses the module file location to determine the project root.
    
    Returns:
        str: Absolute path to the project root directory
    """
    # Get the directory of the current module
    current_module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up to the project root (src/utils -> src -> project_root)
    project_root = os.path.dirname(os.path.dirname(current_module_dir))
    
    log.debug(f"Project root determined as: {project_root}")
    return project_root

def get_data_dir():
    """
    Returns the absolute path to the data directory.
    
    Returns:
        str: Absolute path to the data directory
    """
    data_dir = os.path.join(get_project_root(), 'data')
    log.debug(f"Data directory path: {data_dir}")
    return data_dir

def ensure_dir(file_path, create_if_not=True):
    """
    The function ensures the dir exists,
    if it doesn't it creates it and returns the path or raises FileNotFoundError
    In case file_path is an existing file, returns the path of the parent directory
    """
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # This exception was added for multiprocessing, in case multiple processes try to create the directory
                pass
        else:
            raise FileNotFoundError(
                f"The directory {directory} doesn't exist, create it or pass create_if_not=True"
            )
    return directory
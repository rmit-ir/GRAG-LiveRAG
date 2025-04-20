"""
Utilities for extracting and handling system parameters.

This module provides functions to extract parameter information from system classes,
particularly useful for command-line interfaces and configuration.
"""
import inspect
from typing import Dict, Any, Type, Optional

from utils.logging_utils import get_logger
from systems.rag_system_interface import RAGSystemInterface


logger = get_logger("system_params")


def extract_system_parameters(system_class: Type[RAGSystemInterface]) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter information from a system class's __init__ method.
    
    This function inspects the __init__ method signature and docstring to gather
    information about each parameter, including type, default value, and description.
    
    Args:
        system_class: The system class to inspect
        
    Returns:
        Dictionary mapping parameter names to their type, default value, and description
    """
    # Get the signature of the __init__ method
    signature = inspect.signature(system_class.__init__)
    
    # Get the docstring of the __init__ method
    docstring = system_class.__init__.__doc__ or ""
    
    # Parse parameter descriptions from docstring
    param_descriptions = {}
    current_param = None
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith('Args:'):
            continue
        
        # Check for parameter documentation in the format "param_name: Description"
        if ': ' in line and not line.startswith(' '):
            parts = line.split(': ', 1)
            if len(parts) == 2:
                current_param = parts[0].strip()
                param_descriptions[current_param] = parts[1].strip()
        # Continue description for the current parameter
        elif current_param and line:
            param_descriptions[current_param] += ' ' + line
    
    # Extract parameter information
    params = {}
    for name, param in signature.parameters.items():
        # Skip 'self' parameter
        if name == 'self':
            continue
        
        # Get parameter type annotation
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
        
        # Get default value
        default = param.default if param.default != inspect.Parameter.empty else None
        
        # Get description from docstring or use parameter name if no description is available
        description = param_descriptions.get(name, f"Parameter '{name}'")
        
        params[name] = {
            'type': param_type,
            'default': default,
            'description': description
        }
    
    logger.debug("Extracted system parameters", 
                system_class=system_class.__name__,
                param_count=len(params))
    
    return params


def get_system_params_from_args(system_class: Type[RAGSystemInterface], args) -> Dict[str, Any]:
    """
    Extract system parameters from command-line arguments.
    
    Args:
        system_class: The system class to extract parameters for
        args: The parsed command-line arguments
        
    Returns:
        Dictionary of parameter name to value mappings
    """
    system_params = {}
    for name in extract_system_parameters(system_class).keys():
        # Check if the parameter exists in args
        if hasattr(args, name):
            value = getattr(args, name)
            # Only include non-None values to allow defaults to be used
            if value is not None:
                system_params[name] = value
    
    logger.debug("Extracted system parameters from args", 
                system_class=system_class.__name__,
                param_count=len(system_params))
    
    return system_params

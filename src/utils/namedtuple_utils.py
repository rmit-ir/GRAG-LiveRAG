"""
Utility functions for working with NamedTuples.

This module provides helper functions for common operations with NamedTuples,
such as creating updated copies of immutable NamedTuple instances.
"""
from typing import TypeVar, NamedTuple

from utils.logging_utils import get_logger

logger = get_logger("namedtuple_utils")

T = TypeVar('T', bound=NamedTuple)


def update_tuple(instance: T, **kwargs) -> T:
    """
    Create a new NamedTuple instance with updated fields.
    
    This utility function allows creating a new instance with some fields updated
    while keeping the rest of the fields the same. Works with any NamedTuple type.
    
    Args:
        instance: Original NamedTuple instance
        **kwargs: Fields to update in the new instance
        
    Returns:
        A new NamedTuple instance with updated fields
    """
    # Convert the original instance to a dictionary
    instance_dict = instance._asdict()
    
    # Update the dictionary with the provided kwargs
    instance_dict.update(kwargs)
    
    # Create and return a new instance of the same type
    return type(instance)(**instance_dict)

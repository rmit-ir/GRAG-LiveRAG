"""
Utility functions for implementing retry logic with exponential backoff.
"""
import time
import math
import random
from functools import wraps
from typing import Callable
from utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("retry_utils")


def retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    jitter: bool = True,
    retry_on: tuple = None
):
    """
    Decorator that retries a function with exponential backoff when specific exceptions occur.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay in seconds
        max_delay (float): Maximum delay in seconds
        jitter (bool): Whether to add random jitter to the delay
        retry_on (tuple): Tuple of exception classes to retry on. If None, retries on all exceptions.
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            # Get the logger from the first argument (self) if it's a method
            func_logger = args[0].logger if args and hasattr(args[0], 'logger') else logger
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if we should retry this exception type
                    if retry_on and not isinstance(e, retry_on):
                        func_logger.error(f"Non-retryable error: {str(e)}")
                        raise
                    
                    # Don't retry if we've reached max retries
                    if retries >= max_retries:
                        func_logger.error(f"Max retries ({max_retries}) reached. Giving up.", error=str(e))
                        raise
                    
                    # Don't retry for certain error types based on string content
                    error_str = str(e).lower()
                    if ("auth" in error_str or 
                        "unauthorized" in error_str or 
                        "permission" in error_str or
                        "access denied" in error_str or
                        "validation" in error_str):
                        func_logger.error(f"Authentication/validation error. Not retrying.", error=str(e))
                        raise
                    
                    # Calculate delay using logarithmic curve: base_delay * log(2^(retries+1))
                    # This gives a gentler curve than pure exponential backoff
                    delay = min(base_delay * math.log(2 ** (retries + 1)), max_delay)
                    
                    # Add jitter if enabled (±20% randomness)
                    if jitter:
                        delay = delay * (0.8 + 0.4 * random.random())
                    
                    # Round to 2 decimal places for cleaner logs
                    delay = round(delay, 2)
                    
                    retries += 1
                    func_logger.warning(
                        f"API call failed. Retrying in {delay}s (attempt {retries}/{max_retries})",
                        error=str(e),
                        retry_attempt=retries,
                        max_retries=max_retries,
                        delay_seconds=delay
                    )
                    time.sleep(delay)
        return wrapper
    return decorator

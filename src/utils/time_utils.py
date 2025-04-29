"""
Utility functions for time-related operations.
"""
from datetime import timedelta
from format_duration import format_duration


def format_time_ms(ms: float) -> str:
    """
    Format milliseconds into a human-readable string with appropriate units.
    
    Args:
        ms: Time in milliseconds
        
    Returns:
        Formatted time string (e.g., "50ms", "45s", "5min", etc.)
    """
    # Convert milliseconds to timedelta
    duration = timedelta(milliseconds=ms)
    
    # Format the duration with abbreviated units
    return format_duration(duration, is_abbreviated=True)

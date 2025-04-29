"""
Utility functions for time-related operations.
"""


def format_time_ms(ms: float) -> str:
    """
    Format milliseconds into a human-readable string with appropriate units.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted time string (e.g., "50ms", "45s", "5min", etc.)
    """
    ms = int(ms)
    if ms < 0:
        return ms
    d_ms = ms % 1000
    d_s = (ms // 1000) % 60
    d_m = (ms // 60000) % 60
    d_h = (ms // 3600000) % 24
    d_d = (ms // 86400000) % 30
    d_mo = (ms // 2592000000) % 12
    d_y = ms // 31536000000

    time_str = ""
    if d_y > 1:
        time_str += f"{d_y} years "
    if d_mo > 1:
        time_str += f"{d_mo} months "
    if d_d > 1:
        time_str += f"{d_d} days "
    if d_h > 1:
        time_str += f"{d_h} hours "
    if d_m > 1:
        time_str += f"{d_m}min "
    if d_s > 1:
        time_str += f"{d_s}s "
    if d_ms > 1:
        time_str += f"{d_ms}ms"

    return time_str.strip()


def to_sec(ms: float) -> float:
    """
    Convert milliseconds to seconds but with 3 decimal places.

    Args:
        ms: Time in milliseconds

    Returns:
        Time in seconds
    """
    return round(ms / 1000, 3)

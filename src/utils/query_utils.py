"""
Utility functions for query processing.
"""
import re
import time
import random
from typing import Optional, List
from sqids import Sqids

from utils.logging_utils import get_logger

logger = get_logger("query_utils")

# Initialize Sqids with default settings
sqids = Sqids(min_length=6)

def generate_query_id(question: str, existing_id: Optional[str] = None, max_words: int = 3) -> str:
    """
    Generates a unique query ID based on the question text and a short unique identifier.
    
    Args:
        question: The question text
        existing_id: An existing ID to use if provided
        max_words: Maximum number of words to include from the question
        
    Returns:
        A unique query ID
    """
    if existing_id:
        return existing_id
    
    # Return early if question is empty
    if not question:
        return generate_short_id()
    
    # Sanitize the question text (remove special characters, lowercase)
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', question.lower())
    
    # Take the first few words
    words = sanitized.split()[:max_words]
    prefix = '_'.join(words)
    
    # Generate a short unique ID
    short_id = generate_short_id()
    
    # Combine prefix and short ID
    query_id = f"{prefix}_{short_id}"
    
    logger.debug("Generated query ID", question=question, query_id=query_id)
    return query_id

def generate_short_id() -> str:
    """
    Generates a short unique ID using Sqids.
    
    Returns:
        A short unique ID
    """
    # Use current timestamp (milliseconds) as a number for Sqids
    timestamp = int(time.time() * 1000)
    
    # Add some randomness to ensure uniqueness
    random_num = random.randint(0, 999999)
    
    # Encode the numbers to get a short ID
    short_id = sqids.encode([timestamp, random_num])
    
    return short_id

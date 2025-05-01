"""
Test script for AI71 API using direct httpx requests to match curl behavior.
"""
import os
import httpx
import json
from dotenv import load_dotenv
from utils.logging_utils import get_logger

# Set up logger
logger = get_logger("test_ai71_client")

def test_direct_request():
    """Test AI71 API using direct httpx request to match curl behavior."""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("AI71_API_KEY")
    if not api_key:
        logger.error("AI71_API_KEY environment variable not set")
        return
    
    # API endpoint
    url = "https://api.ai71.ai/v1/completions"
    
    # Request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Request payload - note the \n\n at the end of the prompt
    payload = {
        "model": "tiiuae/falcon3-10b-instruct",
        "prompt": "Hello, my name is \n\n",
        "max_tokens": 1024,
        "temperature": 0.0
    }
    
    logger.info("Sending direct httpx request to AI71 API")
    
    try:
        # Send the request
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Print the response
        logger.info("Response received successfully")
        print("\nResponse from AI71 API (direct httpx):")
        print("-" * 50)
        print(result["choices"][0]["text"])
        print("-" * 50)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in direct request: {str(e)}")
        if hasattr(e, "response") and e.response:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response headers: {e.response.headers}")
            logger.error(f"Response content: {e.response.text}")
        raise

def test_modified_openai_client():
    """Test AI71 API using modified OpenAI client."""
    from openai import OpenAI
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("AI71_API_KEY")
    if not api_key:
        logger.error("AI71_API_KEY environment variable not set")
        return
    
    # Initialize the OpenAI client with default headers
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ai71.ai/v1/",
        max_retries=5,
        timeout=30.0,
        default_headers={
            "Content-Type": "application/json",
        }
    )
    
    logger.info("Sending request via modified OpenAI client")
    
    try:
        # Send the prompt to the completion model
        # Note the \n\n at the end of the prompt
        response = client.completions.create(
            model="tiiuae/falcon3-10b-instruct",
            prompt="Hello, my name is \n\n",
            temperature=0.0,
            max_tokens=1024
        )
        
        # Extract content from the response
        content = response.choices[0].text
        
        # Print the response
        logger.info("Response received successfully")
        print("\nResponse from AI71 API (OpenAI client):")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        return content
        
    except Exception as e:
        logger.error(f"Error in OpenAI client request: {str(e)}")
        raise

if __name__ == "__main__":
    print("Testing direct httpx request...")
    try:
        test_direct_request()
    except Exception as e:
        print(f"Direct request failed: {str(e)}")
    
    print("\nTesting modified OpenAI client...")
    try:
        test_modified_openai_client()
    except Exception as e:
        print(f"Modified OpenAI client failed: {str(e)}")

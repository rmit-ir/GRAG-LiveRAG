"""
Client for interacting with AI71 API using LangChain.
"""
import os
import json
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir

# Initialize logger
logger = get_logger("ai71_client")


def get_llm_response(
    prompt: str,
    api_key: str,
    model: str = "gpt-4",
) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    Sends an evaluation prompt to the AI71 API using LangChain.
    Saves the raw response to a directory.

    Args:
        prompt (str): The formatted prompt containing query, response, and content of retrieved documents.
        api_key (str): API key for authenticating the AI71 API request.
        model (str): The LLM model to use for evaluation (default: "gpt-4").

    Returns:
        Tuple(raw_response, content, reasoning): 
            - raw_response: The complete API response as a dictionary
            - content: The generated text content from the model
            - reasoning: The reasoning behind the model's evaluation (if available)
    """
    # Start timing the API request
    start_time = time.time()

    try:
        # Initialize ChatOpenAI with the AI71 API URL
        chat_model = ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base="https://api.ai71.ai/v1/"
        )

        logger.debug(f"Sending request to AI71 API with model: {model}")

        # Send the message and get the response
        response = chat_model.invoke(prompt)

        # Calculate API response time
        response_time = time.time() - start_time

        # Extract the raw response data
        # Note: LangChain might not expose the full raw response, so we create a structured dict
        raw_response = {
            "model": model,
            "prompt": prompt,
            "response": response.content,
            "timestamp": datetime.now().isoformat()
        }

        # Save the raw response
        save_raw_response(raw_response, prompt, model)

        # Extract content from the response
        content = response.content

        # Try to extract reasoning if available
        # Note: This depends on the model's response structure
        reasoning = None
        if hasattr(response, "additional_kwargs") and response.additional_kwargs:
            reasoning = response.additional_kwargs.get("reasoning")

        # Log response time
        logger.info(
            "AI71 API request completed",
            response_time_ms=round(response_time * 1000)
        )

        # Try to log token usage if available
        if hasattr(response, "usage") and response.usage:
            logger.info(
                "Token usage",
                prompt_tokens=response.usage.get("prompt_tokens", "N/A"),
                completion_tokens=response.usage.get(
                    "completion_tokens", "N/A"),
                total_tokens=response.usage.get("total_tokens", "N/A")
            )

        return raw_response, content, reasoning

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


def save_raw_response(response: Dict[str, Any], prompt: str, model: str) -> None:
    """
    Saves the raw API response to a file for reproducibility and backup.

    Args:
        response (Dict[str, Any]): The raw API response
        prompt (str): The prompt that was sent to the API
        model (str): The model that was used
    """
    try:
        # Create a directory for raw responses if it doesn't exist
        raw_responses_dir = os.path.join(get_data_dir(), "raw_responses")
        os.makedirs(raw_responses_dir, exist_ok=True)

        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.replace(".", "-").replace(":", "-")
        filename = f"ai71_response_{model_name}_{timestamp}.json"
        filepath = os.path.join(raw_responses_dir, filename)

        # Save the response with the prompt
        with open(filepath, "w") as f:
            json.dump({
                "prompt": prompt,
                "model": model,
                "timestamp": timestamp,
                "response": response
            }, f, indent=2)

        logger.debug(f"Raw response saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save raw response: {str(e)}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable
    api_key = os.getenv("AI71_API_KEY")
    
    if not api_key:
        logger.error("AI71_API_KEY environment variable not set")
        exit(1)
    
    # Example prompt
    prompt = "What is retrieval-augmented generation (RAG)?"
    
    try:
        # Call the function
        raw_response, content, reasoning = get_llm_response(
            prompt=prompt,
            api_key=api_key
        )
        
        # Print the response
        print("\nResponse from AI71 API:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        if reasoning:
            print("\nReasoning:")
            print("-" * 50)
            print(reasoning)
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

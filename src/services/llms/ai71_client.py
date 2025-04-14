"""
Client for interacting with AI71 API using LangChain.
"""
import os
import json
import time
from typing import Dict, Tuple, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir

# Initialize logger
logger = get_logger("ai71_client")


class AI71Client:
    """Client for interacting with AI71 API."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str = None
    ):
        """
        Initialize the AI71 client.

        Args:
            model_id (str): The model ID to use
            system_message (str): System message to use for all queries
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            api_key (str, optional): AI71 API key. If None, uses AI71_API_KEY from env
        """
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.environ.get("AI71_API_KEY", "")
            if not api_key:
                logger.error("AI71_API_KEY not provided and not found in environment variables")
                raise ValueError("AI71 API key is required")

        # Initialize ChatOpenAI with the AI71 API URL
        self.chat_model = ChatOpenAI(
            model_name=model_id,
            openai_api_key=api_key,
            openai_api_base="https://api.ai71.ai/v1/",
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.model_id = model_id
        self.system_message = system_message
        logger.debug(f"Initialized AI71 client with model: {model_id}")

    def query(self, prompt: str) -> Tuple[Any, str]:
        """
        Send a query to the AI71 API and get the response.

        Args:
            prompt (str): The prompt to send to the API

        Returns:
            Tuple[Any, str]: A tuple containing:
                - raw_response: The complete API response object
                - content: The generated text content from the model
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Format messages with system message and user prompt
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]

            # Send the message and get the response
            response = self.chat_model.invoke(messages)
            
            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "AI71 API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Extract content from the response
            content = response.content

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.get("prompt_tokens", "N/A"),
                    completion_tokens=response.usage.get("completion_tokens", "N/A"),
                    total_tokens=response.usage.get("total_tokens", "N/A")
                )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": prompt,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata, prompt)

            return response, content

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _save_raw_response(self, response: Dict[str, Any], prompt: str) -> None:
        """
        Saves the raw API response to a file for reproducibility and backup.

        Args:
            response (Dict[str, Any]): The raw API response
            prompt (str): The prompt that was sent to the API
        """
        try:
            # Create a directory for raw responses if it doesn't exist
            raw_responses_dir = os.path.join(get_data_dir(), "raw_responses")
            os.makedirs(raw_responses_dir, exist_ok=True)

            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_id.replace(".", "-").replace(":", "-").replace("/", "-")
            filename = f"ai71_response_{model_name}_{timestamp}.json"
            filepath = os.path.join(raw_responses_dir, filename)

            # Save the response with the prompt
            with open(filepath, "w") as f:
                json.dump({
                    "prompt": prompt,
                    "model": self.model_id,
                    "timestamp": timestamp,
                    "response": response
                }, f, indent=2)

            logger.debug(f"Raw response saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw response: {str(e)}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Check if required environment variables are set
    if not os.environ.get("AI71_API_KEY"):
        logger.error("AI71_API_KEY environment variable not set")
        exit(1)

    # Create an AI71Client instance
    client = AI71Client(
        model_id="tiiuae/falcon3-10b-instruct",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Send the query and get the response
    raw_response, content = client.query("What is retrieval-augmented generation (RAG)?")

    # Print the response content
    print("\nResponse from AI71 API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

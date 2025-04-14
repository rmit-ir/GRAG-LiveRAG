"""
Client for interacting with Amazon Bedrock API using LangChain.
"""
import os
import json
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from langchain_aws import ChatBedrock
from botocore.exceptions import BotoCoreError, ClientError
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir

# Initialize logger
logger = get_logger("bedrock_client")


class BedrockClient:
    """Client for interacting with Amazon Bedrock API."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        region_name: str = None
    ):
        """
        Initialize the Bedrock client.

        Args:
            model_id (str): The model ID to use
            system_message (str): System message to use for all queries
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
        """
        # Set AWS credentials from environment variables with RACE_ prefix
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
            "RACE_AWS_ACCESS_KEY_ID", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
            "RACE_AWS_SECRET_ACCESS_KEY", "")
        os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
            "RACE_AWS_SESSION_TOKEN", "")

        # Use provided region_name or get from environment variable
        if region_name is None:
            region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")

        # Initialize ChatBedrock with the specified model
        self.chat_model = ChatBedrock(
            model_id=model_id,
            region_name=region_name,
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        self.model_id = model_id
        self.system_message = system_message
        logger.debug(f"Initialized Bedrock client with model: {model_id}")

    def query(self, prompt: str) -> str:
        """
        Send a query to the Bedrock API and get the response.

        Args:
            prompt (str): The prompt to send to the API

        Returns:
            str: The generated text content from the model
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
                "Bedrock API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Save response for reproducibility
            self._save_raw_response({
                "model": self.model_id,
                "prompt": prompt,
                "response": response.content,
                "timestamp": datetime.now().isoformat()
            }, prompt)

            return response.content

        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS error: {str(e)}")
            raise
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
            model_name = self.model_id.replace(".", "-").replace(":", "-")
            filename = f"bedrock_response_{model_name}_{timestamp}.json"
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
    if not os.environ.get("RACE_AWS_ACCESS_KEY_ID") or not os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
        logger.error(
            "RACE_AWS_ACCESS_KEY_ID or RACE_AWS_SECRET_ACCESS_KEY environment variables not set")
        exit(1)

    if not os.environ.get("RACE_AWS_REGION"):
        logger.warning(
            "RACE_AWS_REGION environment variable not set, using default: us-west-2")

    # Example prompt
    prompt = "What is retrieval-augmented generation (RAG)?"

    try:
        # Create a BedrockClient instance
        client = BedrockClient(
            model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
            system_message="You are an AI assistant that provides clear, concise explanations."
        )

        # Send the query and get the response
        response = client.query(prompt)

        # Print the response
        print("\nResponse from Bedrock API:")
        print("-" * 50)
        print(response)
        print("-" * 50)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

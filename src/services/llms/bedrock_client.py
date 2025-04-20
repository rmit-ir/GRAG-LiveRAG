"""
Client for interacting with Amazon Bedrock API using LangChain.
"""
import os
import json
import time
from typing import Dict, Tuple, Any, Optional
from datetime import datetime
from langchain_aws import ChatBedrock
from botocore.exceptions import BotoCoreError, ClientError
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir

# Initialize logger
logger = get_logger("bedrock_client")

# Pricing per 1000 tokens for different models (in USD)
# These prices should be updated as AWS changes their pricing
MODEL_PRICING = {
    # Claude 3.5 models
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_price": 1.00,  # $1.00 per 1M input tokens ($0.001 per 1K)
        "output_price": 5.00   # $5.00 per 1M output tokens ($0.005 per 1K)
    },
    # Claude 3 models
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_price": 0.25,  # $0.25 per 1M input tokens ($0.00025 per 1K)
        "output_price": 1.25   # $1.25 per 1M output tokens ($0.00125 per 1K)
    },
    # Default pricing if model not found
    "default": {
        "input_price": 3.00,  # Default to Claude 3.5 Sonnet pricing
        "output_price": 15.00
    }
}


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

        # Set up model kwargs with temperature and max_tokens
        model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Initialize ChatBedrock with the specified model
        self.chat_model = ChatBedrock(
            model_id=model_id,
            region_name=region_name,
            model_kwargs=model_kwargs
        )
        self.model_id = model_id
        self.system_message = system_message
        logger.debug(f"Initialized Bedrock client with model: {model_id}")

    def query(self, prompt: str) -> Tuple[Any, str]:
        """
        Send a query to the Bedrock API and get the response.

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
            logger.debug(f"Raw response from Bedrock API", response=response)
            
            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "Bedrock API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Extract content from the response
            content = response.content
            
            # Extract token usage from response
            token_usage = self._extract_token_usage(response)
            
            # Calculate cost based on token usage
            cost = self._calculate_cost(token_usage)
            
            # Log token usage and cost
            logger.info(
                "Token usage and cost",
                input_tokens=token_usage.get("input_tokens", 0),
                output_tokens=token_usage.get("output_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
                cost_usd=cost
            )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": prompt,
                "response": content,
                "timestamp": datetime.now().isoformat(),
                "token_usage": token_usage,
                "cost_usd": cost
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata, prompt)

            # Add token usage and cost to response object for access by callers
            response.token_usage = token_usage
            response.cost_usd = cost
            
            return response, content

        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS error: {str(e)}")
            # Check if this is an expired token exception
            if isinstance(e, ClientError) and "ExpiredTokenException" in str(e):
                logger.warning("AWS token has expired. Please refresh your AWS credentials (environment variables).")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage information from the API response.
        
        Args:
            response: The API response object
            
        Returns:
            Dictionary containing token usage information
        """
        token_usage = {}
        
        # Try to extract from response_metadata
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        
        # If not found, try additional_kwargs
        elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
            usage = response.additional_kwargs.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        
        # If not found, try usage_metadata
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0)
            }
        
        return token_usage
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API call based on token usage.
        
        Args:
            token_usage: Dictionary containing token usage information
            
        Returns:
            Cost in USD
        """
        # Get pricing for the model
        pricing = MODEL_PRICING.get(self.model_id, MODEL_PRICING["default"])
        
        # Extract token counts
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        
        # Calculate cost (price per 1M tokens, convert to price per token)
        input_cost = (input_tokens / 1000) * (pricing["input_price"] / 1000)
        output_cost = (output_tokens / 1000) * (pricing["output_price"] / 1000)
        
        # Total cost
        total_cost = input_cost + output_cost
        
        return total_cost
    
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

    # Create a BedrockClient instance
    client = BedrockClient(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Send the query and get the response
    raw_response, content = client.query("What is retrieval-augmented generation (RAG)?")

    # Print the response content
    print("\nResponse from Bedrock API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

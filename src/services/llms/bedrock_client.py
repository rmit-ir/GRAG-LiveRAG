"""
Client for interacting with Amazon Bedrock API using LangChain.
"""
import os
import json
import time
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage
from botocore.exceptions import BotoCoreError, ClientError
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir
from services.llms.llm_interface import LLMInterface

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


class BedrockClient(LLMInterface):
    """Client for interacting with Amazon Bedrock API."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        region_name: str = None
    ):
        """
        Initialize the Bedrock client.

        Args:
            model_id (str): The model ID to use
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
        """
        # Initialize the parent class
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Get AWS credentials from environment variables with RACE_ prefix
        aws_access_key_id = os.environ.get("RACE_AWS_ACCESS_KEY_ID", "")
        aws_secret_access_key = os.environ.get(
            "RACE_AWS_SECRET_ACCESS_KEY", "")
        aws_session_token = os.environ.get("RACE_AWS_SESSION_TOKEN", "")

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError(
                "AWS credentials (RACE_AWS_ACCESS_KEY_ID and RACE_AWS_SECRET_ACCESS_KEY) are required for Bedrock API access.")

        # Use provided region_name or get from environment variable
        if region_name is None:
            region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")

        # Set up model kwargs with temperature and max_tokens
        model_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Initialize ChatBedrock with the specified model and credentials
        self.chat_model = ChatBedrock(
            model_id=model_id,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            model_kwargs=model_kwargs
        )

        # Store model ID for reference
        self.model_id = model_id
        logger.debug(f"Initialized Bedrock client with model: {model_id}")

    def complete(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The text prompt to complete

        Returns:
            str: The generated text content from the model

        Raises:
            NotImplementedError: This method is not implemented for Claude v3 models
        """
        raise NotImplementedError(
            "The complete method is not implemented for Claude v3 models. "
            "Please use complete_chat_once instead."
        )

    def complete_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, AIMessage]:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, 
                each containing 'role' (system, user, or assistant) and 'content' keys

        Returns:
            Tuple[str, AIMessage]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        # Start timing the API request
        start_time = time.time()

        try:
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
                "messages": messages,
                "response": content,
                "timestamp": datetime.now().isoformat(),
                "token_usage": token_usage,
                "cost_usd": cost
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)

            # Add token usage and cost to response object for access by callers
            response.token_usage = token_usage
            response.cost_usd = cost

            return content, response

        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS error: {str(e)}")
            # Check if this is an expired token exception
            if isinstance(e, ClientError) and "ExpiredTokenException" in str(e):
                logger.warning(
                    "AWS token has expired. Please refresh your AWS credentials (environment variables).")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in complete_chat: {str(e)}")
            raise

    def complete_chat_once(self, message: str, system_message: Optional[str] = None) -> Tuple[str, AIMessage]:
        """
        Generate a response for a chat conversation with a single call.

        Args:
            message (str): A single prompt message
            system_message (Optional[str]): System message to use for this conversation.
                If None, uses a default system message.

        Returns:
            Tuple[str, AIMessage]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        # Use provided system message or default to a standard assistant message
        system_message = system_message or "You are an AI assistant that provides clear, concise explanations."

        # Format messages with system message and user prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]

        # Use complete_chat to handle the request
        return self.complete_chat(messages)

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

    def _save_raw_response(self, response: Dict[str, Any]) -> None:
        """
        Saves the raw API response to a file for reproducibility and backup.

        Args:
            response (Dict[str, Any]): The raw API response
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

            # Save the response
            with open(filepath, "w") as f:
                json.dump({
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
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0"
    )

    # Send the query and get the response with a custom system message
    content, raw_response = client.complete_chat_once(
        "What is retrieval-augmented generation (RAG)?",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Print the response content
    print("\nResponse from Bedrock API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

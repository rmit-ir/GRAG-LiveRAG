"""
Client for interacting with OpenAI-compatible API using LangChain.
"""
import os
import json
import time
import requests
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime
from services.llms.llm_interface import LLMInterface
from langchain_openai import ChatOpenAI, OpenAI
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir
from langchain_core.messages import AIMessage

# Initialize logger
logger = get_logger("general_openai_client")


class GeneralOpenAIClient(LLMInterface):
    """Client for interacting with OpenAI-compatible API."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: str = None,
        api_base: str = None,
    ):
        """
        Initialize the OpenAI-compatible client.

        Args:
            model_id (str): The model ID to use
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            api_key (str, optional): API key. If None, uses EC2_LLM_API_KEY from env
            api_base (str, optional): API base URL. If None, uses http://localhost:8987/v1/
        """
        # Initialize the parent class
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.environ.get("EC2_LLM_API_KEY", "")
            if not api_key:
                logger.warning(
                    "EC2_LLM_API_KEY not found in environment variables, using empty string")

        # Set default API base URL if not provided
        if api_base is None:
            api_base = "http://localhost:8987/v1/"

        # Check model health
        self._check_model_health(api_base, api_key)

        # Initialize ChatOpenAI with the API URL
        self.chat_model = ChatOpenAI(
            model_name=model_id,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Initialize OpenAI for non-chat completions
        self.completion_model = OpenAI(
            model_name=model_id,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Store model ID for reference
        self.model_id = model_id
        logger.debug(
            f"Initialized OpenAI-compatible client with model: {model_id}")

    def _check_model_health(self, api_base: str, api_key: str) -> None:
        """
        Check if the model is available by querying the health endpoint.

        Args:
            api_base (str): API base URL
            api_key (str): API key for authentication

        Raises:
            ValueError: If the model health check fails
        """
        health_url = api_base.replace("/v1/", "/health")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            logger.info(f"Checking model health at {health_url}")
            response = requests.get(health_url, headers=headers, timeout=5)
            if response.status_code != 200:
                raise ValueError(f"Model health check failed with status code {response.status_code}. "
                                 "Please launch the model using 'uv run scripts/aws/deploy_ec2_llm.py'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to model: {str(e)}")
            raise ValueError("Failed to connect to the model. "
                             "Please launch the model using 'uv run scripts/aws/deploy_ec2_llm.py'") from e

    def complete(self, prompt: str) -> str:
        """
        Generate a text completion for the given prompt.
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the prompt to the completion model
            content = self.completion_model.invoke(prompt)

            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "Completion API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": prompt,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)
            logger.debug("Response content", content=content)

            return content

        except Exception as e:
            logger.error(f"Unexpected error in complete: {str(e)}")
            raise

    def complete_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, AIMessage]:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, 
                each containing 'role' (system, user, or assistant) and 'content' keys

        Returns:
            Tuple[str, Any]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the message and get the response
            response = self.chat_model.invoke(messages)

            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Extract content from the response
            content = response.content

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.get("prompt_tokens", "N/A"),
                    completion_tokens=response.usage.get(
                        "completion_tokens", "N/A"),
                    total_tokens=response.usage.get("total_tokens", "N/A")
                )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": messages,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)
            logger.debug("Response content", content=content)

            return content, response

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
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

    def _save_raw_response(self, response: Dict[str, Any]) -> None:
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
            model_name = self.model_id.replace(
                ".", "-").replace(":", "-").replace("/", "-")
            filename = f"openai_response_{model_name}_{timestamp}.json"
            filepath = os.path.join(raw_responses_dir, filename)

            # Save the response with the prompt
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

    # Create a GeneralOpenAIClient instance
    client = GeneralOpenAIClient(
        model_id="tiiuae/falcon3-10b-instruct"
    )

    # Send the query and get the response with a custom system message
    content, raw_response = client.complete_chat_once(
        "What is retrieval-augmented generation (RAG)?",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Print the response content
    print("\nResponse from API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

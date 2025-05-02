"""
Client for interacting with OpenAI-compatible API using the official OpenAI Python client.
"""
import logging
import os
import json
import time
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime
from services.llms.llm_interface import LLMInterface
from openai import OpenAI
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir


class GeneralOpenAIClient(LLMInterface):
    """Client for interacting with OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        max_retries: int = 5,
        timeout: float = 60.0,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        logger: logging.Logger = get_logger("general_openai_client"),
        llm_name: str = "general_openai_client"
    ):
        """
        Initialize the OpenAI-compatible client.

        Args:
            model_id (str): The model ID to use
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            api_key (str): API key (required)
            api_base (str): API base URL (required)
            logger (logging.Logger): Logger instance
            llm_name (str): Name of the LLM client for file naming
        """
        # Initialize the parent class
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Validate required parameters
        if not api_key:
            raise ValueError("API key is required")

        if not api_base:
            raise ValueError("API base URL is required")

        self.logger = logger
        self.llm_name = llm_name

        # Initialize the OpenAI client with explicit headers
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            max_retries=max_retries,
            timeout=timeout,
            default_headers={
                "Content-Type": "application/json",
            }
        )

        # Store model ID for reference
        self.model_id = model_id
        self.logger.debug(
            f"Initialized OpenAI-compatible client with model: {model_id}")

    def complete(self, prompt: str) -> str:
        """
        Generate a text completion for the given prompt.
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the prompt to the completion model
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt+"\n\n",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract content from the response
            content = response.choices[0].text

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "Completion API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
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
            self.logger.debug("Response content", content=content)

            return content

        except Exception as e:
            self.logger.error(f"Unexpected error in complete: {str(e)}")
            raise

    def complete_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
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
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract content from the response
            content = response.choices[0].message.content

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
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
            self.logger.debug("Response content", content=content)

            return content, response

        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    def complete_chat_once(self, message: str, system_message: Optional[str] = None) -> Tuple[str, Any]:
        """
        Generate a response for a chat conversation with a single call.

        Args:
            message (str): A single prompt message
            system_message (Optional[str]): System message to use for this conversation.
                If None, uses a default system message.

        Returns:
            Tuple[str, Any]: A tuple containing:
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
            filename = f"{self.llm_name}_response_{model_name}_{timestamp}.json"
            filepath = os.path.join(raw_responses_dir, filename)

            # Save the response with the prompt
            with open(filepath, "w") as f:
                json.dump({
                    "model": self.model_id,
                    "timestamp": timestamp,
                    "response": response
                }, f, indent=2)

            self.logger.debug(f"Raw response saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save raw response: {str(e)}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("AI71_API_KEY", "")

    # Create a GeneralOpenAIClient instance
    client = GeneralOpenAIClient(
        api_key=api_key,
        api_base="https://api.ai71.ai/v1/",
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

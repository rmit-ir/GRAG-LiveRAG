"""
Client for interacting with Mini-TGI API server deployed through EC2.
"""
import os
import json
import time
import requests
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime
from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir

# Initialize logger
logger = get_logger("mini_tgi_client")


class MiniTGIClient:
    """Client for interacting with Mini-TGI API server."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str = None,
        api_base: str = None
    ):
        """
        Initialize the Mini-TGI client.

        Args:
            model_id (str): The model ID to use
            system_message (str): System message to use for all queries
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            api_key (str, optional): API key. If None, uses EC2_LLM_API_KEY from env
            api_base (str, optional): API base URL. If None, uses http://localhost:8977/
        """
        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.environ.get("EC2_LLM_API_KEY", "")
            if not api_key:
                logger.warning("EC2_LLM_API_KEY not found in environment variables, using empty string")
        
        # Set default API base URL if not provided
        if api_base is None:
            api_base = "http://localhost:8977/"
        
        # Remove trailing slash if present
        if api_base.endswith("/"):
            api_base = api_base[:-1]
            
        self.api_base = api_base
        self.api_key = api_key
        self.model_id = model_id
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check model health
        self._check_model_health()
        
        logger.debug(f"Initialized Mini-TGI client with model: {model_id}")

    def _check_model_health(self) -> None:
        """
        Check if the model is available by querying the health endpoint.
            
        Raises:
            ValueError: If the model health check fails
        """
        health_url = f"{self.api_base}/health"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            logger.info(f"Checking model health at {health_url}")
            response = requests.get(health_url, headers=headers, timeout=5)
            if response.status_code != 200:
                raise ValueError(f"Model health check failed with status code {response.status_code}. "
                                "Please launch the model using 'uv run scripts/aws/deploy_ec2_llm.py --app-type mini-tgi'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to model: {str(e)}")
            raise ValueError("Failed to connect to the model. "
                            "Please launch the model using 'uv run scripts/aws/deploy_ec2_llm.py --app-type mini-tgi'") from e

    def query(self, prompt: str) -> Tuple[Any, str]:
        """
        Send a query to the API and get the response.

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
            response = self._generate_text(messages=messages)
            
            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "API request completed",
                response_time_ms=round(response_time * 1000)
            )

            # Extract content from the response
            content = response.get("text", "")

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": prompt,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata, prompt)
            logger.debug("Response content", content=content)

            return response, content

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    def _generate_text(self, messages: List[Dict[str, str]] = None, prompt: str = None) -> Dict[str, Any]:
        """
        Generate text using the Mini-TGI API.

        Args:
            messages (List[Dict[str, str]], optional): List of message dictionaries with role and content
            prompt (str, optional): Direct prompt to use (if messages not provided)

        Returns:
            Dict[str, Any]: The API response as a dictionary

        Raises:
            ValueError: If neither messages nor prompt is provided
        """
        if not messages and not prompt:
            raise ValueError("Either messages or prompt must be provided")

        url = f"{self.api_base}/generate"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mini_tgi_client/0.1.0"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare request data
        data = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        if messages:
            data["messages"] = messages
        else:
            data["prompt"] = prompt

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            logger.error(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def get_token_logits(self, prompt: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Get logits for specific tokens given a prompt.

        Args:
            prompt (str): The prompt to send to the API
            tokens (List[str]): List of tokens to get logits for

        Returns:
            Dict[str, Any]: Dictionary containing logits, probabilities, and next token prediction
        """
        url = f"{self.api_base}/logits"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mini_tgi_client/0.1.0"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "prompt": prompt,
            "tokens": tokens
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            logger.error(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
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
            filename = f"mini_tgi_response_{model_name}_{timestamp}.json"
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

    # Create a MiniTGIClient instance
    client = MiniTGIClient(
        model_id="tiiuae/falcon3-10b-instruct",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Send the query and get the response
    raw_response, content = client.query("What is retrieval-augmented generation (RAG)?")

    # Print the response content
    print("\nResponse from API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

    # Test token logits
    logits_response = client.get_token_logits(
        prompt="User: What is the capital of France?\nAssistant: ",
        tokens=["Paris", "London", "Berlin", "Rome"]
    )

    print("\nToken Logits:")
    print("-" * 50)
    print(f"Logits: {logits_response.get('logits', {})}")
    print(f"Probabilities: {logits_response.get('probabilities', {})}")
    print(f"Next token: {logits_response.get('next_token', '')}")
    print("-" * 50)

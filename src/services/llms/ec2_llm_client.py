"""
Client for interacting with EC2-hosted LLM API using LangChain.
"""
import os
import requests
from utils.logging_utils import get_logger
from services.llms.general_openai_client import GeneralOpenAIClient


class EC2LLMClient(GeneralOpenAIClient):
    """Client for interacting with EC2-hosted LLM API."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        # Set up logger
        logger = get_logger("ec2_llm_client")
        
        # Set API base and key
        self.api_base = "http://localhost:8987/v1/"
        self.api_key = os.environ.get("EC2_LLM_API_KEY", "")
        if not self.api_key:
            raise ValueError("EC2 LLM API key is required")
        
        # Check model health before initializing
        self._check_model_health(self.api_base, self.api_key, logger)
            
        # Initialize the parent class (GeneralOpenAIClient)
        super().__init__(
            api_key=self.api_key,
            api_base=self.api_base,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            logger=logger,
            timeout=60.0,
            llm_name="ec2_llm"
        )
    
    def _check_model_health(self, api_base: str, api_key: str, logger) -> None:
        """
        Check if the model is available by querying the health endpoint.

        Args:
            api_base (str): API base URL
            api_key (str): API key for authentication
            logger: Logger instance

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


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Create an EC2LLMClient instance
    client = EC2LLMClient(
        model_id="tiiuae/falcon3-10b-instruct"
    )

    # Send the query and get the response with a custom system message
    content, raw_response = client.complete_chat_once(
        "What is retrieval-augmented generation (RAG)?",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Print the response content
    print("\nResponse from EC2 LLM API:")
    print("-" * 50)
    print(content)
    print("-" * 50)

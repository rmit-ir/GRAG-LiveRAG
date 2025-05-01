"""
Client for interacting with AI71 API using LangChain.
"""
import os
from utils.logging_utils import get_logger
from services.llms.general_openai_client import GeneralOpenAIClient


class AI71Client(GeneralOpenAIClient):
    """Client for interacting with AI71 API."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        # Set up logger
        logger = get_logger("ai71_client")
        
        # Set API base and key
        self.api_base = "https://api.ai71.ai/v1/"
        self.api_key = os.environ.get("AI71_API_KEY", "")
        if not self.api_key:
            raise ValueError("AI71 API key is required")
            
        # Initialize the parent class (GeneralOpenAIClient)
        super().__init__(
            api_key=self.api_key,
            api_base=self.api_base,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            logger=logger,
            llm_name="ai71"
        )



if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Check if required environment variables are set
    if not os.environ.get("AI71_API_KEY"):
        print("AI71_API_KEY environment variable not set")
        exit(1)

    # Create an AI71Client instance
    client = AI71Client(
        model_id="tiiuae/falcon3-10b-instruct"
    )

    # Send the query and get the response with a custom system message
    content, raw_response = client.complete_chat_once(
        "What is retrieval-augmented generation (RAG)?",
        system_message="You are an AI assistant that provides clear, concise explanations."
    )

    # Print the response content
    print("\nResponse from AI71 API:")
    print("-" * 50)
    print(content)
    print("-" * 50)
    
    print(client.complete("Hello, how are you?"))

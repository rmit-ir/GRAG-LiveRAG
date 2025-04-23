from dotenv import load_dotenv
import os

from services.llms.ai71_client import AI71Client
from utils.logging_utils import get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("ai71_client")

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
raw_response, content = client.query(
    "What is retrieval-augmented generation (RAG)?")

# Print the response content
print("\nResponse from AI71 API:")
print("-" * 50)
print(content)
print("-" * 50)
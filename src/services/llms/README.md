# LLM Services

This directory contains client implementations for various LLM providers. Each client provides a consistent interface for interacting with different LLM services.

## Usage

```python
from services.llms.xx_client import XxClient

# Initialize the client
client = XxClient(
    model_id="model_id",
    system_message="You are an AI assistant that provides clear, concise explanations.",
    temperature=0.7,
    max_tokens=1024
)

# Send a query and get the response
raw_response, content = client.query("What is retrieval-augmented generation (RAG)?")

# Print the response content
print(content)

# Access token usage and cost information
print(f"Input tokens: {raw_response.token_usage['input_tokens']}")
print(f"Output tokens: {raw_response.token_usage['output_tokens']}")
# For Bedrock
print(f"Cost: ${raw_response.cost_usd:.6f}")
```

## Available LLM Clients

### AI71Client

**Purpose**: Client for interacting with AI71 API using LangChain.

**Available Models**: To find available models, visit the [AI71 website](https://ai71.ai/) or refer to their API documentation. Common models include:

- `tiiuae/falcon3-10b-instruct`

### BedrockClient

**Purpose**: Client for interacting with Amazon Bedrock API using LangChain.

**Available Models**: To find available models in Amazon Bedrock, visit <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

**Note**: This client automatically calculates and logs token usage and cost information for each request.

### GeneralOpenAIClient

**Purpose**: Client for interacting with OpenAI-compatible APIs using LangChain.

**Available Models**: This client works with any model that supports the OpenAI API format.

### MiniTGIClient

**Purpose**: Client for connecting to local TGI (Text Generation Interface) servers.

**Usage**: This client is designed to work with the mini TGI server that can be launched using:

```bash
uv run scripts/aws/apps/mini_tgi/llm_server.py --port 8977
```

### SageMakerClient

**Purpose**: Client for deploying and interacting with models on Amazon SageMaker.

**Available Models**: This client can deploy any model from Hugging Face. To find available models, visit the [Hugging Face Model Hub](https://huggingface.co/models).

**Note**: This client handles the full lifecycle of SageMaker resources, including deployment and cleanup. Make sure to call `cleanup_resources()` when you're done to avoid unnecessary AWS charges.

### EC2LLMClient

**Purpose**: Client for interacting with LLM models deployed on EC2 instances.

## Environment Variables

The LLM clients use the following environment variables:

- `AI71_API_KEY`: API key for the AI71Client
- `RACE_AWS_ACCESS_KEY_ID`: AWS access key ID for Bedrock and SageMaker clients
- `RACE_AWS_SECRET_ACCESS_KEY`: AWS secret access key for Bedrock and SageMaker clients
- `RACE_AWS_SESSION_TOKEN`: AWS session token for Bedrock and SageMaker clients (optional)
- `RACE_AWS_REGION`: AWS region for Bedrock and SageMaker clients (default: us-west-2)

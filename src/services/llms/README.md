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

### GeneralOpenAIClient

**Purpose**: Client for interacting with OpenAI-compatible APIs using LangChain.

**Available Models**: This client works with any model that supports the OpenAI API format. By default, it connects to a locally running vLLM server at `http://localhost:8987/v1/`. You can deploy a HuggingFace compatible model using the `uv run scripts/aws/deploy_ec2_llm.py` script.

To list available models, you can query the `/v1/models` endpoint using cURL:

```bash
curl -X GET "http://localhost:8987/v1/models" \
  -H "Authorization: Bearer $EC2_LLM_API_KEY" \
  -H "Content-Type: application/json"
```

### AI71Client

**Purpose**: Client for interacting with AI71 API using LangChain.

**Available Models**: To find available models, visit the [AI71 website](https://ai71.ai/) or refer to their API documentation. Common models include:

- `tiiuae/falcon3-10b-instruct`

### BedrockClient

**Purpose**: Client for interacting with Amazon Bedrock API using LangChain.

**Available Models**: To find available models in Amazon Bedrock, visit <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>

**Note**: This client automatically calculates and logs token usage and cost information for each request.

### SageMakerClient

Deprecated: use GeneralOpenAIClient with `uv run scripts/aws/deploy_ec2_llm.py` instead.

**Purpose**: Client for deploying and interacting with models on Amazon SageMaker.

**Available Models**: This client can deploy any model from Hugging Face. To find available models, visit the [Hugging Face Model Hub](https://huggingface.co/models). Common models include:

**Note**: This client handles the full lifecycle of SageMaker resources, including deployment and cleanup. Make sure to call `cleanup_resources()` when you're done to avoid unnecessary AWS charges.

## Environment Variables

The LLM clients use the following environment variables:

- `EC2_LLM_API_KEY`: API key for the GeneralOpenAIClient
- `AI71_API_KEY`: API key for the AI71Client
- `RACE_AWS_ACCESS_KEY_ID`: AWS access key ID for Bedrock and SageMaker clients
- `RACE_AWS_SECRET_ACCESS_KEY`: AWS secret access key for Bedrock and SageMaker clients
- `RACE_AWS_SESSION_TOKEN`: AWS session token for Bedrock and SageMaker clients (optional)
- `RACE_AWS_REGION`: AWS region for Bedrock and SageMaker clients (default: us-west-2)

## Deploying a Local LLM

To deploy a local LLM using vLLM on EC2, use the `uv run scripts/aws/deploy_ec2_llm.py` script:

```bash
uv run scripts/aws/deploy_ec2_llm.py --model-id tiiuae/Falcon3-10B-Instruct
```

This will:

1. Create an EC2 instance with the necessary GPU resources (specify with `--instance-type`)
2. Install vLLM and its dependencies
3. Download and run the specified model
4. Set up port forwarding to access the model locally

Once deployed, you can use the GeneralOpenAIClient to interact with the model.

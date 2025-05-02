# AWS

Check your deployed CloudFormation stacks at: <https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks>

## Deploy EC2 LLM

The `deploy_ec2_llm.py` script allows you to deploy a HuggingFace compatible model on an EC2 instance using CloudFormation. It now supports two application types:

1. **vLLM** - High-performance inference server with OpenAI API compatibility
   - Deploys an EC2 instance with the specified model using vLLM
   - Sets up port forwarding to access the model locally at <http://localhost:8987/v1/> (port customizable)
   - Provides OpenAI API compatible endpoints for easy integration

2. **mini-TGI** - Lightweight Text Generation Inference server with logits support
   - Deploys an EC2 instance with the specified model using a custom TGI implementation
   - Sets up port forwarding to access the model locally at <http://localhost:8977/> (port customizable)
   - Provides both text generation and token logits endpoints

### Basic Usage

Setup credentials:

1. Get aws secret keys at <https://rmit-research.awsapps.com/start/#/>, select an account and click on "Access keys"
2. Copy secrets from "Option 3"
3. Edit .env and fill in `RACE_AWS_ACCESS_KEY_ID`, `RACE_AWS_SECRET_ACCESS_KEY`, `RACE_AWS_SESSION_TOKEN`. Leave `RACE_AWS_REGION` as `us-west-2` (default).

Setup a requirement:

```bash
# Install AWS CLI plugin, used to ssh into the EC2 instance and set up port forwarding
brew install awscli
brew install --cask session-manager-plugin
```

Deploy default model with vLLM (Ctrl+C to stop and destroy all resources):

```bash
uv run scripts/aws/deploy_ec2_llm.py
```

Deploy with mini-TGI:

```bash
uv run scripts/aws/deploy_ec2_llm.py --app-name mini-tgi
```

At this point, you can access the model at:

- vLLM: <http://localhost:8987/v1/> - Try `uv run src/services/llms/general_openai_client.py` to request the model
- mini-TGI: <http://localhost:8977/> - Try `uv run src/services/llms/mini_tgi_client.py` to request the model

> [!IMPORTANT]  
> Leaving the GPU resources running can incur big costs, so make sure to stop it on time.
>
> Actively check at <https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks>
> to see if there are any llm_xxx stacks running, in case they are yours and not deleted or being deleted, delete them.

Extra commands:

```bash
# Deploy a particular model
uv run scripts/aws/deploy_ec2_llm.py --model-id tiiuae/falcon3-10b-instruct

# Deploy with specific app name and parameters
uv run scripts/aws/deploy_ec2_llm.py --app-name mini-tgi --param MAX_BATCH_SIZE=64

# Stop and destroy all resources
uv run scripts/aws/deploy_ec2_llm.py --stop
```

### Application Types

#### vLLM

The vLLM application provides a high-performance inference server with OpenAI API compatibility. It's ideal for applications that need to integrate with the OpenAI API format.

Parameters:

- `MODEL_ID`: Hugging Face model ID (default: "tiiuae/falcon3-10b-instruct")
- `TENSOR_PARALLEL`: Number of GPUs for tensor parallelism (0 for auto)
- `MAX_NUM_BATCHED_TOKENS`: Maximum number of tokens to batch
- `GPU_MEMORY_UTILIZATION`: GPU memory utilization (0.0-1.0)
- `ENABLE_CHUNKED_PREFILL`: Enable chunked prefill (true/false)

Example:

```bash
uv run scripts/aws/deploy_ec2_llm.py --app-name vllm --param MODEL_ID=meta-llama/Llama-2-7b-chat-hf --param TENSOR_PARALLEL=2
```

#### mini-TGI

The mini-TGI application is a lightweight Text Generation Inference server that provides both text generation and token logits endpoints. It's particularly useful for applications that need access to token probabilities.

Parameters:

- `MODEL_ID`: Hugging Face model ID (default: "tiiuae/falcon3-10b-instruct")
- `MAX_BATCH_SIZE`: Maximum batch size (default: 8)

Example:

```bash
uv run scripts/aws/deploy_ec2_llm.py --app-name mini-tgi --param MODEL_ID=tiiuae/falcon3-10b-instruct --param MAX_BATCH_SIZE=64
```

### Client Libraries

Two client libraries are available to interact with the deployed models:

1. **general_openai_client.py** - For interacting with vLLM deployments using the OpenAI API format
2. **mini_tgi_client.py** - For interacting with mini-TGI deployments, supporting both text generation and token logits

## Suggested Workflow

When working with the EC2 LLM, you can use the following workflow to efficiently manage your resources:

```bash
# Start the model (specify app-name if needed)
uv run scripts/aws/deploy_ec2_llm.py --app-name vllm  # or mini-tgi

# Run your task with automatic notification and cleanup
uv run scripts/aws/deploy_ec2_llm.py --wait; say "The EC2 LLM is ready, starting my tasks"; run_your_task; uv run scripts/aws/deploy_ec2_llm.py --stop
```

This workflow:

1. Deploys the EC2 LLM instance with your chosen application type
2. Waits for the instance to be fully ready
3. Runs your tasks
4. Automatically stops and cleans up resources after your tasks are complete

## Manage The EC2 Instance

### Session Manager

To connect to the EC2 instance, we use AWS Session Manager. This allows us to connect to the instance without needing SSH access or opening any ports.

1. Get access keys from [AWS access portal](https://rmit-research.awsapps.com/start/#/?tab=accounts)
2. Set AWS environment variables (copy, paste, and execute)
3. Check if identity is correct `aws sts get-caller-identity`
4. Check all EC2 instances `aws ec2 describe-instances --region us-west-2 --query "Reservations[].Instances[].{ID:InstanceId,Name:Tags[?Key=='Name']|[0].Value,State:State.Name,Type:InstanceType}" --output table`
5. Check all Session Manager instances `aws ssm describe-instance-information --region us-west-2`
6. Check instance status `aws ec2 describe-instance-status --instance-id i-0623b265ce7c3aae9 --region us-west-2`
7. Start session `aws ssm start-session --target i-0623b265ce7c3aae9 --region us-west-2`
   - If you see error "SessionManagerPlugin is not found", install it via `brew install --cask session-manager-plugin`
8. Common commands
   1. `ip addr` - check IP address
   2. `sudo su - ubuntu` - switch to ubuntu user

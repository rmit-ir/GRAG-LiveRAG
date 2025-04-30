# AWS

Check your deployed CloudFormation stacks at: <https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks>

## Deploy EC2 LLM

The `deploy_ec2_llm.py` script allows you to deploy a HuggingFace compatible model on an EC2 instance using CloudFormation. It:

- Deploys an EC2 instance with the specified model using vLLM
- Sets up port forwarding to access the model locally at <http://localhost:8987/v1/> (port customizable)
- Provides OpenAI API compatible endpoints for easy integration

### Basic Usage

Setup credentials:

1. Get aws secret keys at <https://rmit-research.awsapps.com/start/#/>, select an account and click on "Access keys"
2. Copy secrets from "Option 3"
3. Edit .env and fill in `RACE_AWS_ACCESS_KEY_ID`, `RACE_AWS_SECRET_ACCESS_KEY`, `RACE_AWS_SESSION_TOKEN`. Leave `RACE_AWS_REGION` as `us-west-2` (default).

Setup a requirement:

```bash
# Install AWS CLI plugin, used to ssh into the EC2 instance and set up port forwarding
brew install --cask session-manager-plugin
```

Deploy default model (Ctrl+C to stop and destroy all resources)

```bash
uv run scripts/aws/deploy_ec2_llm.py
```

At this point, you can access the model at <http://localhost:8987/v1/>. Try `uv run src/services/llms/general_openai_client.py` to request the model.

> [!IMPORTANT]  
> Leaving the GPU resources running can incur big costs, so make sure to stop it on time.
>
> Actively check at <https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks>
> to see if there are any llm_xxx stacks running, in case they are yours and not deleted or being deleted, delete them.

Extra commands:

```bash
# Deploy a particular model
uv run scripts/aws/deploy_ec2_llm.py --model-id tiiuae/falcon3-10b-instruct

# Stop and destroy all resources
uv run scripts/aws/deploy_ec2_llm.py --stop
```

## Suggested Workflow

When working with the EC2 LLM, you can use the following workflow to efficiently manage your resources:

```bash
# Start the model
uv run scripts/aws/deploy_ec2_llm.py

# Run your task with automatic notification and cleanup
uv run scripts/aws/deploy_ec2_llm.py --wait; say "The EC2 LLM is ready, starting my tasks"; run_your_task; uv run scripts/aws/deploy_ec2_llm.py --stop
```

This workflow:

1. Deploys the EC2 LLM instance
2. Waits for the instance to be fully ready
3. Run your tasks
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

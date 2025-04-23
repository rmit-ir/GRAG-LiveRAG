# AWS

## Deploy EC2 LLM

The `deploy_ec2_llm.py` script allows you to deploy a HuggingFace compatible model on an EC2 instance using CloudFormation. It:

- Deploys an EC2 instance with the specified model using vLLM
- Sets up port forwarding to access the model locally at http://localhost:8987/v1/
- Provides OpenAI API compatible endpoints for easy integration

### Basic Usage

```bash
# Deploy a model (Ctrl+C to stop and destroy all resources)
uv run scripts/aws/deploy_ec2_llm.py --model-id tiiuae/Falcon3-10B-Instruct

# Stop and destroy all resources
uv run scripts/aws/deploy_ec2_llm.py --stop
```

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

"""
Client for deploying and interacting with models on Amazon SageMaker.
"""
import os
import time
import json
from typing import Dict, Any, Optional, Tuple, List

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig
from botocore.exceptions import BotoCoreError, ClientError
from langchain_core.messages import AIMessage

from utils.logging_utils import get_logger
from services.llms.llm_interface import LLMInterface

# Initialize logger
logger = get_logger("sagemaker_client")


class SageMakerClient(LLMInterface):
    """Client for deploying and interacting with models on Amazon SageMaker."""

    def __init__(
        self,
        model_id: str = "tiiuae/falcon-3-10b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        region_name: str = None,
        instance_type: str = "ml.g6.12xlarge",
    ):
        """
        Initialize the SageMaker client.

        Args:
            model_id (str): The Hugging Face model ID to deploy
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
            instance_type (str): SageMaker instance type for deployment
        """
        # Initialize the parent class
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Get AWS credentials from environment variables with RACE_ prefix
        access_key = os.environ.get("RACE_AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get("RACE_AWS_SESSION_TOKEN", "")
        
        # Use provided region_name or get from environment variable
        if region_name is None:
            region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")
        
        # Set up SageMaker session with explicit credentials
        self.boto_session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region_name
        )
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.region_name = self.sagemaker_session.boto_region_name
        
        # Store configuration
        self.model_id = model_id
        self.instance_type = instance_type
        
        # Track deployed resources
        self.endpoint_name = None
        self.model_name = None
        
        logger.debug(f"Initialized SageMaker client with model: {model_id}")

    def _get_sagemaker_execution_role(self) -> str:
        """
        Get a SageMaker execution role.
        
        This method tries multiple approaches:
        1. Check for SAGEMAKER_ROLE_ARN environment variable
        2. Get the execution role from the SageMaker environment
        3. Get a role from the SageMaker domain/user profile
        4. Construct a default role ARN if needed
        
        Returns:
            str: ARN of the SageMaker execution role
        """
        # First, check if SAGEMAKER_ROLE_ARN is set in environment variables
        role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
        if role_arn:
            logger.info("Using SageMaker role ARN from environment: %s", role_arn)
            return role_arn
            
        try:
            # Try to get the execution role from the SageMaker environment
            role = sagemaker.get_execution_role()
            logger.info("Using SageMaker execution role: %s", role)
            return role
        except ValueError:
            # If not in SageMaker environment, try to get a default role
            sm_client = self.boto_session.client('sagemaker')
            
            try:
                # List SageMaker domains to find execution roles
                domains = sm_client.list_domains()
                if domains['Domains']:
                    domain_id = domains['Domains'][0]['DomainId']
                    
                    # Get default user profile
                    user_profiles = sm_client.list_user_profiles(DomainIdEquals=domain_id)
                    if user_profiles['UserProfiles']:
                        user_profile_name = user_profiles['UserProfiles'][0]['UserProfileName']
                        
                        # Get user profile details which contains the execution role
                        user_profile = sm_client.describe_user_profile(
                            DomainId=domain_id,
                            UserProfileName=user_profile_name
                        )
                        
                        if 'ExecutionRole' in user_profile['UserSettings']:
                            role = user_profile['UserSettings']['ExecutionRole']
                            logger.info("Using execution role from SageMaker domain: %s", role)
                            return role
                
                # If we couldn't get a role from the domain, try to construct a default one
                account_id = self.boto_session.client('sts').get_caller_identity()['Account']
                role = f"arn:aws:iam::{account_id}:role/service-role/AmazonSageMaker-ExecutionRole-{int(time.time())}"
                logger.warning("Using constructed default role: %s", role)
                return role
                
            except Exception as e:
                logger.error("Failed to get SageMaker execution role: %s", str(e))
                raise ValueError(
                    "Failed to get SageMaker execution role. Please create a SageMaker execution role "
                    "or set the SAGEMAKER_ROLE_ARN environment variable."
                )
    
    def create_deployment(
        self,
        endpoint_name: Optional[str] = None,
        instance_count: int = 1,
        transformers_version: str = "4.37.0",
        pytorch_version: str = "2.1.0",
        py_version: str = "py310",
        model_server_workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Deploy a Hugging Face model to SageMaker.
        
        Args:
            endpoint_name: Custom endpoint name (optional)
            instance_count: Number of instances to deploy
            transformers_version: Version of transformers library
            pytorch_version: Version of PyTorch
            py_version: Python version
            model_server_workers: Number of model server workers
            
        Returns:
            Dict containing endpoint details
        """
        try:
            # Get IAM role for SageMaker
            role = self._get_sagemaker_execution_role()
            logger.info("Using IAM role: %s", role)
            
            # Generate unique names if not provided
            timestamp = int(time.time())
            if not endpoint_name:
                endpoint_name = f"{self.model_id.split('/')[-1]}-endpoint-{timestamp}"
            
            model_name = f"{self.model_id.split('/')[-1]}-model-{timestamp}"
            
            # Configure Hugging Face model
            hub = {
                'HF_MODEL_ID': self.model_id,
                'HF_TASK': 'text-generation'
            }
            
            # Create Hugging Face Model
            huggingface_model = HuggingFaceModel(
                env=hub,
                role=role,
                model_data=None,  # No custom model data, using Hugging Face Hub
                transformers_version=transformers_version,
                pytorch_version=pytorch_version,
                py_version=py_version,
                model_server_workers=model_server_workers,
                name=model_name,
                sagemaker_session=self.sagemaker_session
            )
            
            logger.info("Deploying model %s to endpoint %s on instance type %s...", 
                        self.model_id, endpoint_name, self.instance_type)
            
            # Deploy the model
            predictor = huggingface_model.deploy(
                initial_instance_count=instance_count,
                instance_type=self.instance_type,
                endpoint_name=endpoint_name,
            )
            
            logger.info("Model deployed successfully to endpoint: %s", endpoint_name)
            
            # Store the resources for later cleanup
            self.endpoint_name = endpoint_name
            self.model_name = model_name
            
            return {
                "endpoint_name": endpoint_name,
                "model_name": model_name,
                "instance_type": self.instance_type,
                "model_id": self.model_id,
                "region": self.region_name
            }
        
        except Exception as e:
            logger.error("Error in deployment: %s", str(e))
            raise

    def cleanup_resources(self) -> None:
        """Clean up all SageMaker resources to avoid costs."""
        if self.endpoint_name:
            try:
                logger.info("Cleaning up endpoint: %s", self.endpoint_name)
                self.sagemaker_session.delete_endpoint(self.endpoint_name)
                self.sagemaker_session.delete_endpoint_config(self.endpoint_name)
                logger.info("Endpoint deleted successfully")
                self.endpoint_name = None
            except Exception as e:
                logger.error("Error deleting endpoint: %s", str(e))
        
        if self.model_name:
            try:
                logger.info("Cleaning up model: %s", self.model_name)
                self.sagemaker_session.delete_model(self.model_name)
                logger.info("Model deleted successfully")
                self.model_name = None
            except Exception as e:
                logger.error("Error deleting model: %s", str(e))

    def complete(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The text prompt to complete

        Returns:
            str: The generated text content from the model
        """
        # Default parameters
        parameters = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True,
        }
        
        # Use the existing query method but extract only the content
        _, content = self._invoke_endpoint(prompt, parameters)
        return content
    
    def complete_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, AIMessage]:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, 
                each containing 'role' (system, user, or assistant) and 'content' keys

        Returns:
            Tuple[str, AIMessage]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object as an AIMessage
        """
        # Convert messages to a prompt format that SageMaker can understand
        prompt = self._format_messages_as_prompt(messages)
        
        # Default parameters
        parameters = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True,
        }
        
        # Invoke the endpoint
        result, content = self._invoke_endpoint(prompt, parameters)
        
        # Create an AIMessage from the response
        ai_message = AIMessage(content=content)
        
        return content, ai_message
    
    def complete_chat_once(self, message: str, system_message: Optional[str] = None) -> Tuple[str, AIMessage]:
        """
        Generate a response for a chat conversation with a single call.

        Args:
            message (str): A single prompt message
            system_message (Optional[str]): System message to use for this conversation.
                If None, uses a default system message.

        Returns:
            Tuple[str, AIMessage]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object as an AIMessage
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
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a list of messages as a single prompt string.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Formatted prompt string
        """
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            else:
                formatted_messages.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_messages) + "\nAssistant:"
    
    def _invoke_endpoint(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Send a query to the SageMaker endpoint and get the response.

        Args:
            prompt (str): The prompt to send to the API
            parameters (Dict[str, Any], optional): Generation parameters

        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - raw_response: The complete API response
                - content: The generated text content from the model
        """
        if not self.endpoint_name:
            raise ValueError("No active endpoint. Please create a deployment first.")
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True,
            }
        
        # Prepare payload
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        # Start timing the API request
        start_time = time.time()
        
        try:
            # Create runtime client
            runtime_client = self.boto_session.client("sagemaker-runtime")
            
            # Send request to endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            
            # Parse response
            result = json.loads(response["Body"].read().decode())
            
            # Log response time
            response_time = time.time() - start_time
            logger.info(
                "SageMaker API request completed",
                response_time=round(response_time, 3)
            )
            
            # Extract content from the response
            content = ""
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    content = result[0]["generated_text"]
                else:
                    content = str(result)
            else:
                content = str(result)
            
            return result, content
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Check if required environment variables are set
    if not os.environ.get("RACE_AWS_ACCESS_KEY_ID") or not os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
        logger.error(
            "RACE_AWS_ACCESS_KEY_ID or RACE_AWS_SECRET_ACCESS_KEY environment variables not set")
        exit(1)

    # Create a SageMakerClient instance
    client = SageMakerClient(
        model_id="tiiuae/falcon-3-10b",
        instance_type="ml.g6.12xlarge"
    )

    try:
        # Deploy the model
        deployment = client.create_deployment()
        print(f"Model deployed to endpoint: {deployment['endpoint_name']}")
        
        # Test the endpoint
        prompt = "Hello, my name is"
        content, raw_response = client.complete_chat_once(
            prompt,
            system_message="You are an AI assistant that provides clear, concise explanations."
        )
        
        print("\nResponse from SageMaker endpoint:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
    finally:
        # Clean up resources
        client.cleanup_resources()

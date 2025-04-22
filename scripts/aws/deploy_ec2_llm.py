#!/usr/bin/env python3
"""
Script to deploy an LLM on EC2 using CloudFormation with Application Load Balancer access.
"""
import os
import sys
import time
import json
import signal
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError, WaiterError

from utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("deploy_ec2_llm")

class EC2LLMDeployer:
    """
    Class to deploy an LLM on EC2 using CloudFormation with Application Load Balancer access.
    """
    def __init__(
        self,
        model_id: str = "tiiuae/falcon-3-10b-instruct",
        region_name: str = None,
        instance_type: str = "g5.xlarge",
        stack_name: str = None,
        api_key: str = None,
        vllm_port: int = 8000,
        ami_id: str = "ami-04f4302ff68e424cf",
        stage_name: str = "prod",
    ):
        """
        Initialize the EC2 LLM deployer.

        Args:
            model_id (str): The Hugging Face model ID to deploy
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
            instance_type (str): EC2 instance type for deployment
            stack_name (str, optional): CloudFormation stack name. If None, generates one
            api_key (str, optional): API key for vLLM. If None, generates one
            vllm_port (int): Port for vLLM API
            ami_id (str): AMI ID to use for the EC2 instance (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6)
            stage_name (str): Stage name for deployment (used for resource naming)
        """
        # Get AWS credentials from environment variables with RACE_ prefix
        access_key = os.environ.get("RACE_AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY", "")
        session_token = os.environ.get("RACE_AWS_SESSION_TOKEN", "")
        
        # Use provided region_name or get from environment variable
        if region_name is None:
            region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")
        
        # Set up boto3 session with explicit credentials
        self.boto_session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region_name
        )
        
        self.region_name = region_name
        self.cf_client = self.boto_session.client('cloudformation')
        self.ec2_client = self.boto_session.client('ec2')
        
        # Store configuration
        self.model_id = model_id
        self.instance_type = instance_type
        
        # Generate short stack name if not provided
        if stack_name is None:
            # Use short model identifier (first letters of model name parts)
            model_parts = model_id.split('/')[-1].split('-')
            model_short = ''.join([part[0] for part in model_parts])
            # Use last 4 digits of timestamp for uniqueness
            timestamp_short = str(int(time.time()))[-4:]
            self.stack_name = f"llm-{model_short}-{timestamp_short}"
        else:
            self.stack_name = stack_name
        
        # Store other parameters
        self.vllm_port = vllm_port
        self.stage_name = stage_name
        self.ami_id = ami_id
        
        # Generate API key if not provided
        if api_key is None:
            import uuid
            self.api_key = str(uuid.uuid4())
        else:
            self.api_key = api_key
        
        # Track deployed resources
        self.instance_id = None
        self.alb_endpoint = None
        self.vllm_api_endpoint = None
        self.vllm_log_endpoint = None
        
        logger.debug(f"Initialized EC2 LLM deployer with model: {model_id}")

    def print_stack_events(self, last_event_id=None):
        """
        Print the latest stack events.
        
        Args:
            last_event_id (str, optional): ID of the last event that was printed
            
        Returns:
            str: ID of the last event that was printed
        """
        try:
            # Get stack events
            response = self.cf_client.describe_stack_events(StackName=self.stack_name)
            events = response.get('StackEvents', [])
            
            # Sort events by timestamp (newest first)
            events.sort(key=lambda x: x['Timestamp'], reverse=True)
            
            # Print only new events (events that occurred after the last_event_id)
            new_events = []
            for event in events:
                if last_event_id and event['EventId'] == last_event_id:
                    break
                new_events.append(event)
            
            # Reverse to print in chronological order
            new_events.reverse()
            
            # Print new events
            for event in new_events:
                # timestamp = event['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                status = event.get('ResourceStatus', 'UNKNOWN')
                resource_type = event.get('ResourceType', 'Unknown')
                logical_id = event.get('LogicalResourceId', 'Unknown')
                reason = event.get('ResourceStatusReason', '')
                
                # Format the status with color if possible
                status_str = status
                if 'COMPLETE' in status:
                    status_str = f"\033[92m{status}\033[0m"  # Green
                elif 'FAILED' in status:
                    status_str = f"\033[91m{status}\033[0m"  # Red
                elif 'IN_PROGRESS' in status:
                    status_str = f"\033[93m{status}\033[0m"  # Yellow
                
                # Print the event
                event_str = f"{status_str} | {resource_type} | {logical_id}"
                if reason:
                    event_str += f" | {reason}"
                logger.info(event_str)
            
            # Return the ID of the most recent event
            return events[0]['EventId'] if events else last_event_id
            
        except ClientError as e:
            # Check if the error is because the stack doesn't exist
            if "does not exist" in str(e):
                logger.info(f"Stack {self.stack_name} no longer exists.")
                return None
            else:
                logger.error(f"Error getting stack events: {str(e)}")
                return last_event_id
        except Exception as e:
            logger.error(f"Error getting stack events: {str(e)}")
            return last_event_id

    def wait_for_stack_operation(self, operation_type: str, success_status: str, failure_statuses: list) -> None:
        """
        Wait for a stack operation to complete while printing events.
        
        Args:
            operation_type (str): Type of operation (e.g., 'creation', 'deletion')
            success_status (str): Status indicating successful completion
            failure_statuses (list): List of statuses indicating failure
            
        Raises:
            Exception: If the operation fails
        """
        logger.info(f"Waiting for stack {operation_type} to complete...")
        logger.info("Stack events:")
        
        # Initialize variables for tracking events
        last_event_id = None
        operation_complete = False
        start_time = time.time()
        max_wait_time = 600  # 10 minutes
        
        # Poll for stack events until operation completes or times out
        while not operation_complete and (time.time() - start_time) < max_wait_time:
            # Print new stack events
            last_event_id = self.print_stack_events(last_event_id)
            
            # Check stack status
            try:
                stack = self.cf_client.describe_stacks(StackName=self.stack_name)['Stacks'][0]
                status = stack['StackStatus']
                
                if status == success_status:
                    operation_complete = True
                    logger.info(f"Stack {operation_type} completed successfully.")
                elif status in failure_statuses:
                    operation_complete = True
                    logger.error(f"Stack {operation_type} failed with status: {status}")
                    raise Exception(f"Stack {operation_type} failed with status: {status}")
                else:
                    # Wait before polling again
                    time.sleep(10)
            except ClientError as e:
                if "does not exist" in str(e):
                    operation_complete = True
                    if operation_type == 'deletion':
                        logger.info("Stack deletion completed successfully.")
                    else:
                        logger.error(f"Stack no longer exists. {operation_type.capitalize()} likely failed and stack was deleted.")
                        raise Exception(f"Stack {operation_type} failed and stack was deleted.")
                else:
                    logger.error(f"Error checking stack status: {str(e)}")
                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error checking stack status: {str(e)}")
                time.sleep(10)
        
        # If we timed out, use the waiter to wait for completion
        if not operation_complete:
            logger.info(f"Still waiting for stack {operation_type} to complete...")
            waiter = self.cf_client.get_waiter(f'stack_{operation_type.split()[0]}_complete')
            waiter.wait(
                StackName=self.stack_name,
                WaiterConfig={
                    'Delay': 10,
                    'MaxAttempts': 60
                }
            )
            logger.info(f"Stack {operation_type} completed successfully.")

    def deploy(self) -> Dict[str, Any]:
        """
        Deploy the LLM on EC2 using CloudFormation.
        
        Returns:
            Dict containing stack details
        """
        try:
            # Get the CloudFormation template
            template_path = Path(__file__).parent / "config" / "ec2_llm_template.yaml"
            with open(template_path, "r") as f:
                template_body = f.read()
            
            # Create CloudFormation stack
            logger.info(f"Creating CloudFormation stack: {self.stack_name}")
            
            # Define parameters
            parameters = [
                {
                    'ParameterKey': 'ModelId',
                    'ParameterValue': self.model_id
                },
                {
                    'ParameterKey': 'InstanceType',
                    'ParameterValue': self.instance_type
                },
                {
                    'ParameterKey': 'VLLMPort',
                    'ParameterValue': str(self.vllm_port)
                },
                {
                    'ParameterKey': 'ApiKey',
                    'ParameterValue': self.api_key
                },
                {
                    'ParameterKey': 'AmiId',
                    'ParameterValue': self.ami_id
                },
                {
                    'ParameterKey': 'StageName',
                    'ParameterValue': self.stage_name
                }
            ]
            
            response = self.cf_client.create_stack(
                StackName=self.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_IAM'],
                OnFailure='DELETE'
            )
            
            # Wait for stack creation to complete
            self.wait_for_stack_operation(
                operation_type="creation",
                success_status="CREATE_COMPLETE",
                failure_statuses=["CREATE_FAILED", "ROLLBACK_COMPLETE", "ROLLBACK_FAILED"]
            )
            
            # Get stack outputs
            stack = self.cf_client.describe_stacks(StackName=self.stack_name)['Stacks'][0]
            outputs = {output['OutputKey']: output['OutputValue'] for output in stack.get('Outputs', [])}
            
            # Store instance ID and endpoints
            self.instance_id = outputs.get('InstanceId')
            self.alb_endpoint = outputs.get('ALBEndpoint')
            self.vllm_api_endpoint = outputs.get('VLLMApiEndpoint')
            self.vllm_log_endpoint = outputs.get('VLLMLogEndpoint')
            
            logger.info(f"Stack creation complete: {self.stack_name}")
            logger.info(f"Instance ID: {self.instance_id}")
            logger.info(f"ALB Endpoint: {self.alb_endpoint}")
            logger.info(f"vLLM API Endpoint: {self.vllm_api_endpoint}")
            logger.info(f"vLLM Log Endpoint: {self.vllm_log_endpoint}")
            logger.info(f"API Key: {self.api_key}")
            
            return {
                'stack_name': self.stack_name,
                'instance_id': self.instance_id,
                'alb_endpoint': self.alb_endpoint,
                'vllm_api_endpoint': self.vllm_api_endpoint,
                'vllm_log_endpoint': self.vllm_log_endpoint,
                'api_key': self.api_key,
                'model_id': self.model_id
            }
        
        except WaiterError as e:
            logger.error(f"Stack creation failed: {str(e)}")
            self.cleanup()
            raise
        except Exception as e:
            logger.error(f"Error in deployment: {str(e)}")
            self.cleanup()
            raise

    def tail_logs(self) -> None:
        """
        Tail the logs of the vLLM service through the Application Load Balancer endpoint.
        """
        if not self.vllm_log_endpoint:
            logger.error("No log endpoint available. Deploy the stack first.")
            return
        
        try:
            # Monitor logs through ALB
            logger.info(f"Tailing logs from {self.vllm_log_endpoint}...")
            
            # Initial log fetch
            last_log_time = time.time()
            last_log_length = 0
            
            while True:
                try:
                    # Fetch logs with increasing number of lines as time passes
                    lines = 100 + int((time.time() - last_log_time) / 10) * 100
                    response = requests.get(f"{self.vllm_log_endpoint}?lines={lines}", timeout=5)
                    
                    if response.status_code == 200:
                        logs = response.text
                        
                        # Only print new log lines
                        if len(logs) > last_log_length:
                            new_logs = logs[last_log_length:]
                            print(new_logs, end="")
                            last_log_length = len(logs)
                    else:
                        logger.warning(f"Failed to fetch logs: HTTP {response.status_code}")
                        
                    time.sleep(2)
                    
                except requests.RequestException as e:
                    logger.warning(f"Error fetching logs: {str(e)}")
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    logger.info("Log tailing interrupted.")
                    return
                    
        except KeyboardInterrupt:
            logger.info("Log tailing interrupted.")
        except Exception as e:
            logger.error(f"Error tailing logs: {str(e)}")

    def wait_for_service(self, timeout: int = 300) -> bool:
        """
        Wait for the vLLM service to be available through the Application Load Balancer.
        
        Args:
            timeout (int): Timeout in seconds
            
        Returns:
            bool: True if service is available, False otherwise
        """
        if not self.vllm_api_endpoint:
            logger.error("No API endpoint available. Deploy the stack first.")
            return False
        
        logger.info(f"Waiting for vLLM service to be available at {self.vllm_api_endpoint}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the ALB endpoint
                headers = {"X-Api-Key": self.api_key}
                response = requests.get(f"{self.vllm_api_endpoint}/health", headers=headers, timeout=5)
                
                if response.status_code == 200:
                    logger.info("vLLM service is available.")
                    return True
            except requests.RequestException:
                pass
            
            logger.info("Waiting for vLLM service to be available...")
            time.sleep(10)
        
        logger.error(f"vLLM service not available after {timeout} seconds.")
        raise TimeoutError("vLLM service not available after timeout.")

    def cleanup(self) -> None:
        """
        Clean up all resources.
        """
        if self.stack_name:
            try:
                # Check if stack exists before attempting to delete
                try:
                    self.cf_client.describe_stacks(StackName=self.stack_name)
                    stack_exists = True
                except ClientError as e:
                    if "does not exist" in str(e):
                        logger.info(f"Stack {self.stack_name} does not exist, no cleanup needed.")
                        stack_exists = False
                    else:
                        # Re-raise if it's a different error
                        raise
                
                if stack_exists:
                    logger.info(f"Deleting CloudFormation stack: {self.stack_name}")
                    self.cf_client.delete_stack(StackName=self.stack_name)
                    
                    # Wait for stack deletion to complete
                    self.wait_for_stack_operation(
                        operation_type="deletion",
                        success_status="DELETE_COMPLETE",
                        failure_statuses=["DELETE_FAILED"]
                    )
                    
                    logger.info(f"Stack deletion complete: {self.stack_name}")
                
                # Reset instance variables regardless of whether stack existed
                self.stack_name = None
                self.instance_id = None
                self.alb_endpoint = None
                self.vllm_api_endpoint = None
                self.vllm_log_endpoint = None
            except Exception as e:
                logger.error(f"Error deleting stack: {str(e)}")


def signal_handler(sig, frame):
    """
    Handle Ctrl+C to terminate the stack and clean up resources.
    """
    logger.info("Received Ctrl+C. Cleaning up resources...")
    if deployer:
        deployer.cleanup()
    sys.exit(0)


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Check if required environment variables are set
    if not os.environ.get("RACE_AWS_ACCESS_KEY_ID") or not os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
        logger.error(
            "RACE_AWS_ACCESS_KEY_ID or RACE_AWS_SECRET_ACCESS_KEY environment variables not set")
        exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deploy an LLM on EC2 using CloudFormation with Application Load Balancer")
    parser.add_argument("--model-id", type=str, default="tiiuae/falcon-3-10b-instruct",
                        help="Hugging Face model ID to deploy")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region name")
    parser.add_argument("--instance-type", type=str, default="g5.xlarge",
                        help="EC2 instance type")
    parser.add_argument("--stack-name", type=str, default=None,
                        help="CloudFormation stack name")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for vLLM")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="Port for vLLM API")
    parser.add_argument("--ami-id", type=str, default="ami-04f4302ff68e424cf",
                        help="AMI ID to use for the EC2 instance (Deep Learning OSS Nvidia Driver AMI)")
    parser.add_argument("--stage-name", type=str, default="prod",
                        help="Stage name for deployment (used for resource naming)")
    
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create deployer
    deployer = EC2LLMDeployer(
        model_id=args.model_id,
        region_name=args.region,
        instance_type=args.instance_type,
        stack_name=args.stack_name,
        api_key=args.api_key,
        vllm_port=args.vllm_port,
        ami_id=args.ami_id,
        stage_name=args.stage_name
    )
    
    try:
        # Deploy the stack
        deployment = deployer.deploy()
        
        # Wait for the service to be available
        if deployer.wait_for_service():
            # Tail the logs
            deployer.tail_logs()
    except KeyboardInterrupt:
        logger.info("Deployment interrupted.")
        deployer.cleanup()
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        deployer.cleanup()

#!/usr/bin/env python3
"""
Script to deploy an LLM on EC2 using CloudFormation.
Connects to the EC2 instance using Session Manager, installs and runs vLLM directly,
and sets up port forwarding for local access.

All resources created in this script must be deleted upon exiting!
"""
import os
import sys
import time
import signal
import argparse
import subprocess
import requests
import atexit
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.exceptions import ClientError, WaiterError

# Add scripts folder to the Python path to allow importing from scripts
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.logging_utils import get_logger
from session_manager import SessionManager

# Initialize logger
logger = get_logger("deploy_ec2_llm")

class EC2LLMDeployer:
    """
    Class to deploy an LLM on EC2 using CloudFormation.
    Connects to the EC2 instance using Session Manager, installs and runs vLLM directly,
    and sets up port forwarding for local access.
    
    Can be used as a context manager to ensure proper cleanup of resources.
    """
    def __init__(
        self,
        model_id: str = "tiiuae/Falcon3-10B-Instruct",
        region_name: str = None,
        instance_type: str = "g6e.4xlarge",
        stack_name: str = None,
        local_port: int = 8987,
        ami_id: str = "ami-04f4302ff68e424cf",
    ):
        """
        Initialize the EC2 LLM deployer.

        Args:
            model_id (str): The Hugging Face model ID to deploy
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
            instance_type (str): EC2 instance type for deployment
            stack_name (str, optional): CloudFormation stack name. If None, generates one
            local_port (int): Local port for port forwarding (default: 8987)
            ami_id (str): AMI ID to use for the EC2 instance (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6)
        """
        # Use provided region_name or get from environment variable
        if region_name is None:
            region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")
        
        self.region_name = region_name
        
        # Initialize session manager to use its boto3 session
        self.session_manager = SessionManager(region_name=region_name)
        self.boto_session = self.session_manager.boto_session
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
        self.remote_port = 8000  # Fixed remote port for vLLM
        self.local_port = local_port  # Configurable local port for port forwarding
        self.ami_id = ami_id
        
        # Use EC2_LLM_API_KEY from environment
        self.api_key = os.environ.get("EC2_LLM_API_KEY", "")
        
        # Track deployed resources
        self.instance_id = None
        
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
                    'ParameterKey': 'AmiId',
                    'ParameterValue': self.ami_id
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
            
            # Store instance ID
            self.instance_id = outputs.get('InstanceId')
            
            logger.info(f"Stack creation complete: {self.stack_name}")
            logger.info(f"Instance ID: {self.instance_id}")
            logger.info(f"API Key: {self.api_key}")
            
            return {
                'stack_name': self.stack_name,
                'instance_id': self.instance_id,
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

    def wait_for_instance_ssm_ready(self, max_attempts=60, delay_seconds=5) -> bool:
        """
        Returns:
            bool: True if the instance is ready, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False
        
        
        logger.info(f"Waiting for instance {self.instance_id} to be ready for SSM connections...")
        
        ssm_client = self.boto_session.client('ssm')
        
        for attempt in range(max_attempts):
            try:
                # Check if the instance is registered with SSM
                response = ssm_client.describe_instance_information(
                    Filters=[
                        {
                            'Key': 'InstanceIds',
                            'Values': [self.instance_id]
                        }
                    ]
                )
                
                if response.get('InstanceInformationList'):
                    instance_info = response['InstanceInformationList'][0]
                    ping_status = instance_info.get('PingStatus')
                    status = instance_info.get('Status')
                    
                    logger.info(f"Instance SSM status: PingStatus={ping_status}, Status={status}")
                    
                    if ping_status == 'Online':
                        logger.info(f"Instance {self.instance_id} is ready for SSM connections")
                        return True
                
                logger.info(f"Instance not ready yet. Attempt {attempt+1}/{max_attempts}. Waiting {delay_seconds} seconds...")
                time.sleep(delay_seconds)
                
            except Exception as e:
                logger.warning(f"Error checking instance SSM status: {str(e)}")
                time.sleep(delay_seconds)
        
        logger.error(f"Instance {self.instance_id} did not become ready for SSM connections after {max_attempts} attempts")
        return False
    
    def connect_to_instance(self) -> bool:
        """
        Connect to the EC2 instance using Session Manager.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False
        
        try:
            # Check if we can connect to the instance
            logger.info("Checking connection to instance...")
            result = self.session_manager.execute_command(
                instance_id=self.instance_id,
                commands=["echo 'Connection successful'"]
            )
            
            if result.get('Status') == 'Success':
                logger.info("Connection to instance successful.")
                return True
            else:
                logger.error("Failed to connect to instance.")
                return False
        except Exception as e:
            logger.error(f"Error connecting to instance: {str(e)}")
            return False
    
    def run_vllm_service(self) -> bool:
        """
        Install and run vLLM directly on the EC2 instance as a systemd service.
        
        Returns:
            bool: True if service is started successfully, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False
        
        try:
            # Upload the installation script
            install_script_path = Path(__file__).parent / "config" / "install_vllm.sh"
            remote_install_path = "/tmp/install_vllm.sh"
            
            logger.info(f"Uploading vLLM installation script to {remote_install_path}...")
            self.session_manager.upload_file(
                instance_id=self.instance_id,
                local_path=str(install_script_path),
                remote_path=remote_install_path
            )
            
            # Make the script executable and run it
            logger.info("Installing vLLM...")
            self.session_manager.execute_command(
                instance_id=self.instance_id,
                commands=[
                    f"chmod +x {remote_install_path}",
                    f"sudo {remote_install_path}"
                ]
            )
            
            # Upload the service setup script
            service_script_path = Path(__file__).parent / "config" / "setup_vllm_service.sh"
            remote_service_path = "/tmp/setup_vllm_service.sh"
            
            logger.info(f"Uploading vLLM service setup script to {remote_service_path}...")
            self.session_manager.upload_file(
                instance_id=self.instance_id,
                local_path=str(service_script_path),
                remote_path=remote_service_path
            )
            
            # Make the script executable and run it with parameters
            logger.info("Setting up vLLM service...")
            result = self.session_manager.execute_command(
                instance_id=self.instance_id,
                commands=[
                    f"chmod +x {remote_service_path}",
                    f"sudo {remote_service_path} {self.model_id} {self.remote_port} {self.api_key}"
                ]
            )
            
            # Check if the service was started successfully
            if result.get('Status') == 'Success':
                logger.info("vLLM service started successfully.")
                return True
            else:
                time.sleep(3600)  # Wait for a moment before checking logs
                logger.error("Failed to start vLLM service.")
                return False
        except Exception as e:
            logger.error(f"Error setting up vLLM service: {str(e)}")
            return False
    
    def setup_port_forwarding(self) -> Dict[str, Any]:
        """
        Set up port forwarding to the EC2 instance using SessionManager.
        
        Returns:
            Dict containing the process and monitoring thread information
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return None
        
        try:
            # Use the SessionManager's setup_port_forwarding method
            return self.session_manager.setup_port_forwarding(
                instance_id=self.instance_id,
                remote_port=self.remote_port,
                local_port=self.local_port
            )
        except Exception as e:
            logger.error(f"Error setting up port forwarding: {str(e)}")
            return None
    
    def tail_service_logs(self, in_thread=False) -> Optional[threading.Thread]:
        """
        Tail the logs of the vLLM systemd service using direct AWS SSM command.
        Logs are redirected to a file in /tmp/ instead of being displayed in real-time.
        
        Args:
            in_thread (bool): If True, run the log tailing in a separate thread
            
        Returns:
            Optional[threading.Thread]: The thread object if in_thread is True, None otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return None
        
        def _tail_logs():
            try:
                logger.info("Tailing vLLM service logs...")
                
                # Create a log file with a unique name
                log_file_path = f"/tmp/vllm_service_logs_{self.instance_id}_{int(time.time())}.log"
                log_file = open(log_file_path, "w")
                
                # Construct the AWS SSM command to directly execute journalctl
                ssm_command = [
                    "aws", "ssm", "start-session",
                    "--target", self.instance_id,
                    "--region", self.region_name,
                    "--document-name", "AWS-StartInteractiveCommand",
                    "--parameters", "command=['sudo journalctl -u vllm.service -f']"
                ]
                
                # Set up environment variables for the subprocess
                env = os.environ.copy()
                
                # Copy RACE_ prefixed AWS credentials to standard AWS environment variables
                if os.environ.get("RACE_AWS_ACCESS_KEY_ID"):
                    env["AWS_ACCESS_KEY_ID"] = os.environ.get("RACE_AWS_ACCESS_KEY_ID")
                if os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
                    env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY")
                if os.environ.get("RACE_AWS_SESSION_TOKEN"):
                    env["AWS_SESSION_TOKEN"] = os.environ.get("RACE_AWS_SESSION_TOKEN")
                if os.environ.get("RACE_AWS_REGION"):
                    env["AWS_REGION"] = os.environ.get("RACE_AWS_REGION")
                
                # Execute the command and redirect output to the log file
                logger.info(f"Running command: {' '.join(ssm_command)}")
                process = subprocess.Popen(
                    ssm_command,
                    stdout=log_file,
                    stderr=log_file,
                    env=env
                )
                
                logger.info(f"To view vLLM logs in real-time, run: tail -f {log_file_path}")
                
                # Wait for the process to complete
                process.wait()
                
                # Close the log file
                log_file.close()
                
                # Check return code
                if process.returncode != 0:
                    logger.error(f"Log tailing process exited with code {process.returncode}")
                    logger.error(f"Check logs at {log_file_path} for details")
                
            except KeyboardInterrupt:
                logger.info("Log tailing interrupted.")
                logger.info(f"Logs are available at: {log_file_path}")
            except Exception as e:
                logger.error(f"Error tailing service logs: {str(e)}")
        
        if in_thread:
            # Run in a separate thread
            log_thread = threading.Thread(target=_tail_logs, daemon=True)
            log_thread.start()
            logger.info("Started vLLM log tailing in a separate thread")
            return log_thread
        else:
            # Run in the main thread
            try:
                _tail_logs()
            except KeyboardInterrupt:
                logger.info("Log tailing interrupted.")
            return None
    
    def wait_for_llm_ready(self, max_wait_time: int = 900, check_interval: int = 5) -> bool:
        """
        Poll the vLLM health endpoint until it returns a successful response or times out.
        
        Args:
            max_wait_time (int): Maximum time to wait in seconds
            check_interval (int): Time between health checks in seconds
            
        Returns:
            bool: True if the LLM is ready, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False
        
        health_url = f"http://localhost:{self.local_port}/health"
        logger.info(f"Waiting for vLLM to be ready (polling {health_url})...")
        
        # Prepare headers with API key if provided
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            try:
                response = requests.get(health_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM is ready!")
                    
                    # Print common vLLM API endpoints
                    base_url = f"http://localhost:{self.local_port}"
                    logger.info("Common vLLM API endpoints:")
                    logger.info(f"  Health check:     {base_url}/health")
                    logger.info(f"  List models:      {base_url}/v1/models")
                    logger.info(f"  Text completion:  {base_url}/v1/completions")
                    logger.info(f"  Chat completion:  {base_url}/v1/chat/completions")
                    logger.info(f"  Embeddings:       {base_url}/v1/embeddings")
                    
                    return True
                else:
                    logger.debug(f"vLLM not ready yet. Status code: {response.status_code}")
            except requests.exceptions.RequestException:
                logger.debug("vLLM health check failed, retrying...")
            
            # Wait before checking again
            time.sleep(check_interval)
        
        logger.error(f"Timed out waiting for vLLM to be ready after {max_wait_time} seconds")
        raise TimeoutError(f"vLLM did not become ready in ({max_wait_time} seconds)")
    
    def test_llm(self, max_wait_time: int = 900) -> bool:
        """
        Send a test request to the LLM and print the response.
        
        Args:
            max_wait_time (int): Maximum time to wait for the LLM to be ready in seconds
            
        Returns:
            bool: True if the test was successful, False otherwise
            
        Raises:
            TimeoutError: If the LLM does not become ready within the specified time
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False
        
        # Poll the health endpoint until the LLM is ready
        try:
            self.wait_for_llm_ready(max_wait_time=max_wait_time)
        except TimeoutError as e:
            logger.error(f"LLM is not ready: {str(e)}")
            raise
        
        try:
            # Prepare the test request
            url = f"http://localhost:{self.local_port}/v1/completions"
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Prepare a friendly welcome message
            data = {
                "model": self.model_id,
                "prompt": "Say hello and introduce yourself as a helpful AI assistant deployed on AWS EC2.",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            logger.info("Sending test request to the LLM...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("text", "").strip()
                
                logger.info("LLM Test Response:")
                logger.info("-" * 50)
                logger.info(generated_text)
                logger.info("-" * 50)
                logger.info("LLM is working correctly!")
                return True
            else:
                logger.error(f"Error testing LLM: Status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to LLM: {str(e)}")
            logger.info("The LLM might still be initializing. Try again in a few moments.")
            return False
        except Exception as e:
            logger.error(f"Error testing LLM: {str(e)}")
            return False
    
    def __enter__(self):
        """
        Enter the context manager.
        
        Returns:
            EC2LLMDeployer: The deployer instance
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and clean up resources.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        logger.info("Exiting context manager, cleaning up resources...")
        self.cleanup()
    
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
            except ClientError as e:
                if "ExpiredToken" in str(e) or "InvalidToken" in str(e) or "NotSignedUp" in str(e) or "CredentialsNotFound" in str(e) or "InvalidClientTokenId" in str(e):
                    stack_name = self.stack_name
                    region = self.region_name
                    logger.error(f"Session token issue detected during cleanup: {str(e)}")
                    logger.error(f"IMPORTANT: Please manually delete the CloudFormation stack '{stack_name}' in region '{region}'")
                    logger.error(f"Visit: https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks")
                    logger.error("Find your stack in the list, select it, and click 'Delete' from the Actions menu")
                else:
                    logger.error(f"Error deleting stack: {str(e)}")
            except Exception as e:
                logger.error(f"Error deleting stack: {str(e)}")
                logger.error(f"If cleanup failed due to session issues, please manually delete the stack '{self.stack_name}'")
                logger.error(f"Visit: https://{self.region_name}.console.aws.amazon.com/cloudformation/home?region={self.region_name}#/stacks")



def signal_handler(sig, frame):
    """
    Handle signals to terminate the stack and clean up resources.
    """
    signal_name = signal.Signals(sig).name
    logger.info(f"Received signal {signal_name}. Cleaning up resources...")
    if deployer:
        deployer.cleanup()
    sys.exit(0)


def cleanup_on_exit():
    """
    Ensure cleanup happens when the script exits for any reason.
    """
    logger.info("Script is exiting. Ensuring all resources are cleaned up...")
    if deployer:
        deployer.cleanup()


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
    parser = argparse.ArgumentParser(description="Deploy an LLM on EC2 using CloudFormation")
    parser.add_argument("--model-id", type=str, default="tiiuae/Falcon3-10B-Instruct",
                        help="Hugging Face model ID to deploy")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region name")
    parser.add_argument("--instance-type", type=str, default="g6e.4xlarge",
                        help="EC2 instance type")
    parser.add_argument("--stack-name", type=str, default=None,
                        help="CloudFormation stack name")
    parser.add_argument("--local-port", type=int, default=8987,
                        help="Local port for port forwarding (default: 8987)")
    parser.add_argument("--ami-id", type=str, default="ami-04f4302ff68e424cf",
                        help="AMI ID to use for the EC2 instance (Deep Learning OSS Nvidia Driver AMI)")
    parser.add_argument("--stage-name", type=str, default="prod",
                        help="Stage name for deployment (used for resource naming)")
    parser.add_argument("--port-forward", action="store_true", default=True,
                        help="Set up port forwarding to the EC2 instance")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Send a test request to the LLM after deployment")
    parser.add_argument("--wait-time", type=int, default=1800,
                        help="Time to wait in seconds before sending the test request")
    
    args = parser.parse_args()
    
    # Register signal handlers for various termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    signal.signal(signal.SIGHUP, signal_handler)   # Terminal closed
    
    # Register atexit handler to ensure cleanup on exit
    atexit.register(cleanup_on_exit)
    
    # Create deployer using context manager pattern
    with EC2LLMDeployer(
        model_id=args.model_id,
        region_name=args.region,
        instance_type=args.instance_type,
        stack_name=args.stack_name,
        local_port=args.local_port,
        ami_id=args.ami_id
    ) as deployer:
        try:
            # Deploy the stack
            deployment = deployer.deploy()
            
            # Wait for the instance to be ready for SSM connections
            if not deployer.wait_for_instance_ssm_ready():
                logger.error("Instance not ready for SSM connections. Exiting.")
                raise
                
            # Connect to the instance
            if not deployer.connect_to_instance():
                logger.error("Failed to connect to instance. Exiting.")
                raise
                
            # Install and run vLLM service
            if not deployer.run_vllm_service():
                logger.error("Failed to run vLLM service. Exiting.")
                raise
                
            # Start log tailing in a separate thread
            log_thread = deployer.tail_service_logs(in_thread=True)

            # Set up port forwarding if requested
            port_forwarding = None
            if args.port_forward:
                port_forwarding = deployer.setup_port_forwarding()
                
                # Send a test request to the LLM if requested
                if args.test:
                    deployer.test_llm(max_wait_time=args.wait_time)
                                    
            # Keep the main thread running until interrupted
            try:
                logger.info("Press Ctrl+C to stop...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
            finally:
                # Clean up port forwarding
                if port_forwarding:
                    logger.info("Stopping port forwarding...")
                    # Terminate the process
                    if 'process' in port_forwarding:
                        port_forwarding['process'].terminate()
                    # Close the log file if it exists
                    if 'log_file' in port_forwarding and port_forwarding['log_file']:
                        try:
                            port_forwarding['log_file'].close()
                            logger.info(f"Port forwarding logs are available at: {port_forwarding.get('log_file_path', 'unknown')}")
                        except Exception as e:
                            logger.warning(f"Error closing log file: {str(e)}")
        except KeyboardInterrupt:
            logger.info("Deployment interrupted.")
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            # Context manager will handle cleanup

#!/usr/bin/env python3
"""
Python client for AWS Systems Manager Session Manager.
Provides functionality for connecting to EC2 instances, port forwarding, and executing commands.
"""
import os
import sys
import time
import json
import subprocess
import threading
from typing import Dict, Any, List, Union

import boto3
from botocore.exceptions import ClientError

from utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("session_manager")

class SessionManager:
    """
    Client for AWS Systems Manager Session Manager.
    Provides functionality for connecting to EC2 instances, port forwarding, and executing commands.
    """
    def __init__(
        self,
        region_name: str = None,
    ):
        """
        Initialize the Session Manager client.

        Args:
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
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
        self.ssm_client = self.boto_session.client('ssm')
        self.ec2_client = self.boto_session.client('ec2')
        
        # Check if Session Manager plugin is installed
        self._check_session_manager_plugin()
        
        logger.debug(f"Initialized Session Manager client for region: {region_name}")

    def _check_session_manager_plugin(self) -> bool:
        """
        Check if the Session Manager plugin is installed.
        
        Returns:
            bool: True if installed, False otherwise
        """
        try:
            # First check if the command exists
            result = subprocess.run(
                ["which", "session-manager-plugin"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                plugin_path = result.stdout.strip()
                logger.debug(f"Session Manager plugin found at: {plugin_path}")
                
                # Verify the plugin works by checking its version
                version_result = subprocess.run(
                    ["session-manager-plugin", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                # If we got any output or a zero return code, consider it working
                if version_result.returncode == 0 or version_result.stdout or "SessionManagerPlugin" in version_result.stderr:
                    logger.debug("Session Manager plugin is installed and working")
                    return True
            
            # If we reach here, the plugin exists but might not be working correctly
            logger.warning(
                "Session Manager plugin found but may not be working correctly. Please verify installation: "
                "https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html\n"
                "On macOS, you can install it using: brew install --cask session-manager-plugin\n"
            )
            return False
            
        except FileNotFoundError:
            logger.warning(
                "Session Manager plugin not found. Please install it: "
                "https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html\n"
                "On macOS, you can install it using: brew install --cask session-manager-plugin\n"
            )
            return False

    def list_instances(self) -> List[Dict[str, Any]]:
        """
        List all EC2 instances that are managed by Systems Manager.
        
        Returns:
            List[Dict[str, Any]]: List of instance information
        """
        try:
            response = self.ssm_client.describe_instance_information()
            instances = response.get('InstanceInformationList', [])
            
            # Get additional information from EC2 API
            if instances:
                instance_ids = [instance['InstanceId'] for instance in instances]
                ec2_response = self.ec2_client.describe_instances(
                    InstanceIds=instance_ids
                )
                
                # Create a mapping of instance ID to EC2 instance details
                ec2_instances = {}
                for reservation in ec2_response.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        instance_id = instance['InstanceId']
                        name_tag = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), '')
                        ec2_instances[instance_id] = {
                            'InstanceType': instance.get('InstanceType', ''),
                            'State': instance.get('State', {}).get('Name', ''),
                            'PrivateIpAddress': instance.get('PrivateIpAddress', ''),
                            'PublicIpAddress': instance.get('PublicIpAddress', ''),
                            'Name': name_tag
                        }
                
                # Merge SSM and EC2 information
                for instance in instances:
                    instance_id = instance['InstanceId']
                    if instance_id in ec2_instances:
                        instance.update(ec2_instances[instance_id])
            
            return instances
        
        except ClientError as e:
            logger.error(f"Error listing instances: {str(e)}")
            return []

    def start_session(self, instance_id: str) -> subprocess.Popen:
        """
        Start an interactive session with an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            
        Returns:
            subprocess.Popen: Process object for the session
        """
        try:
            logger.info(f"Starting session with instance {instance_id}")
            
            # Start the session using boto3
            response = self.ssm_client.start_session(
                Target=instance_id
            )
            
            # Format the command to call session-manager-plugin
            plugin_command = [
                "session-manager-plugin",
                json.dumps(response),
                self.region_name,
                "StartSession",
                "default",  # Use "default" instead of empty string for profile name
                json.dumps({}),
                f"https://ssm.{self.region_name}.amazonaws.com"
            ]
            logger.info(f"Session command: {' '.join(plugin_command)}")
            
            # Execute the session
            process = subprocess.Popen(
                plugin_command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                stdin=sys.stdin
            )
            
            return process
        
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}")
            raise

    def start_port_forwarding(
        self,
        instance_id: str,
        remote_port: int,
        local_port: int = None
    ) -> subprocess.Popen:
        """
        Start port forwarding to an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            remote_port (int): Port on the remote instance
            local_port (int, optional): Local port to forward to. If None, uses the same as remote_port
            
        Returns:
            subprocess.Popen: Process object for the port forwarding
        """
        if local_port is None:
            local_port = remote_port
        
        try:
            logger.info(f"Starting port forwarding: localhost:{local_port} -> {instance_id}:{remote_port}")
            
            # Start port forwarding using boto3
            response = self.ssm_client.start_session(
                Target=instance_id,
                DocumentName="AWS-StartPortForwardingSession",
                Parameters={
                    "portNumber": [str(remote_port)],
                    "localPortNumber": [str(local_port)]
                }
            )
            
            # Format the command to call session-manager-plugin
            plugin_command = [
                "session-manager-plugin",
                json.dumps(response),
                self.region_name,
                "StartSession",
                "default",  # Use "default" instead of empty string for profile name
                json.dumps({}),
                f"https://ssm.{self.region_name}.amazonaws.com"
            ]
            
            # Execute the port forwarding
            process = subprocess.Popen(
                plugin_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the connection to establish
            time.sleep(2)
            
            # Check if the process is still running
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                logger.error(f"Port forwarding failed: {stderr}")
                raise Exception(f"Port forwarding failed: {stderr}")
            
            logger.info(f"Port forwarding established: localhost:{local_port} -> {instance_id}:{remote_port}")
            return process
        
        except Exception as e:
            logger.error(f"Error setting up port forwarding: {str(e)}")
            raise

    def execute_command(
        self,
        instance_id: str,
        commands: Union[str, List[str]],
        working_directory: str = None,
        execution_timeout: int = 3600,
        wait_for_completion: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a command on an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            commands (Union[str, List[str]]): Command(s) to execute
            working_directory (str, optional): Working directory for command execution
            execution_timeout (int, optional): Timeout in seconds
            wait_for_completion (bool, optional): Whether to wait for command completion
            
        Returns:
            Dict[str, Any]: Command execution results
        """
        if isinstance(commands, str):
            commands = [commands]
        
        try:
            # Prepare parameters
            parameters = {
                'commands': commands
            }
            
            if working_directory:
                parameters['workingDirectory'] = [working_directory]
            
            # Send the command
            logger.info(f"Executing command on instance {instance_id}: {commands}")
            response = self.ssm_client.send_command(
                InstanceIds=[instance_id],
                DocumentName='AWS-RunShellScript',
                Parameters=parameters,
                TimeoutSeconds=execution_timeout
            )
            
            command_id = response['Command']['CommandId']
            logger.debug(f"Command ID: {command_id}")
            
            if not wait_for_completion:
                return {
                    'CommandId': command_id,
                    'Status': 'Pending'
                }
            
            # Wait for command completion
            while True:
                time.sleep(1)
                result = self.ssm_client.get_command_invocation(
                    CommandId=command_id,
                    InstanceId=instance_id
                )
                
                status = result['Status']
                
                if status in ['Success', 'Failed', 'Cancelled', 'TimedOut']:
                    logger.info(f"Command execution completed with status: {status}")
                    return result
        
        except ClientError as e:
            logger.error(f"Error executing command: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            raise

    def execute_script(
        self,
        instance_id: str,
        script_content: str,
        working_directory: str = None,
        execution_timeout: int = 3600,
        save_output_path: str = None
    ) -> Dict[str, Any]:
        """
        Execute a script on an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            script_content (str): Content of the script to execute
            working_directory (str, optional): Working directory for script execution
            execution_timeout (int, optional): Timeout in seconds
            save_output_path (str, optional): Path to save command output locally
            
        Returns:
            Dict[str, Any]: Script execution results
        """
        try:
            # Execute the script
            result = self.execute_command(
                instance_id=instance_id,
                commands=[script_content],
                working_directory=working_directory,
                execution_timeout=execution_timeout
            )
            
            # Save output if requested
            if save_output_path and 'StandardOutputContent' in result:
                output_dir = os.path.dirname(save_output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(save_output_path, 'w') as f:
                    f.write(result.get('StandardOutputContent', ''))
                    if result.get('StandardErrorContent'):
                        f.write('\n--- STDERR ---\n')
                        f.write(result.get('StandardErrorContent', ''))
                
                logger.info(f"Command output saved to {save_output_path}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing script: {str(e)}")
            raise

    def upload_file(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str
    ) -> Dict[str, Any]:
        """
        Upload a file to an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            local_path (str): Path to the local file
            remote_path (str): Path on the remote instance
            
        Returns:
            Dict[str, Any]: Upload results
        """
        try:
            logger.info(f"Uploading file: {local_path} -> {instance_id}:{remote_path}")
            
            # Create remote directory if it doesn't exist
            self.execute_command(
                instance_id=instance_id,
                commands=[f"mkdir -p {os.path.dirname(remote_path)}"]
            )
            
            # Read the local file content in binary mode to handle all file types
            import base64
            with open(local_path, 'rb') as f:
                file_content = f.read()
            
            # Write the content to the remote file using execute_command
            # We use base64 encoding to handle binary data and special characters
            encoded_content = base64.b64encode(file_content).decode()
            
            # Create a command to decode and write the file on the remote instance
            write_command = f"echo '{encoded_content}' | base64 -d > {remote_path}"
            
            # Execute the command to write the file
            result = self.execute_command(
                instance_id=instance_id,
                commands=[write_command]
            )
            
            if result.get('Status') == 'Success':
                # Make shell scripts executable
                if local_path.endswith('.sh'):
                    logger.info(f"Making shell script executable: {remote_path}")
                    chmod_result = self.execute_command(
                        instance_id=instance_id,
                        commands=[f"sudo chmod +x {remote_path}"]
                    )
                    if chmod_result.get('Status') != 'Success':
                        logger.warning(f"Failed to make script executable: {chmod_result.get('StandardErrorContent', 'Unknown error')}")
                
                logger.info(f"File uploaded successfully: {local_path} -> {instance_id}:{remote_path}")
                return {
                    'Status': 'Success',
                    'Output': result.get('StandardOutputContent', '')
                }
            else:
                error_msg = f"Failed to upload file: {result.get('StandardErrorContent', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise

    def download_file(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str
    ) -> Dict[str, Any]:
        """
        Download a file from an EC2 instance.
        
        Args:
            instance_id (str): ID of the EC2 instance
            remote_path (str): Path on the remote instance
            local_path (str): Path to save the file locally
            
        Returns:
            Dict[str, Any]: Download results
        """
        try:
            logger.info(f"Downloading file: {instance_id}:{remote_path} -> {local_path}")
            
            # Create local directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            # Read the remote file using cat and base64 encoding
            read_command = f"cat {remote_path} | base64"
            result = self.execute_command(
                instance_id=instance_id,
                commands=[read_command]
            )
            
            if result.get('Status') == 'Success':
                # Decode the base64 content
                import base64
                encoded_content = result.get('StandardOutputContent', '').strip()
                file_content = base64.b64decode(encoded_content)
                
                # Write the content to the local file in binary mode
                with open(local_path, 'wb') as f:
                    f.write(file_content)
                
                logger.info(f"File downloaded successfully: {instance_id}:{remote_path} -> {local_path}")
                return {
                    'Status': 'Success',
                    'Output': ''
                }
            else:
                error_msg = f"Failed to download file: {result.get('StandardErrorContent', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    
    def setup_port_forwarding(
        self,
        instance_id: str,
        remote_port: int,
        local_port: int = None,
        health_check_fn: callable = None,
    ) -> Dict[str, Any]:
        """
        Set up port forwarding to the EC2 instance using direct AWS SSM command.
        Logs are redirected to a file in /tmp/ instead of being displayed in real-time.
        If health_check_fn is provided, it will monitor the port forwarding and restart it if needed.
        
        Args:
            instance_id (str): ID of the EC2 instance
            remote_port (int): Port on the remote instance
            local_port (int, optional): Local port to forward to. If None, uses the same as remote_port
            health_check_fn (callable, optional): Function that returns True if port forwarding is healthy, False otherwise
            
        Returns:
            Dict containing the process information and log file path
        """
        if local_port is None:
            local_port = remote_port
            
        try:
            logger.info(f"Setting up port forwarding: localhost:{local_port} -> {instance_id}:{remote_port}")
            
            # Create a log file with a unique name
            log_file_path = f"/tmp/port_forwarding_{instance_id}_{remote_port}_{local_port}_{int(time.time())}.log"
            log_file = open(log_file_path, "w")
            
            # Construct the AWS SSM command for port forwarding
            ssm_command = [
                "aws", "ssm", "start-session",
                "--target", instance_id,
                "--region", self.region_name,
                "--document-name", "AWS-StartPortForwardingSession",
                "--parameters", f'{{"portNumber":["{remote_port}"],"localPortNumber":["{local_port}"]}}'
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
            
            # Start port forwarding using direct AWS SSM command, redirecting output to the log file
            process = subprocess.Popen(
                ssm_command,
                stdout=log_file,
                stderr=log_file,
                env=env
            )
            
            # Wait a moment for the connection to establish
            time.sleep(2)
            
            # Check if the process is still running
            if process.poll() is not None:
                log_file.close()
                with open(log_file_path, "r") as f:
                    error_output = f.read()
                logger.error(f"Port forwarding failed. See logs at {log_file_path}")
                logger.error(f"Error: {error_output}")
                raise Exception(f"Port forwarding failed. See logs at {log_file_path}")
            
            logger.info(f"Port forwarding established: localhost:{local_port} -> {instance_id}:{remote_port}")
            logger.info(f"Port forwarding logs are being written to: {log_file_path}")
            
            result = {
                'process': process,
                'log_file': log_file,
                'log_file_path': log_file_path
            }
            
            # Set up health check monitoring if a health check function is provided
            if health_check_fn:
                def monitor_port_forwarding():
                    logger.info(f"Starting health check monitoring for port forwarding: localhost:{local_port} -> {instance_id}:{remote_port}")
                    while True:
                        # Check if the process is still running
                        if process.poll() is not None:
                            logger.warning(f"Port forwarding process terminated unexpectedly. Restarting...")
                            break
                        
                        # Check health using the provided function
                        try:
                            if not health_check_fn():
                                logger.warning(f"Health check failed for port forwarding: localhost:{local_port} -> {instance_id}:{remote_port}. Restarting...")
                                break
                        except Exception as e:
                            logger.error(f"Error in health check function: {str(e)}")
                        
                        # Wait before next check
                        time.sleep(5)
                    
                    # If we're here, either the process died or health check failed
                    # Try to terminate the process if it's still running
                    try:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=5)
                    except Exception as e:
                        logger.error(f"Error terminating port forwarding process: {str(e)}")
                    
                    # Close the log file
                    try:
                        if not log_file.closed:
                            log_file.close()
                    except Exception as e:
                        logger.error(f"Error closing log file: {str(e)}")
                    
                    # Restart port forwarding
                    try:
                        logger.info(f"Restarting port forwarding: localhost:{local_port} -> {instance_id}:{remote_port}")
                        new_result = self.setup_port_forwarding(
                            instance_id=instance_id,
                            remote_port=remote_port,
                            local_port=local_port,
                            health_check_fn=health_check_fn
                        )
                        # Update the original result dictionary with new values
                        result.update(new_result)
                    except Exception as e:
                        logger.error(f"Failed to restart port forwarding: {str(e)}")
                
                # Start monitoring in a separate thread
                monitor_thread = threading.Thread(target=monitor_port_forwarding, daemon=True)
                monitor_thread.start()
                result['monitor_thread'] = monitor_thread
            
            return result
        except Exception as e:
            logger.error(f"Error setting up port forwarding: {str(e)}")
            raise

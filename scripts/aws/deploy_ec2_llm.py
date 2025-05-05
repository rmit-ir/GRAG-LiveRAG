#!/usr/bin/env python3
"""
Script to deploy an application on EC2 using CloudFormation.
Connects to the EC2 instance using Session Manager, installs and runs the application directly,
and sets up port forwarding for local access.

All resources created in this script must be deleted upon exiting!
"""
from services import aws_costs
from services.aws_costs import get_ec2_price
from session_manager import SessionManager
from utils.query_utils import generate_short_id
from utils.logging_utils import get_logger
from ec2_app import EC2App, create_vllm_app, create_mini_tgi_app
import os
import sys
import time
import signal
import argparse
import subprocess
import atexit
import threading
import socket
import glob
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from botocore.exceptions import ClientError, WaiterError

# Add scripts folder to the Python path to allow importing from scripts
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# Initialize logger
logger = get_logger("deploy_ec2_app")


class EC2Deployer:
    """
    Class to deploy an application on EC2 using CloudFormation.
    Connects to the EC2 instance using Session Manager, installs and runs the application directly,
    and sets up port forwarding for local access.

    Can be used as a context manager to ensure proper cleanup of resources.
    """

    def __init__(
        self,
        app: EC2App,
        region_name: str = None,
        instance_type: str = "g6e.4xlarge",
        stack_name: str = None,
        ami_id: str = "ami-04f4302ff68e424cf",
        deployer_id: str = None,
        print_info: bool = True,
    ):
        """
        Initialize the EC2 deployer.

        Args:
            app: The EC2App to deploy
            region_name: AWS region name. If None, uses RACE_AWS_REGION from env
            instance_type: EC2 instance type for deployment
            stack_name: CloudFormation stack name. If None, generates one
            ami_id: AMI ID to use for the EC2 instance
            deployer_id: Custom ID for this deployer. If None, generates one
            print_info: Whether to print stack information
        """
        # Initialize timing tracking
        self.start_time = None
        self.stop_time = None
        # Generate a short ID for this deployer or use the provided one
        self.deployer_id = deployer_id if deployer_id else generate_short_id()

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
        self.app = app
        self.instance_type = instance_type

        # Generate short stack name if not provided
        if stack_name is None:
            # Use app name and timestamp for uniqueness
            timestamp_short = str(int(time.time()))[-4:]
            self.stack_name = f"{app.name}-{timestamp_short}"
        else:
            self.stack_name = stack_name

        # Store other parameters
        self.ami_id = ami_id

        # Track deployed resources
        self.instance_id = None

        # Port forwarding related attributes
        self.port_forwardings = None  # Dictionary of all port forwardings

        # Socket related attributes
        self.socket_path = f"/tmp/ec2_app_{self.deployer_id}_app{app.name}_port{app.local_port}"
        self.socket_thread = None
        self.socket_server = None
        self._should_stop = False

        logger.info(
            f"Initialized EC2 deployer with app: {app.name}, Deployer ID: {self.deployer_id}")
        if print_info:
            self.print_stack_info()

    def print_stack_info(self):
        """
        Print key information about the deploying stack and provide a link to the AWS calculator
        for cost estimation.
        """
        # Get and display hourly price information
        hourly_price = get_ec2_price(self.instance_type, self.region_name)
        if hourly_price is not None:
            logger.info(f"EC2 Hourly Price: ${hourly_price:.4f}/hour")
        else:
            logger.info("EC2 Hourly Price: Price information not available")

        # Use the app's print_info method
        self.app.print_info(
            stack_name=self.stack_name,
            region_name=self.region_name,
            instance_type=self.instance_type,
            ami_id=self.ami_id
        )

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
            response = self.cf_client.describe_stack_events(
                StackName=self.stack_name)
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
                stack = self.cf_client.describe_stacks(
                    StackName=self.stack_name)['Stacks'][0]
                status = stack['StackStatus']

                if status == success_status:
                    operation_complete = True
                    logger.info(
                        f"Stack {operation_type} completed successfully.")
                elif status in failure_statuses:
                    operation_complete = True
                    logger.error(
                        f"Stack {operation_type} failed with status: {status}")
                    raise Exception(
                        f"Stack {operation_type} failed with status: {status}")
                else:
                    # Wait before polling again
                    time.sleep(10)
            except ClientError as e:
                if "does not exist" in str(e):
                    operation_complete = True
                    if operation_type == 'deletion':
                        logger.info("Stack deletion completed successfully.")
                    else:
                        logger.error(
                            f"Stack no longer exists. {operation_type.capitalize()} likely failed and stack was deleted.")
                        raise Exception(
                            f"Stack {operation_type} failed and stack was deleted.")
                else:
                    logger.error(f"Error checking stack status: {str(e)}")
                    time.sleep(10)
            except Exception as e:
                logger.error(f"Error checking stack status: {str(e)}")
                time.sleep(10)

        # If we timed out, use the waiter to wait for completion
        if not operation_complete:
            logger.info(
                f"Still waiting for stack {operation_type} to complete...")
            waiter = self.cf_client.get_waiter(
                f'stack_{operation_type.split()[0]}_complete')
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
        Deploy the application on EC2 using CloudFormation.

        Returns:
            Dict containing stack details
        """
        # Record the start time
        self.start_time = datetime.datetime.now()
        logger.info(
            f"Stack deployment started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Get the CloudFormation template
            template_path = Path(__file__).parent / \
                "config" / "ec2_llm_template.yaml"
            with open(template_path, "r") as f:
                template_body = f.read()

            # Create CloudFormation stack
            logger.info(f"Creating CloudFormation stack: {self.stack_name}")

            # Define parameters
            parameters = [
                {'ParameterKey': 'InstanceType', 'ParameterValue': self.instance_type},
                {'ParameterKey': 'AmiId', 'ParameterValue': self.ami_id}
            ]

            self.cf_client.create_stack(
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
                failure_statuses=["CREATE_FAILED",
                                  "ROLLBACK_COMPLETE", "ROLLBACK_FAILED"]
            )

            # Get stack outputs
            stack = self.cf_client.describe_stacks(
                StackName=self.stack_name)['Stacks'][0]
            outputs = {output['OutputKey']: output['OutputValue']
                       for output in stack.get('Outputs', [])}

            # Store instance ID
            self.instance_id = outputs.get('InstanceId')

            logger.info(f"Stack creation complete: {self.stack_name}")
            logger.info(f"Instance ID: {self.instance_id}")
            logger.info(f"API Key: {self.app.api_key}")

            return {
                'stack_name': self.stack_name,
                'instance_id': self.instance_id,
                'api_key': self.app.api_key,
                'app_name': self.app.name
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

        logger.info(
            f"Waiting for instance {self.instance_id} to be ready for SSM connections...")

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

                    logger.info(
                        f"Instance SSM status: PingStatus={ping_status}, Status={status}")

                    if ping_status == 'Online':
                        logger.info(
                            f"Instance {self.instance_id} is ready for SSM connections")
                        return True

                logger.info(
                    f"Instance not ready yet. Attempt {attempt+1}/{max_attempts}. Waiting {delay_seconds} seconds...")
                time.sleep(delay_seconds)

            except Exception as e:
                logger.warning(f"Error checking instance SSM status: {str(e)}")
                time.sleep(delay_seconds)

        logger.error(
            f"Instance {self.instance_id} did not become ready for SSM connections after {max_attempts} attempts")
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

    def run_app_service(self) -> bool:
        """
        Install and run the application on the EC2 instance.

        Returns:
            bool: True if service is started successfully, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False

        try:
            # Upload the setup script
            remote_setup_path = f"/tmp/{os.path.basename(self.app.setup_script)}"
            logger.info(f"Uploading setup script to {remote_setup_path}...")
            self.session_manager.upload_file(
                instance_id=self.instance_id,
                local_path=str(self.app.setup_script),
                remote_path=remote_setup_path
            )

            # Make the script executable and run it
            logger.info(f"Installing {self.app.name}...")
            self.session_manager.execute_command(
                instance_id=self.instance_id,
                commands=[
                    f"chmod +x {remote_setup_path}",
                    f"sudo {remote_setup_path}"
                ]
            )

            # Upload the program file if provided
            if self.app.program_file:
                remote_program_path = f"/tmp/{os.path.basename(self.app.program_file)}"
                logger.info(
                    f"Uploading program file to {remote_program_path}...")
                self.session_manager.upload_file(
                    instance_id=self.instance_id,
                    local_path=str(self.app.program_file),
                    remote_path=remote_program_path
                )

            # Upload the launch script
            remote_launch_path = f"/tmp/{os.path.basename(self.app.launch_script)}"
            logger.info(f"Uploading launch script to {remote_launch_path}...")
            self.session_manager.upload_file(
                instance_id=self.instance_id,
                local_path=str(self.app.launch_script),
                remote_path=remote_launch_path
            )

            # Make the script executable and run it with parameters
            logger.info(f"Setting up {self.app.name} service...")

            # Prepare environment variables
            env_vars_dict = {'UPLOADED_PROGRAM_FILE': remote_launch_path}
            if self.app.remote_port:
                env_vars_dict["PORT"] = self.app.remote_port
                

            # Add program file path if provided
            if self.app.program_file:
                env_vars_dict["PROGRAM_FILE"] = self.app.program_file
            
            # loop over self.app.params and add them to env_vars_dict, if duplicate, don't add and warn
            for key, value in self.app.params.items():
                if key in env_vars_dict:
                    logger.warning(
                        f"Skipping duplicated env var. Existing ({key}): {env_vars_dict[key]}, New: {value}")
                else:
                    env_vars_dict[key] = value

            # Add additional parameters from app.params
            env_vars = " ".join([
                f"{key}=\"{value}\"" for key, value in env_vars_dict.items()
            ])

            # Prepare final command arguments with environment variables
            # Use sudo -E to preserve environment variables or pass them directly to sudo
            command_args = [
                f"chmod +x {remote_launch_path}",
                f"export {env_vars}",
                f"sudo -E {remote_launch_path}"
            ]

            result = self.session_manager.execute_command(
                instance_id=self.instance_id,
                commands=command_args
            )

            # Check if the service was started successfully
            if result.get('Status') == 'Success':
                logger.info(f"{self.app.name} service started successfully.")
                return True
            else:
                logger.error(f"Failed to start {self.app.name} service.")
                return False
        except Exception as e:
            logger.error(f"Error setting up {self.app.name} service: {str(e)}")
            return False

    def start_port_forwarding_monitoring(self, check_interval: int = 5, max_retries: int = 10) -> None:
        """
        Start monitoring threads for each port forwarding. Each thread will periodically check
        if the port forwarding is working by attempting to establish a socket connection to the
        forwarded port, and restart it if needed.

        Args:
            check_interval (int): Time between connection checks in seconds
            max_retries (int): Maximum number of consecutive retries before giving up
        """
        if not self.port_forwardings:
            logger.warning("No port forwardings to monitor")
            return

        logger.info(f"Starting port forwarding monitoring for {len(self.port_forwardings)} ports")

        for description, port_forwarding in self.port_forwardings.items():
            # Find the corresponding port mapping to get the local port
            port_mapping = next(
                (pm for pm in self.app.port_mappings if pm.get("description") == description),
                self.app.port_mappings[0]  # Default to first mapping if not found
            )
            local_port = port_mapping["local_port"]
            
            # Start a monitoring thread for each port forwarding
            thread = threading.Thread(
                target=self._monitor_port_forwarding,
                args=(description, port_forwarding, local_port, check_interval, max_retries),
                daemon=True
            )
            thread.start()
            logger.info(f"Started monitoring thread for {description} (port {local_port})")

    def _check_port_connection(self, port: int, timeout: int = 5) -> bool:
        """
        Check if a connection can be established to the given port.

        Args:
            port (int): Port to check
            timeout (int): Connection timeout in seconds

        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Try to connect to localhost:port
            result = sock.connect_ex(('localhost', port))
            
            # Close the socket
            sock.close()
            
            # If result is 0, the connection was successful
            return result == 0
        except Exception as e:
            logger.debug(f"Error checking port {port} connection: {str(e)}")
            return False

    def _monitor_port_forwarding(self, description: str, port_forwarding: Dict[str, Any], 
                                local_port: int, check_interval: int, max_retries: int) -> None:
        """
        Monitor a port forwarding and restart it if needed.

        Args:
            description (str): Description of the port forwarding
            port_forwarding (Dict[str, Any]): Port forwarding information
            local_port (int): Local port to check
            check_interval (int): Time between connection checks in seconds
            max_retries (int): Maximum number of consecutive retries before giving up
        """
        consecutive_failures = 0
        
        while not self._should_stop:
            try:
                # Sleep first to avoid immediate checking after setup
                time.sleep(check_interval)
                
                # Early exit if cleanup has been initiated
                if self._should_stop:
                    logger.info(f"Stopping port forwarding monitor for {description} (port {local_port})")
                    break
                
                # Check if the port is accessible by attempting to establish a connection
                is_connected = self._check_port_connection(local_port)
                
                if is_connected:
                    # Reset failure counter on success
                    if consecutive_failures > 0:
                        logger.info(f"Port forwarding for {description} (port {local_port}) is working again after {consecutive_failures} failures")
                        consecutive_failures = 0
                    continue
                
                # If we get here, the connection check failed
                consecutive_failures += 1
                logger.warning(f"Connection check failed for {description} (port {local_port}) (failure {consecutive_failures}/{max_retries})")
                
                # Early exit if cleanup has been initiated
                if self._should_stop:
                    logger.info(f"Stopping port forwarding monitor for {description} (port {local_port}) after connection check")
                    break
                
                # Check if the port forwarding process is already closed
                process_closed = False
                if 'process' in port_forwarding:
                    try:
                        # Check if process has terminated
                        if port_forwarding['process'].poll() is not None:
                            logger.warning(f"Port forwarding process for {description} (port {local_port}) has already terminated")
                            process_closed = True
                    except Exception as e:
                        logger.error(f"Error checking port forwarding process status: {str(e)}")
                
                # Early exit if cleanup has been initiated
                if self._should_stop:
                    logger.info(f"Stopping port forwarding monitor for {description} (port {local_port}) after process check")
                    break
                
                # If process is already closed or we've reached max retries, reconnect
                if process_closed or consecutive_failures >= max_retries:
                    logger.error(f"Port forwarding for {description} (port {local_port}) failed {consecutive_failures} times, restarting...")
                    
                    # Terminate the existing port forwarding process
                    if 'process' in port_forwarding:
                        try:
                            port_forwarding['process'].terminate()
                            logger.info(f"Terminated existing port forwarding process for {description}")
                        except Exception as e:
                            logger.error(f"Error terminating port forwarding process: {str(e)}")
                    
                    # Close the log file if it exists
                    if 'log_file' in port_forwarding and port_forwarding['log_file']:
                        try:
                            port_forwarding['log_file'].close()
                        except Exception as e:
                            logger.error(f"Error closing log file: {str(e)}")
                    
                    # One last check before restarting
                    if self._should_stop:
                        logger.info(f"Stopping port forwarding monitor for {description} (port {local_port}) after cleanup")
                        break
                    
                    # Extract port information from the port mapping
                    port_mapping = next(
                        (pm for pm in self.app.port_mappings if pm.get("description") == description),
                        self.app.port_mappings[0]  # Default to first mapping if not found
                    )
                    remote_port = port_mapping["remote_port"]
                    
                    # Restart port forwarding
                    try:
                        new_port_forwarding = self.session_manager.setup_port_forwarding(
                            instance_id=self.instance_id,
                            remote_port=remote_port,
                            local_port=local_port
                        )
                        
                        # Update the port_forwardings dictionary with the new process and log file
                        self.port_forwardings[description] = new_port_forwarding
                        logger.info(f"Restarted port forwarding for {description} (port {local_port})")
                        
                        # Reset failure counter
                        consecutive_failures = 0
                    except Exception as e:
                        logger.error(f"Error restarting port forwarding for {description}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in port forwarding monitoring for {description}: {str(e)}")
                # Sleep a bit to avoid tight loop in case of persistent errors
                time.sleep(5)
        
        logger.info(f"Port forwarding monitor for {description} (port {local_port}) has stopped")

    def setup_port_forwarding(self) -> Dict[str, Any]:
        """
        Set up port forwarding to the EC2 instance using SessionManager for all port mappings in the app.

        Returns:
            Dict containing the process and log file information for all port mappings
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return None

        try:
            # Set up port forwarding for all port mappings
            port_forwardings = {}

            for i, mapping in enumerate(self.app.port_mappings):
                remote_port = mapping["remote_port"]
                local_port = mapping["local_port"]
                description = mapping.get("description", f"Port mapping {i+1}")

                logger.info(
                    f"Setting up port forwarding for {description}: localhost:{local_port} -> {remote_port}")

                # Use the SessionManager's setup_port_forwarding method
                port_forwarding = self.session_manager.setup_port_forwarding(
                    instance_id=self.instance_id,
                    remote_port=remote_port,
                    local_port=local_port
                )

                # Store in the dictionary with the description as the key
                port_forwardings[description] = port_forwarding

            logger.info(
                f"Set up port forwarding for {len(port_forwardings)} ports")
            self.port_forwardings = port_forwardings
            return port_forwardings
        except Exception as e:
            logger.error(f"Error setting up port forwarding: {str(e)}")
            return None

    def tail_service_logs(self, in_thread=False) -> Optional[threading.Thread]:
        """
        Tail the logs of the application service using the configured log command.
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
                logger.info(f"Tailing {self.app.name} service logs...")

                # Create a log file with a unique name
                log_file_path = f"/tmp/{self.app.name}_logs_{self.instance_id}_{int(time.time())}.log"
                log_file = open(log_file_path, "w")

                # Construct the AWS SSM command to directly execute the log command
                ssm_command = [
                    "aws", "ssm", "start-session",
                    "--target", self.instance_id,
                    "--region", self.region_name,
                    "--document-name", "AWS-StartInteractiveCommand",
                    "--parameters", f"command=['{self.app.log_command}']"
                ]

                # Set up environment variables for the subprocess
                env = os.environ.copy()

                # Copy RACE_ prefixed AWS credentials to standard AWS environment variables
                if os.environ.get("RACE_AWS_ACCESS_KEY_ID"):
                    env["AWS_ACCESS_KEY_ID"] = os.environ.get(
                        "RACE_AWS_ACCESS_KEY_ID")
                if os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
                    env["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
                        "RACE_AWS_SECRET_ACCESS_KEY")
                if os.environ.get("RACE_AWS_SESSION_TOKEN"):
                    env["AWS_SESSION_TOKEN"] = os.environ.get(
                        "RACE_AWS_SESSION_TOKEN")
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

                logger.info(
                    f"To view {self.app.name} logs in real-time, run: tail -f {log_file_path}")

                # Wait for the process to complete
                process.wait()

                # Close the log file
                log_file.close()

                # Check return code
                if process.returncode != 0:
                    logger.error(
                        f"Log tailing process exited with code {process.returncode}")
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
            logger.info(
                f"Started {self.app.name} log tailing in a separate thread")
            return log_thread
        else:
            # Run in the main thread
            try:
                _tail_logs()
            except KeyboardInterrupt:
                logger.info("Log tailing interrupted.")
            return None

    def wait_for_app_ready(self, max_wait_time: int = 900, check_interval: int = 5) -> bool:
        """
        Poll the application health endpoint until it returns a successful response or times out.

        Args:
            max_wait_time (int): Maximum time to wait in seconds
            check_interval (int): Time between health checks in seconds

        Returns:
            bool: True if the application is ready, False otherwise
        """
        if not self.instance_id:
            logger.error("No instance ID available. Deploy the stack first.")
            return False

        # Use the app's wait_for_ready method
        try:
            return self.app.wait_for_ready(max_wait_time=max_wait_time, check_interval=check_interval)
        except Exception as e:
            logger.error(f"Error waiting for app to be ready: {str(e)}")
            return False

    def test_app(self) -> bool:
        """
        Send a test request to the application
        """
        # Use the app's test_request method
        return self.app.test_request()

    def register_socket(self):
        """
        Register a socket at /tmp/ec2_app_{deployer_id}_port{port} that listens for 'kill-stack' commands.
        When 'kill-stack' is received, the deployer will clean up everything.

        Returns:
            bool: True if socket was successfully registered, False otherwise
        """
        try:
            # Create a Unix domain socket
            self.socket_server = socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM)

            # Update socket path to include app type and port information
            self.socket_path = f"/tmp/ec2_app_{self.deployer_id}_app{self.app.name}_port{self.app.local_port}"

            # Remove the socket file if it already exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            # Bind the socket to the path
            self.socket_server.bind(self.socket_path)

            # Listen for connections
            self.socket_server.listen(1)

            logger.info(f"Registered socket at {self.socket_path}")

            # Start a thread to handle socket connections
            self._should_stop = False
            self.socket_thread = threading.Thread(
                target=self._socket_listener, daemon=True)
            self.socket_thread.start()

            return True
        except Exception as e:
            logger.error(f"Error registering socket: {str(e)}")
            return False

    def _socket_listener(self):
        """
        Listen for connections on the socket and handle 'kill-stack' commands.
        This method runs in a separate thread.
        """
        logger.info(f"Socket listener started for {self.socket_path}")

        # Set a timeout so we can check _should_stop periodically
        self.socket_server.settimeout(1.0)

        while not self._should_stop:
            try:
                # Accept a connection
                conn, _ = self.socket_server.accept()

                # Receive data
                data = conn.recv(1024).decode('utf-8').strip()
                logger.info(f"Received command on socket: {data}")

                # Check if the command is 'kill-stack'
                if data == 'kill-stack':
                    logger.info(
                        "Received kill-stack command, cleaning up resources...")
                    conn.sendall(b"Cleaning up resources...\n")
                    conn.close()

                    # Clean up in the main thread to avoid threading issues
                    cleanup_thread = threading.Thread(
                        target=self.cleanup, daemon=True)
                    cleanup_thread.start()
                    cleanup_thread.join()  # Wait for cleanup to finish
                    sys.exit(0)  # Exit after cleanup completes
                else:
                    # Send a response for unknown commands
                    conn.sendall(f"Unknown command: {data}\n".encode('utf-8'))
                    conn.close()
            except socket.timeout:
                # This is expected due to the timeout we set
                pass
            except Exception as e:
                if not self._should_stop:  # Only log if we're not intentionally stopping
                    logger.error(f"Error in socket listener: {str(e)}")
                time.sleep(1)  # Avoid tight loop in case of persistent errors

        logger.info("Socket listener stopped")

    def _cleanup_socket(self):
        """
        Clean up the socket resources.
        """
        try:
            # Signal the socket thread to stop
            self._should_stop = True

            # Close the socket server if it exists
            if self.socket_server:
                self.socket_server.close()

            # Remove the socket file if it exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            logger.info(f"Cleaned up socket at {self.socket_path}")
        except Exception as e:
            logger.error(f"Error cleaning up socket: {str(e)}")

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            EC2Deployer: The deployer instance
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
        # Signal all monitoring threads to stop
        self._should_stop = True
        
        # Record the stop time
        self.stop_time = datetime.datetime.now()

        # Calculate and report the runtime and cost
        if self.start_time:
            runtime = self.stop_time - self.start_time
            hours, remainder = divmod(runtime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            logger.info("=" * 60)
            logger.info("STACK TIMING AND COST REPORT")
            logger.info("=" * 60)
            logger.info(
                f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(
                f"Stop time: {self.stop_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(
                f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")

            # Calculate and report cost
            try:
                duration_seconds = (
                    self.stop_time - self.start_time).total_seconds()
                cost_result = aws_costs.calculate_cost(
                    instance_type=self.instance_type,
                    region_code=self.region_name,
                    duration_seconds=duration_seconds
                )
                if cost_result:
                    logger.info(
                        f"Estimated cost: ${cost_result.total_cost:.2f}")
                    logger.info(
                        f"Hourly rate: ${cost_result.hourly_price:.4f}/hour")
                else:
                    logger.warning(
                        "Could not calculate cost - price information not available")
            except Exception as e:
                logger.error(f"Error calculating cost: {str(e)}")

            logger.info("=" * 60)

        self._cleanup_socket()

        # Clean up all port forwardings if we have a dictionary of them
        if hasattr(self, 'port_forwardings') and self.port_forwardings:
            logger.info("Stopping all port forwardings...")
            for description, port_forwarding in self.port_forwardings.items():
                try:
                    # Terminate the process
                    if 'process' in port_forwarding:
                        port_forwarding['process'].terminate()
                    # Close the log file if it exists
                    if 'log_file' in port_forwarding and port_forwarding['log_file']:
                        port_forwarding['log_file'].close()
                        logger.info(
                            f"{description} port forwarding logs are available at: {port_forwarding.get('log_file_path', 'unknown')}")
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up port forwarding for {description}: {str(e)}")

        if self.stack_name:
            try:
                # Check if stack exists before attempting to delete
                try:
                    self.cf_client.describe_stacks(StackName=self.stack_name)
                    stack_exists = True
                except ClientError as e:
                    if "does not exist" in str(e):
                        logger.info(
                            f"Stack {self.stack_name} does not exist, no cleanup needed.")
                        stack_exists = False
                    else:
                        # Re-raise if it's a different error
                        raise

                if stack_exists:
                    logger.info(
                        f"Deleting CloudFormation stack: {self.stack_name}")
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
                    logger.error(
                        f"Session token issue detected during cleanup: {str(e)}")
                    logger.error(
                        f"IMPORTANT: Please manually delete the CloudFormation stack '{stack_name}' in region '{region}'")
                    logger.error(
                        f"Visit: https://{region}.console.aws.amazon.com/cloudformation/home?region={region}#/stacks")
                    logger.error(
                        "Find your stack in the list, select it, and click 'Delete' from the Actions menu")
                else:
                    logger.error(f"Error deleting stack: {str(e)}")
            except Exception as e:
                logger.error(f"Error deleting stack: {str(e)}")
                logger.error(
                    f"If cleanup failed due to session issues, please manually delete the stack '{self.stack_name}'")
                logger.error(
                    f"Visit: https://{self.region_name}.console.aws.amazon.com/cloudformation/home?region={self.region_name}#/stacks")
        else:
            logger.info(f"IMPORTANT: Please manually delete the stack at: https://{self.region_name}.console.aws.amazon.com/cloudformation/home?region={self.region_name}#/stacks")

# Global variable to store the deployer instance
deployer = None


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
    global deployer
    if deployer is not None:
        deployer.cleanup()


def list_ec2_app_sockets() -> List[str]:
    """
    List all socket files matching the pattern /tmp/ec2_app_*

    Returns:
        List[str]: List of socket file paths
    """
    socket_files = glob.glob("/tmp/ec2_app_*")
    return socket_files


def extract_port_from_socket_path(socket_path: str) -> Optional[int]:
    try:
        # Extract port from socket path using regex
        import re
        match = re.search(r'_port(\d+)$', socket_path)
        if match:
            return int(match.group(1))
        return None
    except Exception as e:
        logger.error(f"Error extracting port from socket path: {str(e)}")
        return None


def extract_app_type_from_socket_path(socket_path: str) -> Optional[str]:
    """
    Extract app type from socket path.

    Args:
        socket_path (str): Path to the socket file

    Returns:
        Optional[str]: App type if found, None otherwise
    """
    try:
        # Extract app type from socket path using regex
        import re
        match = re.search(r'_app([^_]+)', socket_path)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        logger.error(f"Error extracting app type from socket path: {str(e)}")
        return None


def find_socket_by_id(deployer_id: str) -> List[str]:
    socket_files = glob.glob(f"/tmp/ec2_app_{deployer_id}*")
    return socket_files


def stop_instances(deployer_id: Optional[str] = None, app_type: Optional[str] = None) -> None:
    """
    Stop running EC2 app instances.

    Args:
        deployer_id (Optional[str]): ID of the deployer to stop. If None, stops all instances.
        app_type (Optional[str]): Type of application to stop (vllm, mini-tgi). If None, stops all app types.
    """
    # Get the list of socket files based on deployer_id
    if deployer_id is None:
        socket_files = list_ec2_app_sockets()
    else:
        socket_files = find_socket_by_id(deployer_id)
    
    if not socket_files:
        logger.info(f"No running EC2 app instances found{' for deployer ID ' + deployer_id if deployer_id else ''}.")
        return
    
    # Filter socket files by app_type if specified
    if app_type is not None:
        filtered_socket_files = []
        for socket_file in socket_files:
            socket_app_type = extract_app_type_from_socket_path(socket_file)
            if socket_app_type == app_type:
                filtered_socket_files.append(socket_file)
        
        socket_files = filtered_socket_files
        
        if not socket_files:
            logger.info(f"No running {app_type} instances found{' for deployer ID ' + deployer_id if deployer_id else ''}.")
            return
    
    # Log the found instances
    logger.info(f"Found {len(socket_files)} running EC2 app instances{' for deployer ID ' + deployer_id if deployer_id else ''}{' of type ' + app_type if app_type else ''}:")
    for socket_file in socket_files:
        port = extract_port_from_socket_path(socket_file)
        app_type_info = extract_app_type_from_socket_path(socket_file)
        info = f" (type: {app_type_info}, port: {port})" if app_type_info and port else f" (port: {port})" if port else ""
        logger.info(f"  {socket_file}{info}")
    
    # Send kill command to all matching sockets
    logger.info(f"Sending kill command to {len(socket_files)} instances...")
    for socket_file in socket_files:
        send_kill_command_to_socket(socket_file)
    
    logger.info(f"All matching instances have been instructed to shut down.")


def wait_for_app_instances(deployer_id: Optional[str] = None, app_type: Optional[str] = None, check_interval: int = 5) -> bool:
    """
    Wait for app instances to be ready by polling test_app until it returns true.

    Args:
        deployer_id (Optional[str]): ID of the deployer to wait for. If None, waits for any instance.
        app_type (Optional[str]): Type of application to wait for (vllm, mini-tgi). If None, uses generic health check.
        check_interval (int): Time between checks in seconds

    Returns:
        bool: True if an instance is ready, False otherwise
    """
    logger.info(
        f"Waiting for {'any' if deployer_id is None else deployer_id} app instance{' of type ' + app_type if app_type else ''} to be ready...")
    stack_info_printed = False

    while True:
        # List all socket files or specific instance sockets
        if deployer_id is None:
            socket_files = list_ec2_app_sockets()
        else:
            socket_files = find_socket_by_id(deployer_id)

        if not socket_files:
            logger.info(
                f"No {'any' if deployer_id is None else deployer_id} app instances found. Retrying in {check_interval} seconds...")
            time.sleep(check_interval)
            continue

        # Filter socket files by app_type if specified
        if app_type is not None:
            filtered_socket_files = []
            for socket_file in socket_files:
                socket_app_type = extract_app_type_from_socket_path(socket_file)
                if socket_app_type == app_type:
                    filtered_socket_files.append(socket_file)
            
            socket_files = filtered_socket_files
            
            if not socket_files:
                logger.info(f"No running {app_type} instances found{' for deployer ID ' + deployer_id if deployer_id else ''}. Retrying in {check_interval} seconds...")
                time.sleep(check_interval)
                continue

        # Try to test each instance
        for socket_file in socket_files:
            try:
                # Extract port from socket path
                port = extract_port_from_socket_path(socket_file)
                if port is None:
                    logger.warning(
                        f"Could not extract port from socket path: {socket_file}")
                    continue

                socket_app_type = extract_app_type_from_socket_path(socket_file)
                logger.info(f"Testing app{' of type ' + socket_app_type if socket_app_type else ''} at port {port}...")

                # Create the appropriate EC2App instance based on app_type
                app = None
                if app_type == "vllm" or (app_type is None and socket_app_type == "vllm"):
                    app = create_vllm_app(local_port=port)
                elif app_type == "mini-tgi" or (app_type is None and socket_app_type == "mini-tgi"):
                    app = create_mini_tgi_app(local_port=port)

                if app:
                    # Use the app's test_request method to check if it's ready
                    try:
                        if app.test_request():
                            logger.info(
                                f"App instance at port {port} is ready!")
                            return True
                        else:
                            logger.debug(
                                f"App at port {port} not ready (test_request returned False)")
                    except Exception as e:
                        logger.debug(f"App at port {port} not ready: {str(e)}")
                else:
                    # Fallback to generic health check if app_type is not specified and socket_app_type is not recognized
                    try:
                        import requests
                        response = requests.get(
                            f"http://localhost:{port}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info(
                                f"App instance at port {port} is ready!")
                            return True
                    except Exception as e:
                        logger.debug(f"App at port {port} not ready: {str(e)}")

            except Exception as e:
                logger.debug(f"Error testing app at {socket_file}: {str(e)}")

        # Wait before checking again
        logger.info(
            f"No ready app instances found. Retrying in {check_interval} seconds...")
        time.sleep(check_interval)


def send_kill_command_to_socket(socket_path: str) -> bool:
    """
    Send a kill-stack command to a specific socket.

    Args:
        socket_path (str): Path to the socket file

    Returns:
        bool: True if the command was sent successfully, False otherwise
    """
    try:
        if not os.path.exists(socket_path):
            logger.error(f"Socket file {socket_path} does not exist")
            return False

        # Create a socket client
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Connect to the socket
        logger.info(f"Connecting to socket {socket_path}...")
        client.connect(socket_path)

        # Send the kill-stack command
        logger.info(f"Sending kill-stack command to {socket_path}...")
        client.sendall(b"kill-stack")

        # Receive the response
        response = client.recv(1024).decode('utf-8').strip()
        logger.info(f"Response from socket: {response}")

        # Close the socket
        client.close()

        return True
    except Exception as e:
        logger.error(
            f"Error sending kill command to socket {socket_path}: {str(e)}")
        return False


# These functions have been moved to ec2_app.py


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
    command_desc = """Deploy an application on EC2 using CloudFormation.
Before deploying EC2 app stack, make sure that your environment variables (or .env) include RACE_AWS_ aws access keys."""
    parser = argparse.ArgumentParser(description=command_desc)
    parser.add_argument("--app-name", type=str, choices=["vllm", "mini-tgi"], required=True,
                        help="Name of application to deploy (REQUIRED)")
    parser.add_argument("--model-id", type=str, default="tiiuae/falcon3-10b-instruct",
                        help="Hugging Face model ID to deploy (for vLLM)")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region name")
    parser.add_argument("--instance-type", type=str, default="g6e.4xlarge",
                        help="EC2 instance type, see https://aws.amazon.com/ec2/instance-types/")
    parser.add_argument("--stack-name", type=str, default=None,
                        help="CloudFormation stack name")
    parser.add_argument("--local-port", type=int, default=None,
                        help="Local port for port forwarding (uses app's default if not specified)")
    parser.add_argument("--ami-id", type=str, default="ami-04f4302ff68e424cf",
                        help="AMI ID to use for the EC2 instance (Deep Learning OSS Nvidia Driver AMI)")
    parser.add_argument("--stage-name", type=str, default="prod",
                        help="Stage name for deployment (used for resource naming)")
    parser.add_argument("--port-forward", action="store_true", default=True,
                        help="Set up port forwarding to the EC2 instance")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Send a test request to the app after deployment")
    parser.add_argument("--wait-time", type=int, default=1800,
                        help="Timeout for retrying app service after it starts until it's fully ready")
    parser.add_argument("--id", type=str, default=None,
                        help="Custom ID for this deployer (used for deployer identification)")
    parser.add_argument("--stop", type=str, default=None, nargs='?', const='all', metavar="ID",
                        help="Stop running EC2 app instances. If no ID is provided, stops all instances. If ID is provided, stops only that instance.")
    parser.add_argument("--wait", type=str, default=None, nargs='?', const='all', metavar="ID",
                        help="Wait for app deployer to be ready. If no ID is provided, waits for any instance. If Deployer ID is provided, waits only for that deployer.")
    parser.add_argument("--param", action="append", metavar="KEY=VALUE",
                        help="Additional parameters to pass to the application. Can be specified multiple times. Use --param-help for details.")
    parser.add_argument("--param-help", action="store_true",
                        help="Show detailed help for available parameters for each app type")
    parser.add_argument("--connect", type=str, default=None,
                        help="Connect to an existing EC2 instance with the specified instance ID, skipping deployment and setup steps")

    args = parser.parse_args()

    # Handle the --stop argument if provided (if provided but without id, it will be all, if not provided, it will be None)
    if args.stop is not None:
        deployer_id = None if args.stop == 'all' else args.stop
        stop_instances(deployer_id, app_type=args.app_name)
        sys.exit(0)

    # Handle the --wait argument if provided (if provided but without id, it will be all, if not provided, it will be None)
    if args.wait is not None:
        deployer_id = None if args.wait == 'all' else args.wait
        wait_for_app_instances(
            deployer_id, app_type=args.app_name, check_interval=5)
        sys.exit(0)

    # Register signal handlers for various termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    signal.signal(signal.SIGHUP, signal_handler)   # Terminal closed

    # Register atexit handler to ensure cleanup on exit
    atexit.register(cleanup_on_exit)

    # Show detailed parameter help if requested
    if args.param_help:
        print("\nDetailed Parameter Help:")
        print("\nFor vLLM app:")
        print("  --param MODEL_ID=tiiuae/falcon3-10b-instruct  # Hugging Face model ID")
        print("  --param TENSOR_PARALLEL=2                     # Number of GPUs for tensor parallelism (0 for auto)")
        print("  --param MAX_NUM_BATCHED_TOKENS=2048           # Maximum number of tokens to batch")
        print("  --param GPU_MEMORY_UTILIZATION=0.95           # GPU memory utilization (0.0-1.0)")
        print("  --param ENABLE_CHUNKED_PREFILL=true           # Enable chunked prefill (true/false)")
        print("\nFor mini-TGI app:")
        print("  --param MODEL_ID=tiiuae/falcon3-10b-instruct  # Hugging Face model ID")
        print("  --param MAX_BATCH_SIZE=64                     # Maximum batch size")
        print("\nExample usage:")
        print("  python scripts/aws/deploy_ec2_llm.py --app-name vllm --param MODEL_ID=meta-llama/Llama-2-7b-chat-hf --param TENSOR_PARALLEL=2")
        sys.exit(0)

    # Parse parameters from --param arguments
    params = {}
    if args.param:
        for param in args.param:
            try:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
            except ValueError:
                logger.warning(
                    f"Invalid parameter format: {param}. Expected format: key=value")

    # Add model_id to params if specified
    if args.model_id:
        params['MODEL_ID'] = args.model_id

    # Create the appropriate EC2App based on the app name
    if args.app_name == "vllm":
        app = create_vllm_app(local_port=args.local_port, params=params)
    elif args.app_name == "mini-tgi":
        app = create_mini_tgi_app(local_port=args.local_port, params=params)
    else:
        logger.error(f"Unsupported app name: {args.app_name}")
        sys.exit(1)

    # Create deployer using context manager pattern
    deployer = None  # Initialize deployer variable for signal handler
    with EC2Deployer(
        app=app,
        region_name=args.region,
        instance_type=args.instance_type,
        stack_name=args.stack_name,
        ami_id=args.ami_id,
        deployer_id=args.id
    ) as deployer:
        # Register the socket to listen for kill-stack commands
        deployer.register_socket()
        try:
            # Check if we're connecting to an existing instance
            if args.connect:
                logger.info(f"Connecting to existing instance: {args.connect}")
                # Set the instance ID directly
                deployer.instance_id = args.connect
                
                # Record the start time for cost calculation
                deployer.start_time = datetime.datetime.now()
                
                # Wait for the instance to be ready for SSM connections
                if not deployer.wait_for_instance_ssm_ready():
                    logger.error(
                        "Instance not ready for SSM connections. Exiting.")
                    raise Exception("Instance not ready for SSM connections")

                # Connect to the instance
                if not deployer.connect_to_instance():
                    logger.error("Failed to connect to instance. Exiting.")
                    raise Exception("Failed to connect to instance")
                
                logger.info(f"Successfully connected to existing instance: {args.connect}")
            else:
                # Deploy a new stack
                deployment = deployer.deploy()

                # Wait for the instance to be ready for SSM connections
                if not deployer.wait_for_instance_ssm_ready():
                    logger.error(
                        "Instance not ready for SSM connections. Exiting.")
                    raise Exception("Instance not ready for SSM connections")

                # Connect to the instance
                if not deployer.connect_to_instance():
                    logger.error("Failed to connect to instance. Exiting.")
                    raise Exception("Failed to connect to instance")

                # Install and run app service
                if not deployer.run_app_service():
                    logger.error(f"Failed to run {app.name} service. Exiting.")
                    raise Exception(f"Failed to run {app.name} service")

            # Start log tailing in a separate thread
            log_thread = deployer.tail_service_logs(in_thread=True)

            # Set up port forwarding if requested
            port_forwardings = None
            if args.port_forward:
                port_forwardings = deployer.setup_port_forwarding()

                # Send a test request to the app if requested
                if args.test:
                    deployer.wait_for_app_ready(max_wait_time=args.wait_time)
                    # Start monitoring port forwarding after the app is ready
                    deployer.start_port_forwarding_monitoring()
                    
                    # Send a test request to the app
                    deployer.test_app()

            # Keep the main thread running until interrupted
            try:
                logger.info("Press Ctrl+C to stop...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
            finally:
                # Port forwarding cleanup is handled by the deployer's cleanup method
                pass
        except KeyboardInterrupt:
            logger.info("Deployment interrupted.")
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            # Context manager will handle cleanup

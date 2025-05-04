#!/usr/bin/env python3
"""
EC2App class for deploying applications on EC2 instances.
"""
import os
import requests
import time
from typing import Dict, Any, Optional, List, TypedDict, Union
from pathlib import Path
from utils.logging_utils import get_logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize logger
logger = get_logger("ec2_app")


class PortMapping(TypedDict):
    """
    TypedDict for port mapping configuration.
    """
    remote_port: int
    local_port: int
    description: Optional[str]


class EC2App:
    """
    Class representing an application to be deployed on EC2.
    """

    def __init__(
        self,
        name: str,
        setup_script: str,
        launch_script: str,
        program_file: Optional[str] = None,
        port_mappings: List[PortMapping] = None,
        log_command: str = None,
        default_remote_port: int = 8000,
        default_local_port: int = 8987,
        api_key: str = None,
        test_request_function: Optional[callable] = None,
        params: Dict[str, str] = None,
    ):
        """
        Initialize an EC2App.

        Args:
            name: Name of the application
            setup_script: Path to the setup script
            launch_script: Path to the launch script
            program_file: Optional path to a program file to upload
            port_mappings: List of port mappings
            log_command: Command to view logs (e.g., "journalctl -u service.name -f")
            default_remote_port: Default remote port if not specified in port_mappings
            default_local_port: Default local port if not specified in port_mappings
            api_key: API key for the application
            test_request_function: Function to test the application
            params: Dictionary of additional parameters to pass to the application
        """
        self.name = name
        self.setup_script = setup_script
        self.launch_script = launch_script
        self.program_file = program_file
        self.port_mappings = port_mappings or [
            {
                "remote_port": default_remote_port,
                "local_port": default_local_port,
                "description": "Main application port"
            }
        ]
        self.log_command = log_command or f"echo 'No log command specified for {name}'"
        self.api_key = api_key or os.environ.get(f"{name.upper()}_API_KEY", "")
        self.test_request_function = test_request_function
        self.params = params or {}

        # Validate that the scripts exist
        if not os.path.exists(self.setup_script):
            raise FileNotFoundError(
                f"Setup script not found: {self.setup_script}")
        if not os.path.exists(self.launch_script):
            raise FileNotFoundError(
                f"Launch script not found: {self.launch_script}")
        if self.program_file and not os.path.exists(self.program_file):
            raise FileNotFoundError(
                f"Program file not found: {self.program_file}")
        logger.info(f"EC2App name: {self.name}")
        logger.info(f"EC2App setup_script: {self.setup_script}")
        logger.info(f"EC2App launch_script: {self.launch_script}")
        logger.info(f"EC2App program_file: {self.program_file}")
        logger.info(f"EC2App port_mappings: {self.port_mappings}")
        logger.info(f"EC2App log_command: {self.log_command}")
        logger.info(f"EC2App default_remote_port: {default_remote_port}")
        logger.info(f"EC2App default_local_port: {default_local_port}")
        logger.info(f"EC2App api_key: {self.api_key}")
        logger.info(f"EC2App params", params=self.params)

    @property
    def primary_port_mapping(self) -> PortMapping:
        """
        Get the primary port mapping (first in the list).
        """
        return self.port_mappings[0]

    @property
    def remote_port(self) -> int:
        """
        Get the primary remote port.
        """
        return self.primary_port_mapping["remote_port"]

    @property
    def local_port(self) -> int:
        """
        Get the primary local port.
        """
        return self.primary_port_mapping["local_port"]

    def print_info(self, stack_name: str, region_name: str, instance_type: str, ami_id: str):
        """
        Print key information about the application and deployment.

        Args:
            stack_name: CloudFormation stack name
            region_name: AWS region name
            instance_type: EC2 instance type
            ami_id: AMI ID
        """
        logger.info("=" * 60)
        logger.info("EC2 APP INFORMATION")
        logger.info("=" * 60)

        # Basic stack information
        logger.info(f"Stack Name", stack_name=stack_name)
        logger.info(f"AWS Region", region=region_name)
        logger.info(f"Instance Type", instance_type=instance_type)
        logger.info(f"AMI ID", ami_id=ami_id)
        logger.info(f"Application", app_name=self.name)

        # Print port mappings
        for port_mapping in self.port_mappings:
            desc = port_mapping.get("description", "Port mapping")
            logger.info(f"{desc}",
                        local_endpoint=f"http://localhost:{port_mapping['local_port']}")

        logger.info("=" * 60)

    def wait_for_ready(self, max_wait_time: int = 900, check_interval: int = 5, verbose: bool = True) -> bool:
        """
        Poll the application health endpoint until it returns a successful response or times out.

        Args:
            max_wait_time (int): Maximum time to wait in seconds
            check_interval (int): Time between health checks in seconds
            verbose (bool): Whether to log info messages (default: True)

        Returns:
            bool: True if the application is ready, False otherwise
        """
        health_url = f"http://localhost:{self.local_port}/health"
        if verbose:
            logger.info(
                f"Waiting for {self.name} to be ready (polling {health_url} )...")

        # Prepare headers with API key if provided
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            try:
                response = requests.get(health_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    if verbose:
                        logger.info(f"{self.name} is ready!")
                    return True
                else:
                    logger.debug(
                        f"{self.name} not ready yet. Status code: {response.status_code}")
            except requests.exceptions.RequestException:
                logger.debug(f"{self.name} health check failed, retrying...")

            # Wait before checking again
            time.sleep(check_interval)

        if verbose:
            logger.error(
                f"Timed out waiting for {self.name} to be ready after {max_wait_time} seconds")
        return False

    def test_request(self) -> bool:
        """
        Send a test request to the application

        Returns:
            bool: True if the test was successful, False otherwise
        """
        if self.test_request_function:
            return self.test_request_function(self)
        else:
            logger.warning(f"No test request function defined for {self.name}")
            return False


def test_vllm_request(app: EC2App) -> bool:
    """
    Send a test request to the vLLM application

    Args:
        app: The EC2App instance

    Returns:
        bool: True if the test was successful, False otherwise
    """
    try:
        # Prepare the test request
        url = f"http://localhost:{app.local_port}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if provided
        if app.api_key:
            headers["Authorization"] = f"Bearer {app.api_key}"
        print(f"test_vllm_request API Key: {app.api_key}")
        print(f"test headers: {headers}")

        # Prepare a friendly welcome message using chat format
        data = {
            "model": app.params.get("MODEL_ID", "tiiuae/falcon3-10b-instruct"),
            "messages": [
                {"role": "system", "content": f"You are a helpful AI assistant deployed on AWS EC2 using {app.name}."},
                {"role": "user", "content": "Say hello and introduce yourself."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        logger.info(f"Sending test request to {app.name}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get(
                "message", {}).get("content", "").strip()

            logger.info(f"{app.name} Test Response:")
            logger.info("-" * 50)
            logger.info(generated_text)
            logger.info("-" * 50)
            logger.info(f"{app.name} is working correctly!")
            return True
        else:
            logger.error(
                f"Error testing {app.name}: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to {app.name}: {str(e)}")
        logger.info(
            f"The {app.name} might still be initializing. Try again in a few moments.")
        return False
    except Exception as e:
        logger.error(f"Error testing {app.name}: {str(e)}")
        return False


def create_vllm_app(local_port: Optional[int] = None, api_key: Optional[str] = None, params: Dict[str, str] = None) -> EC2App:
    """
    Create an EC2App for vLLM.

    Args:
        local_port: Local port for port forwarding
        api_key: API key for the application
        params: Dictionary of additional parameters to pass to the application.
               Supported parameters:
               - MODEL_ID: Hugging Face model ID (default: "tiiuae/falcon3-10b-instruct")
               - TENSOR_PARALLEL: Number of GPUs for tensor parallelism, 0 for auto-detect (default: 0)
               - MAX_NUM_BATCHED_TOKENS: Maximum number of tokens to batch (default: 2048)
               - GPU_MEMORY_UTILIZATION: GPU memory utilization (default: 0.95)
               - ENABLE_CHUNKED_PREFILL: Enable chunked prefill (default: "true")

    Returns:
        EC2App: The configured vLLM app
    """
    # Use default values if None is provided
    if local_port is None:
        local_port = 8987
    if api_key is None:
        api_key = os.environ.get("EC2_LLM_API_KEY", "random_key")
    # Define paths to scripts
    setup_script = str(Path(__file__).parent / "apps" /
                       "vllm" / "install_vllm.sh")
    launch_script = str(Path(__file__).parent / "apps" /
                        "vllm" / "setup_vllm_service.sh")
    if params is None:
        params = {}
    if 'API_KEY' not in params:
        params['API_KEY'] = api_key

    # Create the EC2App
    app = EC2App(
        name="vllm",
        setup_script=setup_script,
        launch_script=launch_script,
        program_file=None,  # vLLM doesn't need a program file
        default_remote_port=8000,
        default_local_port=local_port,
        log_command="sudo journalctl -u vllm.service -f",
        api_key=api_key,
        test_request_function=test_vllm_request,
        params=params
    )

    return app


def test_mini_tgi_request(app: EC2App) -> bool:
    """
    Send a test request to the mini-TGI application using the logits endpoint

    Args:
        app: The EC2App instance

    Returns:
        bool: True if the test was successful, False otherwise
    """
    try:
        # Use the logits endpoint as specified
        url = f"http://localhost:{app.local_port}/logits"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ec2_app/0.0.1"
        }

        # Add API key if provided
        if app.api_key:
            headers["Authorization"] = f"Bearer {app.api_key}"

        # Prepare the test data
        data = {
            "prompt": "User: Hello, nice to meet you, how are you doing?\nAssistant: ",
            "tokens": ["Yes", "No", "What", "Hello"]
        }

        logger.info(f"Sending test request to {app.name} logits endpoint...")
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()

            # Log the response details
            logger.info(f"{app.name} Test Response:")
            logger.info("-" * 50)
            logger.info(f"Logits: {result.get('logits', {})}")
            logger.info(f"Probabilities: {result.get('probabilities', {})}")
            logger.info(f"Next token: {result.get('next_token', '')}")
            logger.info("-" * 50)
            logger.info(f"{app.name} is working correctly!")
            return True
        else:
            logger.error(
                f"Error testing {app.name}: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to {app.name}: {str(e)}")
        logger.info(
            f"The {app.name} might still be initializing. Try again in a few moments.")
        return False
    except Exception as e:
        logger.error(f"Error testing {app.name}: {str(e)}")
        return False


def create_mini_tgi_app(local_port: Optional[int] = None, api_key: Optional[str] = None, params: Dict[str, str] = None) -> EC2App:
    """
    Create an EC2App for mini-TGI.

    Args:
        local_port: Local port for port forwarding
        api_key: API key for the application
        params: Dictionary of additional parameters to pass to the application.
               Supported parameters:
               - MODEL_ID: Hugging Face model ID (default: "tiiuae/falcon3-10b-instruct")
               - MAX_BATCH_SIZE: Maximum batch size (default: 8)
               - MAX_TOKENS: Maximum number of tokens to generate (default: 2048)

    Returns:
        EC2App: The configured mini-TGI app
    """
    # Use default values if None is provided
    if local_port is None:
        local_port = 8977
    if api_key is None:
        api_key = os.environ.get("EC2_LLM_API_KEY", "random_key")
    # Define paths to scripts
    setup_script = str(Path(__file__).parent / "apps" /
                       "mini_tgi" / "install_mini_tgi.sh")
    launch_script = str(Path(__file__).parent / "apps" /
                        "mini_tgi" / "setup_mini_tgi_service.sh")
    program_file = str(Path(__file__).parent / "apps" /
                       "mini_tgi" / "llm_server.py")

    # Create the EC2App
    app = EC2App(
        name="mini-tgi",
        setup_script=setup_script,
        launch_script=launch_script,
        program_file=program_file,
        default_remote_port=8000,
        default_local_port=local_port,
        log_command="sudo journalctl -u mini-tgi.service -f",
        api_key=api_key,
        test_request_function=test_mini_tgi_request,
        params=params
    )

    return app

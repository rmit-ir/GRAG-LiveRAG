#!/usr/bin/env python3
"""
vLLM app creation and testing functions for EC2 deployment.
"""
import os
import requests
import time
from pathlib import Path
from typing import Dict, Optional
from utils.logging_utils import get_logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("vllm_app")

# Import EC2App from parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ec2_app import EC2App


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

        # Prepare a friendly welcome message using chat format
        data = {
            "model": app.params.get("MODEL_ID", "tiiuae/falcon3-10b-instruct"),
            "messages": [
                {"role": "system", "content": f"You are a helpful AI assistant deployed on AWS EC2 using {app.name}."},
                {"role": "user", "content": "Say hello and introduce yourself."}
            ],
            "max_tokens": 500,
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
    setup_script = str(Path(__file__).parent / "setup_vllm.sh")
    launch_script = str(Path(__file__).parent / "launch_vllm.sh")
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

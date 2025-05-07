#!/usr/bin/env python3
"""
Mini-TGI app creation and testing functions for EC2 deployment.
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
logger = get_logger("mini_tgi_app")

# Import EC2App from parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ec2_app import EC2App


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
    setup_script = str(Path(__file__).parent / "install_mini_tgi.sh")
    launch_script = str(Path(__file__).parent / "setup_mini_tgi_service.sh")
    program_file = str(Path(__file__).parent / "llm_server.py")

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

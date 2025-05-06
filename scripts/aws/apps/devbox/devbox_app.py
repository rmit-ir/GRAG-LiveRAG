#!/usr/bin/env python3
"""
Devbox app creation and testing functions for EC2 deployment.
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
logger = get_logger("devbox_app")

# Import EC2App from parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ec2_app import EC2App


class DevboxApp(EC2App):
    """
    Custom EC2App class for devbox with FileBrowser.
    Overrides the wait_for_ready method to check the FileBrowser root URL instead of /health.
    """
    
    def wait_for_ready(self, max_wait_time: int = 900, check_interval: int = 5, verbose: bool = True) -> bool:
        """
        Poll the FileBrowser root URL until it returns a successful response or times out.

        Args:
            max_wait_time (int): Maximum time to wait in seconds
            check_interval (int): Time between health checks in seconds
            verbose (bool): Whether to log info messages (default: True)

        Returns:
            bool: True if the application is ready, False otherwise
        """
        root_url = f"http://localhost:{self.local_port}/"
        if verbose:
            logger.info(
                f"Waiting for {self.name} to be ready (polling {root_url} )...")

        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            try:
                response = requests.get(root_url, timeout=5)
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

    def post_deploy(self):
        logger.info(f"Devbox file browser: \nhttp://localhost:{self.local_port}/\n")


def create_devbox_app(local_port: Optional[int] = None, api_key: Optional[str] = None, params: Dict[str, str] = None) -> DevboxApp:
    """
    Create a DevboxApp for FileBrowser.

    Args:
        local_port: Local port for port forwarding
        api_key: API key for the application (not used for devbox)
        params: Dictionary of additional parameters to pass to the application.

    Returns:
        DevboxApp: The configured devbox app
    """
    # Use default values if None is provided
    if local_port is None:
        local_port = 8080
    
    # Define paths to scripts
    setup_script = str(Path(__file__).parent / "setup_devbox.sh")
    launch_script = str(Path(__file__).parent / "launch_devbox.sh")

    # Create the DevboxApp
    app = DevboxApp(
        name="devbox",
        setup_script=setup_script,
        launch_script=launch_script,
        program_file=None,  # devbox doesn't need a program file
        default_remote_port=8000,  # FileBrowser port in Docker
        default_local_port=local_port,
        log_command="sudo docker logs -f filebrowser",
        api_key=None,  # No API key needed for nginx
        test_request_function=None,  # No test request function as specified
        params=params or {}
    )

    return app

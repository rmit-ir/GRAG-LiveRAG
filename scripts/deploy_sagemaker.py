#!/usr/bin/env python
"""
Script to deploy Falcon3-10b model on SageMaker with automatic cleanup on exit.
"""
import sys
import time
import signal
import argparse
import atexit

from utils.logging_utils import get_logger
from services.llms.sagemaker_client import SageMakerClient

logger = get_logger("falcon-deploy")

# Global variable to track client for cleanup
sagemaker_client = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy Falcon3-10b on SageMaker")
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g6.12xlarge",
        help="SageMaker instance type for deployment",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="tiiuae/falcon-3-10b",
        help="Hugging Face model ID to deploy",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Custom endpoint name (default: auto-generated)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: from AWS config)",
    )
    return parser.parse_args()


def cleanup_resources():
    """Clean up all SageMaker resources to avoid costs."""
    global sagemaker_client
    
    if sagemaker_client:
        try:
            logger.info("Cleaning up SageMaker resources...")
            sagemaker_client.cleanup_resources()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up resources: %s", str(e))


def signal_handler(sig, frame):
    """Handle interruption signals by cleaning up resources before exit."""
    logger.info("Received signal %s, cleaning up resources...", sig)
    cleanup_resources()
    sys.exit(0)
def test_endpoint(client: SageMakerClient):
    """
    Test the deployed endpoint with a simple prompt.
    
    Args:
        client: SageMaker client with active deployment
    """
    prompt = "Hello, my name is"
    logger.info("Testing endpoint with prompt: %s", prompt)
    
    try:
        result, content = client.query(prompt, {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
        })
        logger.info("Response from model: %s", content)
        return result
    except Exception as e:
        logger.error("Error testing endpoint: %s", str(e))
        return None


def interactive_session(client: SageMakerClient):
    """
    Start an interactive session with the deployed model.
    
    Args:
        client: SageMaker client with active deployment
    """
    logger.info("Starting interactive session with endpoint: %s", client.endpoint_name)
    logger.info("Type 'exit' to quit and clean up resources")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ")
            if prompt.lower() == "exit":
                break
            
            parameters = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True,
            }
            
            start_time = time.time()
            _, content = client.query(prompt, parameters)
            elapsed_time = time.time() - start_time
            
            print("\nResponse:")
            print(content)
            print(f"\n[Generated in {elapsed_time:.2f} seconds]")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error("Error: %s", str(e))
def main():
    """Main function to deploy and interact with the model."""
    args = parse_args()
    
    # Register signal handlers for proper cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function to run on normal exit
    atexit.register(cleanup_resources)
    
    try:
        # Initialize SageMaker client
        global sagemaker_client
        sagemaker_client = SageMakerClient(
            model_id=args.model_id,
            instance_type=args.instance_type,
            region_name=args.region
        )
        
        # Deploy the model
        deployment = sagemaker_client.create_deployment(
            endpoint_name=args.endpoint_name
        )
        
        logger.info("Model deployed successfully to endpoint: %s", deployment["endpoint_name"])
        
        # Test the endpoint
        test_endpoint(sagemaker_client)
        
        # Start interactive session
        interactive_session(sagemaker_client)
        
    except Exception as e:
        logger.error("Error in deployment: %s", str(e))
    finally:
        # Ensure cleanup happens even if there's an unhandled exception
        cleanup_resources()


if __name__ == "__main__":
    main()

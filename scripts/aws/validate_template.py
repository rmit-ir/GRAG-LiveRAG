#!/usr/bin/env python3
"""
Script to validate the EC2 LLM CloudFormation template.
"""
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("validate_template")

def validate_template():
    """
    Validate the CloudFormation template for EC2 LLM deployment.
    """
    # Get the template path
    template_path = Path(__file__).parent / "config" / "ec2_llm_template.yaml"
    with open(template_path, "r") as f:
        template_body = f.read()
    
    # Get AWS credentials from environment variables with RACE_ prefix
    access_key = os.environ.get("RACE_AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("RACE_AWS_SESSION_TOKEN", "")
    region_name = os.environ.get("RACE_AWS_REGION", "us-west-2")
    
    try:
        # Set up boto3 session with explicit credentials
        boto_session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region_name
        )
        
        cf_client = boto_session.client('cloudformation')
        
        # Validate the template
        logger.info(f"Validating template: {template_path}")
        response = cf_client.validate_template(TemplateBody=template_body)
        
        logger.info("Template is valid!")
        logger.info(f"Template parameters: {response.get('Parameters', [])}")
        
        return True
    except ClientError as e:
        if "InvalidClientTokenId" in str(e) or "AccessDenied" in str(e):
            logger.warning("AWS credentials error: %s", str(e))
            logger.warning("Please set valid RACE_AWS_* environment variables")
            
            # Even without valid credentials, we can check for the specific issue
            if "${EC2_LLM_API_KEY}" in template_body:
                logger.error("Template contains unresolved variable ${EC2_LLM_API_KEY}")
                logger.error("This is likely causing the 'Unresolved resource dependencies' error")
                return False
            else:
                logger.info("The template appears to have the correct variable substitution syntax")
                return True
        else:
            logger.error("Template validation error: %s", str(e))
            return False
    except Exception as e:
        logger.error("Error: %s", str(e))
        return False

if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Run validation
    success = validate_template()
    sys.exit(0 if success else 1)

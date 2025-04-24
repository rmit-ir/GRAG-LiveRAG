"""
AWS utility functions for authentication and parameter retrieval.
"""
import boto3
import os
from utils.logging_utils import get_logger


class LiveRAGAWSUtils:
    """
    Utility class for AWS operations, including SSM parameter retrieval
    and session management.
    """
    log = get_logger("live_rag_aws_utils")

    def __init__(self):
        """
        Initialize the LiveRAGAWSUtils with AWS region configuration.
        """
        self.aws_region_name = os.environ.get('AWS_LIVE_RAG_REGION')
        if not self.aws_region_name:
            raise ValueError("AWS_LIVE_RAG_REGION environment variable is required")

    def get_session(self):
        """
        Get a boto3 session configured with AWS credentials.
        
        Returns:
            boto3.Session: Configured AWS session
        """
        # Get AWS credentials from environment variables
        access_key = os.environ.get('AWS_LIVE_RAG_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_LIVE_RAG_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            raise ValueError("AWS_LIVE_RAG_ACCESS_KEY_ID and AWS_LIVE_RAG_SECRET_ACCESS_KEY environment variables are required")
            
        return boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=self.aws_region_name
        )

    def get_ssm_value(self, key: str) -> str:
        """
        Get a cleartext value from AWS SSM.
        
        Args:
            key: The SSM parameter key
            
        Returns:
            The parameter value
        """
        session = self.get_session()
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key)["Parameter"]["Value"]

    def get_ssm_secret(self, key: str) -> str:
        """
        Get an encrypted value from AWS SSM.
        
        Args:
            key: The SSM parameter key
            
        Returns:
            The decrypted parameter value
        """
        session = self.get_session()
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]

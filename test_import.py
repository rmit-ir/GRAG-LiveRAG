"""
Test script to verify that imports work correctly.
"""
# First import the setup_path module to add src to the Python path
import setup_path

# Now we can import from services directly
from services.aws_utils import AWSUtils

print("Import successful!")
print(f"AWSUtils class: {AWSUtils}")

# Create an instance (won't actually run since it requires AWS_REGION env var)
try:
    aws = AWSUtils()
    print(f"AWS Region: {aws.aws_region_name}")
except ValueError as e:
    print(f"Expected error: {e}")

# Let's try another import
from services.pinecone_index import PineconeService
print(f"PineconeService class: {PineconeService}")

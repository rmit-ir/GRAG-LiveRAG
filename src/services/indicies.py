"""
Vector index service module for vector operations.

This module imports specialized services for AWS utilities, embedding utilities,
and specific vector database implementations (Pinecone and OpenSearch).
"""
from dotenv import load_dotenv

# Import specialized services
from services.aws_utils import AWSUtils
from services.embedding_utils import EmbeddingUtils
from services.pinecone_index import PineconeService
from services.opensearch_index import OpenSearchService

# Load environment variables from .env file
load_dotenv()

# Example usage
if __name__ == "__main__":
    # Initialize the services individually
    aws_utils = AWSUtils()
    embedding_utils = EmbeddingUtils("intfloat/e5-base-v2")
    pinecone_service = PineconeService(
        embedding_model_name="intfloat/e5-base-v2",
        index_name="fineweb10bt-512-0w-e5-base-v2",
        namespace="default"
    )
    opensearch_service = OpenSearchService(
        embedding_model_name="intfloat/e5-base-v2",
        index_name="fineweb10bt-512-0w-e5-base-v2"
    )
    
    # Example Pinecone query
    print("=== Pinecone Single Query Example ===")
    results = pinecone_service.query_pinecone("What is a second brain?")
    PineconeService.show_pinecone_results(results)
    
    # Example batch Pinecone query
    print("\n=== Pinecone Batch Query Example ===")
    batch_results = pinecone_service.batch_query_pinecone(
        ["What is a second brain?", "how does a brain work?", "Where is Paris?"],
        top_k=2
    )
    for results in batch_results:
        PineconeService.show_pinecone_results(results)
    
    # Example OpenSearch query
    print("\n=== OpenSearch Single Query Example ===")
    results = opensearch_service.query_opensearch("What is a second brain?")
    OpenSearchService.show_opensearch_results(results)
    
    # Example batch OpenSearch query
    print("\n=== OpenSearch Batch Query Example ===")
    batch_results = opensearch_service.batch_query_opensearch(
        ["What is a second brain?", "how does a brain work?", "Where is Paris?"],
        top_k=1
    )
    for results in batch_results['responses']:
        OpenSearchService.show_opensearch_results(results)

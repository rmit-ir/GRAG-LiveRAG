"""
Manages OpenSearch index connections and queries.

Provides functionality for querying OpenSearch vector databases,
handling authentication, and processing results.
"""
from typing import List, Dict, Any, Optional
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

# Import local utilities
from services.aws_utils import AWSUtils
from services.embedding_utils import EmbeddingUtils

# Load environment variables from .env file
load_dotenv()


class OpenSearchService:
    """
    Service for interacting with OpenSearch vector database.
    Provides query functionality for both single and batch queries.
    """

    def __init__(
        self,
        embedding_model_name: str = "intfloat/e5-base-v2",
        index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        region_name: Optional[str] = None
    ):
        """
        Initialize the OpenSearchService with configuration parameters.

        Args:
            embedding_model_name: Name of the embedding model to use
            index_name: Name of the OpenSearch index
            region_name: AWS region name (defaults to AWS_REGION environment variable)
        """
        self.aws_utils = AWSUtils(region_name)
        self.embedding_utils = EmbeddingUtils(embedding_model_name)
        self.index_name = index_name
        self._opensearch_client = None

    def get_opensearch_client(self):
        """Get the OpenSearch client."""
        if self._opensearch_client is None:
            # Create credentials manually
            session = self.aws_utils.get_session()
            credentials = session.get_credentials()
            auth = AWSV4SignerAuth(credentials, region=self.aws_utils.aws_region_name)
            host_name = self.aws_utils.get_ssm_value("/opensearch/endpoint")
            
            self._opensearch_client = OpenSearch(
                hosts=[{"host": host_name, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
            
        return self._opensearch_client

    def query_opensearch(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Query the OpenSearch index with a single query.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            OpenSearch query results
        """
        client = self.get_opensearch_client()
        results = client.search(
            index=self.index_name,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        
        return results

    def batch_query_opensearch(
        self,
        queries: List[str],
        top_k: int = 10,
        n_parallel: int = 10
    ) -> Dict[str, Any]:
        """
        Query the OpenSearch index with multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of top results to return for each query
            n_parallel: Number of parallel queries to run (used for configuration)
            
        Returns:
            OpenSearch multi-search query results
        """
        client = self.get_opensearch_client()
        request = []
        
        for query in queries:
            req_head = {"index": self.index_name}
            req_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "size": top_k,
            }
            request.extend([req_head, req_body])
            
        return client.msearch(body=request)

    @staticmethod
    def show_opensearch_results(results: Dict[str, Any]) -> None:
        """
        Display OpenSearch query results in a readable format.
        
        Args:
            results: OpenSearch query results
        """
        if "hits" in results:
            # Single query results
            for match in results["hits"]["hits"]:
                print("chunk:", match["_id"], "score:", match["_score"])
                print(match["_source"]["text"])
                print()
        elif "responses" in results:
            # Batch query results
            for response in results["responses"]:
                for match in response["hits"]["hits"]:
                    print("chunk:", match["_id"], "score:", match["_score"])
                    print(match["_source"]["text"])
                    print()


# Example usage
if __name__ == "__main__":
    # Initialize the service
    service = OpenSearchService()
    
    # Example OpenSearch query
    print("=== OpenSearch Single Query Example ===")
    results = service.query_opensearch("What is a second brain?")
    service.show_opensearch_results(results)
    
    # Example batch OpenSearch query
    print("\n=== OpenSearch Batch Query Example ===")
    batch_results = service.batch_query_opensearch(
        ["What is a second brain?", "how does a brain work?", "Where is Paris?"],
        top_k=1
    )
    service.show_opensearch_results(batch_results)

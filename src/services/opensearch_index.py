"""
Manages OpenSearch index connections and queries.

Provides functionality for querying OpenSearch vector databases,
handling authentication, and processing results.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

# Import local utilities
from utils.logging_utils import get_logger
from services.live_rag_aws_utils import LiveRAGAWSUtils
from services.live_rag_metadata import LiveRAGMetadata

# Dataclasses for structured OpenSearch results


@dataclass
class OpenSearchHit:
    """Represents a single hit from an OpenSearch query."""
    index: str
    id: str
    score: float
    source: LiveRAGMetadata

    def score_percentage(self) -> float:
        """Return the score as a percentage (normalized for display)."""
        # OpenSearch scores can vary widely, this is a simple normalization
        # that may need adjustment based on your specific use case
        return round(min(self.score * 5, 100), 2)


@dataclass
class OpenSearchShardInfo:
    """Information about shards in an OpenSearch query."""
    total: int
    successful: int
    skipped: int
    failed: int


@dataclass
class OpenSearchTotalHits:
    """Information about total hits in an OpenSearch query."""
    value: int
    relation: str


@dataclass
class OpenSearchResult:
    """
    Structured representation of an OpenSearch query result.

    Provides convenient access to hits and metadata.
    """
    took: int
    timed_out: bool
    shards: OpenSearchShardInfo
    hits: List[OpenSearchHit] = field(default_factory=list)
    total_hits: Optional[OpenSearchTotalHits] = None
    max_score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchResult":
        """
        Create an OpenSearchResult instance from a raw OpenSearch response dictionary.

        Args:
            data: Raw OpenSearch response dictionary

        Returns:
            Structured OpenSearchResult object
        """
        # Extract shard information
        shards = OpenSearchShardInfo(
            total=data.get("_shards", {}).get("total", 0),
            successful=data.get("_shards", {}).get("successful", 0),
            skipped=data.get("_shards", {}).get("skipped", 0),
            failed=data.get("_shards", {}).get("failed", 0)
        )

        # Extract hits information
        hits_data = data.get("hits", {})
        total_hits = None
        if "total" in hits_data:
            if isinstance(hits_data["total"], dict):
                total_hits = OpenSearchTotalHits(
                    value=hits_data["total"].get("value", 0),
                    relation=hits_data["total"].get("relation", "eq")
                )
            else:
                # Handle case where total is just a number
                total_hits = OpenSearchTotalHits(
                    value=hits_data["total"],
                    relation="eq"
                )

        # Extract individual hits
        hits = []
        for hit in hits_data.get("hits", []):
            hits.append(OpenSearchHit(
                index=hit.get("_index", ""),
                id=hit.get("_id", ""),
                score=hit.get("_score", 0.0),
                source=LiveRAGMetadata.from_dict(hit.get("_source", {}))
            ))

        return cls(
            took=data.get("took", 0),
            timed_out=data.get("timed_out", False),
            shards=shards,
            hits=hits,
            total_hits=total_hits,
            max_score=hits_data.get("max_score")
        )


# Load environment variables from .env file
load_dotenv()


class OpenSearchService:
    """
    Service for interacting with OpenSearch vector database.
    Provides query functionality for both single and batch queries.
    """

    # Class logger
    log = get_logger("OpenSearchService")

    def __init__(
        self,
        index_name: str = "fineweb10bt-512-0w-e5-base-v2"
    ):
        """
        Initialize the OpenSearchService with configuration parameters.

        Args:
            index_name: Name of the OpenSearch index
        """
        self.live_rag_aws_utils = LiveRAGAWSUtils()
        self.index_name = index_name
        self._opensearch_client = None
        self.log.info("OpenSearchService initialized",
                      index_name=index_name)

    def get_opensearch_client(self):
        """Get the OpenSearch client."""
        if self._opensearch_client is None:
            self.log.debug("Connecting to OpenSearch",
                           index_name=self.index_name)
            # Create credentials manually
            session = self.live_rag_aws_utils.get_session()
            credentials = session.get_credentials()
            auth = AWSV4SignerAuth(
                credentials, region=self.live_rag_aws_utils.aws_region_name)
            host_name = self.live_rag_aws_utils.get_ssm_value("/opensearch/endpoint")

            self._opensearch_client = OpenSearch(
                hosts=[{"host": host_name, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )
            self.log.debug("Connected to OpenSearch",
                           index_name=self.index_name)

        return self._opensearch_client

    def query_opensearch(self, query: str, top_k: int = 10) -> OpenSearchResult:
        """
        Query the OpenSearch index with a single query.

        Args:
            query: The query text
            top_k: Number of top results to return

        Returns:
            Structured OpenSearchResult object
        """
        self.log.debug("Querying OpenSearch", query=query, top_k=top_k)

        client = self.get_opensearch_client()
        results = client.search(
            index=self.index_name,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )

        match_count = len(results.get("hits", {}).get("hits", []))
        self.log.debug("Query completed",
                       matches_found=match_count, results=results)

        return OpenSearchResult.from_dict(results)

    def batch_query_opensearch(
        self,
        queries: List[str],
        top_k: int = 10,
        n_parallel: int = 10
    ) -> List[OpenSearchResult]:
        """
        Query the OpenSearch index with multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of top results to return for each query
            n_parallel: Number of parallel queries to run (used for configuration)

        Returns:
            List of structured OpenSearchResult objects
        """
        self.log.debug("Batch querying OpenSearch",
                       query_count=len(queries),
                       top_k=top_k,
                       n_parallel=n_parallel)

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

        results = client.msearch(body=request)
        self.log.debug("Batch query completed",
                       result_count=len(results.get("responses", [])),
                       results=results)

        # Process the responses into structured objects
        structured_results = []
        if "responses" in results:
            for response in results["responses"]:
                structured_results.append(OpenSearchResult.from_dict(response))

        return structured_results

    def show_opensearch_results(self, result: OpenSearchResult,
                                max_text_length: Optional[int] = 100,
                                show_metadata_fields: Optional[List[str]] = None) -> None:
        """
        Helper method to display a single OpenSearch result.

        Args:
            hit: The OpenSearchHit to display
            index: The result number to display
            max_text_length: Optional maximum length for text display
            show_metadata_fields: Optional list of metadata fields to display
        """
        hits = result.hits
        if not hits:
            self.log.info(f"No matches found.")
            return

        self.log.info(f"Show {len(hits)} results")
        for i, hit in enumerate(hits, 1):
            # Format the score as percentage
            score_percent = hit.score_percentage()

            # Create a header with result number, ID and score
            print(
                f"🔍 RESULT #{i} | ID: {hit.id} | RELEVANCE: {score_percent}%")

            # Handle text content with optional truncation
            if hit.text:
                text = hit.text
                if max_text_length and len(text) > max_text_length:
                    display_text = text[:max_text_length] + "..."
                else:
                    display_text = text

                print("\nCONTENT:")
                print(f"{display_text}")
            else:
                print("\nCONTENT: <No text available>")

            # Show additional metadata fields if requested
            if show_metadata_fields:
                print("\nADDITIONAL METADATA:")
                for field in show_metadata_fields:
                    if field != "text" and hasattr(hit.source, field):
                        print(f"  {field}: {getattr(hit.source, field)}")

            # Add separator between results
            print("-" * 80)


# Example usage
if __name__ == "__main__":
    # Initialize the service
    service = OpenSearchService()

    # Example 1: Basic usage with default parameters
    print("=== Example 1: Basic OpenSearch Query Results ===")
    results = service.query_opensearch("What is a second brain?", top_k=3)
    service.show_opensearch_results(results)

    # Example 2: With text truncation
    print("\n=== Example 2: Truncated Text (100 characters) ===")
    results = service.query_opensearch("What is a second brain?", top_k=2)
    service.show_opensearch_results(results, max_text_length=100)

    # Example 3: Display additional metadata fields if they exist
    print("\n=== Example 3: Showing Additional Metadata Fields ===")
    results = service.query_opensearch("What is a second brain?", top_k=2)
    service.show_opensearch_results(
        results, max_text_length=100, show_metadata_fields=["source", "url", "title"])

    # Example 4: Batch query with multiple prompts
    print("\n=== Example 4: Batch Query Example ===")
    batch_results = service.batch_query_opensearch(
        ["How does a brain work?", "Where is Paris?",
            "What is artificial intelligence?"],
        top_k=1
    )
    for batch in batch_results:
        service.show_opensearch_results(batch, max_text_length=150)

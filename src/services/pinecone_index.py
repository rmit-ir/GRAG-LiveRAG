"""
Manages Pinecone index connections and queries.

Provides functionality for querying Pinecone vector databases,
handling authentication, and processing results.
"""
from utils.logging_utils import get_logger
from services.embedding_utils import EmbeddingUtils
from services.live_rag_aws_utils import LiveRAGAWSUtils
from services.live_rag_metadata import LiveRAGMetadata, live_rag_metadata_from_dict
from typing import List, Dict, Any, Optional, NamedTuple
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
from pinecone import Pinecone


class PineconeMatch(NamedTuple):
    """Represents a single match from a Pinecone query."""
    id: str
    score: float
    metadata: LiveRAGMetadata
    values: List[float] = []


class PineconeUsage(NamedTuple):
    """Represents usage information from a Pinecone query."""
    read_units: int


class PineconeResult(NamedTuple):
    """
    Structured representation of a Pinecone query result.

    Provides convenient access to matches and metadata.
    """
    matches: List[PineconeMatch] = []
    namespace: str = "default"
    usage: Optional[PineconeUsage] = None


def pinecone_result_from_dict(data: Dict[str, Any]) -> PineconeResult:
    """
    Create a PineconeResult instance from a raw Pinecone response dictionary.

    Args:
        data: Raw Pinecone response dictionary

    Returns:
        Structured PineconeResult object
    """
    matches = [
        PineconeMatch(
            id=match["id"],
            score=match["score"],
                metadata=live_rag_metadata_from_dict(match.get("metadata", {})),
            values=match.get("values", [])
        )
        for match in data.get("matches", [])
    ]

    usage = None
    if "usage" in data:
        usage = PineconeUsage(
            read_units=data["usage"].get("read_units", 0))

    return PineconeResult(
        matches=matches,
        namespace=data.get("namespace", "default"),
        usage=usage
    )


# Load environment variables from .env file
load_dotenv()


class PineconeService:
    """
    Service for interacting with Pinecone vector database.
    Provides query functionality for both single and batch queries.
    """

    # Class logger
    log = get_logger("PineconeService")

    def __init__(
        self,
        embedding_model_name: str = "intfloat/e5-base-v2",
        index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        namespace: str = "default"
    ):
        """
        Initialize the PineconeService with configuration parameters.

        Args:
            embedding_model_name: Name of the embedding model to use
            index_name: Name of the Pinecone index
            namespace: Namespace in the Pinecone index
        """
        self.live_rag_aws_utils = LiveRAGAWSUtils()
        self.embedding_utils = EmbeddingUtils(embedding_model_name)
        self.index_name = index_name
        self.namespace = namespace
        self._pinecone_index = None
        self.log.info("PineconeService initialized",
                      embedding_model=embedding_model_name,
                      index_name=index_name)

    def get_pinecone_index(self):
        """Get the Pinecone index client."""
        if self._pinecone_index is None:
            self.log.debug("Connecting to Pinecone index",
                           index_name=self.index_name)
            pc = Pinecone(api_key=self.live_rag_aws_utils.get_ssm_secret(
                "/pinecone/ro_token"))
            self._pinecone_index = pc.Index(name=self.index_name)
            self.log.debug("Connected to Pinecone index",
                           index_name=self.index_name)
        return self._pinecone_index

    def query_pinecone(
        self,
        query: str,
        top_k: int = 10,
        namespace: Optional[str] = None
    ) -> PineconeResult:
        """
        Query the Pinecone index with a single query.

        Args:
            query: The query text
            top_k: Number of top results to return
            namespace: Pinecone namespace to query (defaults to self.namespace)

        Returns:
            Structured PineconeResult object
        """
        if namespace is None:
            namespace = self.namespace

        self.log.debug("Querying Pinecone", query=query, top_k=top_k)

        index = self.get_pinecone_index()
        results = index.query(
            vector=self.embedding_utils.embed_query(query),
            top_k=top_k,
            include_values=False,
            namespace=namespace,
            include_metadata=True
        )

        match_count = len(results.get("matches", []))
        self.log.debug("Query completed",
                       matches_found=match_count, results=results)

        return pinecone_result_from_dict(results)

    def batch_query_pinecone(
        self,
        queries: List[str],
        top_k: int = 10,
        namespace: Optional[str] = None,
        n_parallel: int = 10
    ) -> List[PineconeResult]:
        """
        Query the Pinecone index with multiple queries in parallel.

        Args:
            queries: List of query texts
            top_k: Number of top results to return for each query
            namespace: Pinecone namespace to query (defaults to self.namespace)
            n_parallel: Number of parallel queries to run

        Returns:
            List of structured PineconeResult objects
        """
        if namespace is None:
            namespace = self.namespace

        self.log.debug("Batch querying Pinecone",
                       query_count=len(queries),
                       top_k=top_k,
                       n_parallel=n_parallel)

        index = self.get_pinecone_index()
        embeds = self.embedding_utils.batch_embed_queries(queries)

        pool = ThreadPool(n_parallel)
        results = pool.map(
            lambda x: index.query(
                vector=x,
                top_k=top_k,
                include_values=False,
                namespace=namespace,
                include_metadata=True
            ),
            embeds
        )

        self.log.debug("Batch query completed",
                       result_count=len(results), results=results)
        return [pinecone_result_from_dict(result) for result in results]

    def show_pinecone_results(self, results: PineconeResult, max_text_length: Optional[int] = 100,
                              show_metadata_fields: Optional[List[str]] = None) -> None:
        """
        Display Pinecone query results in a readable format with improved formatting.

        Args:
            results: Structured PineconeResult object
            max_text_length: Optional maximum length for text display (None for full text)
            show_metadata_fields: Optional list of metadata fields to display besides 'text'
        """
        if not results.matches:
            self.log.info("No matches found in Pinecone results")
            return

        match_count = len(results.matches)
        self.log.info(f"Show {match_count} results")
        print("-" * 80)

        for i, match in enumerate(results.matches, 1):
            # Format the score as percentage
            score_percent = round(match.score * 100, 2)
            # Create a header with result number, ID and score
            print(
                f"🔍 RESULT #{i} | ID: {match.id} | RELEVANCE: {score_percent}%")

            # Handle text content with optional truncation
            if match.text:
                text = match.text
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
                    if field != "text" and hasattr(match.metadata, field):
                        print(f"  {field}: {getattr(match.metadata, field)}")

            # Add separator between results
            print("-" * 80)


# Example usage
if __name__ == "__main__":
    # Initialize the service
    service = PineconeService()

    # Example 1: Basic usage with default parameters
    print("=== Example 1: Basic Pinecone Query Results ===")
    results = service.query_pinecone("What is a second brain?", top_k=3)
    service.show_pinecone_results(results)

    # Example 2: With text truncation
    print("\n=== Example 2: Truncated Text (100 characters) ===")
    results = service.query_pinecone("What is a second brain?", top_k=2)
    service.show_pinecone_results(results, max_text_length=100)

    # Example 3: Display additional metadata fields if they exist
    print("\n=== Example 3: Showing Additional Metadata Fields ===")
    results = service.query_pinecone("What is a second brain?", top_k=2)
    service.show_pinecone_results(results, show_metadata_fields=[
                                  "source", "url", "title"])

    # Example 4: Batch query with multiple prompts
    print("\n=== Example 4: Batch Query Example ===")
    batch_results = service.batch_query_pinecone(
        ["How does a brain work?", "Where is Paris?"],
        top_k=1
    )
    for i, results in enumerate(batch_results):
        print(f"\nBatch Result #{i+1}:")
        service.show_pinecone_results(results, max_text_length=150)

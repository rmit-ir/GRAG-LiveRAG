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
            failed=data.get("_shards", {}).get("failed", 0),
        )

        # Extract hits information
        hits_data = data.get("hits", {})
        total_hits = None
        if "total" in hits_data:
            if isinstance(hits_data["total"], dict):
                total_hits = OpenSearchTotalHits(
                    value=hits_data["total"].get("value", 0),
                    relation=hits_data["total"].get("relation", "eq"),
                )
            else:
                # Handle case where total is just a number
                total_hits = OpenSearchTotalHits(
                    value=hits_data["total"], relation="eq"
                )

        # Extract individual hits
        hits = []
        for hit in hits_data.get("hits", []):
            hits.append(
                OpenSearchHit(
                    index=hit.get("_index", ""),
                    id=hit.get("_id", ""),
                    score=hit.get("_score", 0.0),
                    source=LiveRAGMetadata.from_dict(hit.get("_source", {})),
                )
            )

        return cls(
            took=data.get("took", 0),
            timed_out=data.get("timed_out", False),
            shards=shards,
            hits=hits,
            total_hits=total_hits,
            max_score=hits_data.get("max_score"),
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

    def __init__(self, index_name: str = "fineweb10bt-512-0w-e5-base-v2"):
        """
        Initialize the OpenSearchService with configuration parameters.

        Args:
            index_name: Name of the OpenSearch index
        """
        self.live_rag_aws_utils = LiveRAGAWSUtils()
        self.index_name = index_name
        self._opensearch_client = None
        self.log.info("OpenSearchService initialized", index_name=index_name)

    def get_opensearch_client(self):
        """Get the OpenSearch client."""
        if self._opensearch_client is None:
            self.log.debug("Connecting to OpenSearch",
                           index_name=self.index_name)
            # Create credentials manually
            session = self.live_rag_aws_utils.get_session()
            credentials = session.get_credentials()
            auth = AWSV4SignerAuth(
                credentials, region=self.live_rag_aws_utils.aws_region_name
            )
            host_name = self.live_rag_aws_utils.get_ssm_value(
                "/opensearch/endpoint")

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
            body={"query": {"match": {"text": query}}, "size": top_k},
        )

        match_count = len(results.get("hits", {}).get("hits", []))
        self.log.debug("Query completed",
                       matches_found=match_count, results=results)

        return OpenSearchResult.from_dict(results)

    def batch_query_opensearch(
        self, queries: List[str], top_k: int = 10, n_parallel: int = 10
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
        self.log.debug(
            "Batch querying OpenSearch",
            query_count=len(queries),
            top_k=top_k,
            n_parallel=n_parallel,
        )

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
        self.log.debug(
            "Batch query completed",
            result_count=len(results.get("responses", [])),
            results=results,
        )

        # Process the responses into structured objects
        structured_results = []
        if "responses" in results:
            for response in results["responses"]:
                structured_results.append(OpenSearchResult.from_dict(response))

        return structured_results

    def get_docs(self, doc_ids: List[str], size_per_doc: int = 20) -> OpenSearchResult:
        """
        Retrieve all chunks belonging to multiple documents by their doc_ids in a single request.

        This method queries OpenSearch for chunks that match any of the provided document IDs,
        allowing efficient retrieval of multiple documents and their chunks in one request.

        Args:
            doc_ids: List of document IDs to search for
            size_per_doc: Maximum number of chunks to return per document (default: 100)

        Returns:
            Structured OpenSearchResult object containing all chunks for the requested documents
        """
        if not doc_ids:
            self.log.warning("No document IDs provided")
            return OpenSearchResult(took=0, timed_out=False,
                                    shards=OpenSearchShardInfo(0, 0, 0, 0))

        self.log.debug("Retrieving chunks for multiple documents",
                       doc_count=len(doc_ids),
                       doc_ids=doc_ids)

        client = self.get_opensearch_client()

        _doc_id_set = set()
        doc_id_list = []
        for doc_id in doc_ids:
            if doc_id.startswith("doc-"):
                # Transform doc-<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>::chunk-1
                # to <urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>
                doc_id = doc_id.split("::")[0][4:]
            if doc_id not in _doc_id_set:
                _doc_id_set.add(doc_id)
                doc_id_list.append(doc_id)

        # Calculate total size based on number of documents and size per document
        total_size = len(_doc_id_set) * size_per_doc

        # Use terms query to match multiple document IDs
        results = client.search(
            index=self.index_name,
            body={
                "query": {
                    "terms": {
                        "doc_id": doc_id_list
                    }
                },
                "size": total_size
            }
        )

        # Store combined hits in a dictionary keyed by doc_id
        doc_hits_dict: Dict[str, OpenSearchHit] = {}

        # Process all hits from OpenSearch
        for hit in results.get("hits", {}).get("hits", []):
            # Extract the document ID from the hit
            source = hit.get("_source", {})
            doc_id = source.get("doc_id", "")

            if doc_id not in doc_hits_dict:
                # Create a new hit with default metadata values that make sense for combined chunks
                source_data = hit.get("_source", {})
                doc_hits_dict[doc_id] = OpenSearchHit(
                    index=hit.get("_index", ""),
                    id=doc_id,
                    score=hit.get("_score", 0.0),
                    source=LiveRAGMetadata(
                        chunk_order=0.0,  # Combined chunk, so default to 0.0
                        doc_id=doc_id,
                        is_first_chunk=True,
                        is_last_chunk=True,
                        text=source_data.get("text", ""),
                        total_doc_chunks=1.0
                    )
                )
            else:
                # Append the chunk's text to the existing document's text
                existing_hit = doc_hits_dict[doc_id]
                chunk_text = source.get('text', None)
                if chunk_text:
                    existing_hit.source.text += f"\n\n{chunk_text.strip()}"

        # Create the final list of hits in the same order as the input doc_ids
        doc_hits_merged: List[OpenSearchHit] = []
        for doc_id in doc_id_list:
            if doc_id in doc_hits_dict:
                doc_hits_merged.append(doc_hits_dict[doc_id])

        return OpenSearchResult(
            took=results.get("took", 0),
            timed_out=results.get("timed_out", False),
            shards=OpenSearchShardInfo(
                total=results.get("_shards", {}).get("total", 0),
                successful=results.get("_shards", {}).get("successful", 0),
                skipped=results.get("_shards", {}).get("skipped", 0),
                failed=results.get("_shards", {}).get("failed", 0)
            ),
            hits=doc_hits_merged,
            total_hits=OpenSearchTotalHits(
                value=results.get("hits", {}).get(
                    "total", {}).get("value", 0),
                relation=results.get("hits", {}).get(
                    "total", {}).get("relation", "eq")
            ),
            # doesn't make sense, we are getting docs by id
            max_score=0.0
        )

    def show_opensearch_results(
        self,
        result: OpenSearchResult,
        max_text_length: Optional[int] = 100,
        show_metadata_fields: Optional[List[str]] = None,
    ) -> None:
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
            if hasattr(hit.source, "text"):
                text = hit.source.text
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

    # Example: Get chunks for multiple documents
    print("\n=== Getting chunks by multiple doc IDs ===")
    # id got from https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer
    doc_ids = ["doc-<urn:uuid:3ac370c9-8a66-43d1-beae-0f7d3c23f78d>::chunk-0", "doc-<urn:uuid:58254756-1664-4f28-82cc-456768d65852>::chunk-3"]
    result = service.get_docs(doc_ids)
    for hit in result.hits:
        print(f"Document ID: {hit.id}")
        print("-" * 80)
    # we can't find it in Opensearch index

    # Example: Get chunks for multiple documents
    print("\n=== Getting chunks by multiple doc IDs ===")
    doc_ids = ["<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>", "<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>", "<urn:uuid:32d4d757-52c6-4a26-a038-9eb45f29389a>", "<urn:uuid:4cb3fec5-366c-4f66-a78e-3390c4b8fcc2>", "<urn:uuid:56a0db0c-f984-4f4b-bad4-36e317725375>", "<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>",
               "<urn:uuid:75931ce4-3825-4ef9-9673-b07dfc319d66>", "<urn:uuid:fc1fd791-7a5c-4d05-a1be-0e1bc92fd342>", "<urn:uuid:75931ce4-3825-4ef9-9673-b07dfc319d66>", "<urn:uuid:164b81ba-7a4b-449c-8232-41c809ef65db>", "<urn:uuid:57031123-cef2-4f51-adbe-4b6070379a8d>", "<urn:uuid:54236e11-cfe0-4870-8679-dcca522c66bc>", "<urn:uuid:380337c6-3957-4c63-9583-ba486d4994c1>"]
    result = service.get_docs(doc_ids)
    for hit in result.hits:
        print(f"Document ID: {hit.id}")
        print("-" * 80)

"""
Vector index service module for vector operations.

This module imports specialized services for AWS utilities, embedding utilities,
and specific vector database implementations (Pinecone and OpenSearch).
"""
from typing import Optional, List, Dict, NamedTuple
from dotenv import load_dotenv

# Import utilities
from utils.logging_utils import get_logger

# Import specialized services
import pandas as pd
from trectools import TrecRun, fusion
from services.live_rag_metadata import LiveRAGMetadata
from services.pinecone_index import PineconeMatch, PineconeService
from services.opensearch_index import OpenSearchHit, OpenSearchService

# Load environment variables from .env file
load_dotenv()


class SearchHit(NamedTuple):
    """Represents a search hit from any search backend."""
    id: str
    score: float
    metadata: LiveRAGMetadata
    retrieval_model: str


def search_hit_from_pinecone(record: PineconeMatch) -> SearchHit:
    """
    Create a SearchHit from a PineconeMatch.

    Args:
        record: PineconeMatch object

    Returns:
        SearchHit object
    """
    return SearchHit(
        id=record.id,
        score=record.score,
        metadata=record.metadata,
        retrieval_model="embedding"
    )


def search_hit_from_opensearch(record: OpenSearchHit) -> SearchHit:
    """
    Create a SearchHit from an OpenSearchHit.

    Args:
        record: OpenSearchHit object

    Returns:
        SearchHit object
    """
    return SearchHit(
        id=record.id,
        score=record.score,
        metadata=record.source,
        retrieval_model="BM25"
    )


def search_hit_to_dict(hit: SearchHit) -> Dict:
    """
    Convert a SearchHit to a dictionary.

    Args:
        hit: SearchHit object

    Returns:
        Dictionary representation of the SearchHit
    """
    return {
        "id": hit.id,
        "score": hit.score,
        "metadata": hit.metadata.to_dict(),
        "retrieval_model": hit.retrieval_model
    }


class QueryService:
    """
    Service for querying both embedding-based and keyword-based search systems.
    Encapsulates both PineconeService and OpenSearchService for unified access.
    """

    # Class logger
    log = get_logger("QueryService")

    def __init__(
        self,
        pinecone_embedding_model_name: str = "intfloat/e5-base-v2",
        pinecone_index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        pinecone_namespace: str = "default",
        opensearch_index_name: str = "fineweb10bt-512-0w-e5-base-v2"
    ):
        self.pinecone_service = PineconeService(
            embedding_model_name=pinecone_embedding_model_name,
            index_name=pinecone_index_name,
            namespace=pinecone_namespace
        )
        self.opensearch_service = OpenSearchService(
            index_name=opensearch_index_name
        )

    def query_embedding(
        self,
        query: str,
        k: int = 10,
        namespace: Optional[str] = None,
        **kwargs
    ) -> List[SearchHit]:
        """
        Retrieves relevant documents using embedding-based search (vector retrieval).

        Args:
            query (str): The query to run.
            k (int): Maximum number of documents to return.
            namespace (str, optional): The namespace in Pinecone to query.

        Returns:
            List[SearchHit]: A list of SearchHit objects containing retrieved documents.
        """
        self.log.debug("Starting embedding query",
                       query=query, k=k, namespace=namespace)
        results = self.pinecone_service.query_pinecone(
            query, top_k=k, namespace=namespace, **kwargs)
        hits = [search_hit_from_pinecone(match) for match in results.matches]
        self.log.debug("Embedding query completed", hits_count=len(hits))
        return hits

    def query_keywords(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[SearchHit]:
        """
        Retrieves relevant documents from OpenSearch using keyword-based search.

        Args:
            query (str): The user query.
            k (int): Maximum number of documents to return.

        Returns:
            List[SearchHit]: A list of SearchHit objects containing retrieved documents.
        """
        self.log.debug("Starting keyword query", query=query, k=k)
        results = self.opensearch_service.query_opensearch(
            query, top_k=k, **kwargs)
        hits = [search_hit_from_opensearch(hit) for hit in results.hits]
        self.log.debug("Keyword query completed", hits_count=len(hits))
        return hits

    def query_fusion(
        self,
        query: str,
        k: int = 10,
        per_source_k: int = 10,
        return_raw_df: bool = False,
        pk_namespace: Optional[str] = None,
        rrf_k: int = 60,
        rrf_max_docs: int = 1000,
    ) -> List[SearchHit] | pd.DataFrame:
        """
        Retrieves relevant documents using both embedding-based and keyword-based search,
        then fuses the results using reciprocal rank fusion.

        Args:
            query (str): The query to run.
            k (int): Maximum number of documents to return in the final result.
            namespace (Optional[str]): The namespace in Pinecone to query.
            return_raw_df (bool): Whether to return the raw TrecTools DataFrame instead of SearchHit objects.
            rrf_k (int): Higher values (e.g. 60) give more weight to lower ranks, while lower values (e.g. 1) heavily favor top results.
            rrf_max_docs (int): Maximum number of documents from each ranking to include in the fusion.
            **kwargs: Additional arguments to pass to the query methods.

        Returns:
            List[SearchHit] | pd.DataFrame: A list of SearchHit objects or a DataFrame containing retrieved documents.
        """
        self.log.debug("Starting fusion query",
                       query=query,
                       k=k,
                       rrf_k=rrf_k,
                       rrf_max_docs=rrf_max_docs)

        # Query both embedding and keywords with the same k
        embedding_hits = self.query_embedding(
            query, k=per_source_k, namespace=pk_namespace)
        keyword_hits = self.query_keywords(query, k=per_source_k)

        # Convert hits to TrecRun objects
        embedding_run = self._hits_to_trecrun(
            query, embedding_hits, "embedding")
        keyword_run = self._hits_to_trecrun(query, keyword_hits, "keywords")

        self.log.debug("Performing reciprocal rank fusion",
                       embedding_hits=len(embedding_hits),
                       keyword_hits=len(keyword_hits))

        # Fuse the results using reciprocal rank fusion with the specified parameters
        fused_run = fusion.reciprocal_rank_fusion(
            [embedding_run, keyword_run], k=rrf_k, max_docs=rrf_max_docs)

        # Get the top k results from the fused run
        fused_df = fused_run.run_data.head(k)

        if return_raw_df:
            self.log.debug("Returning raw DataFrame", rows=len(fused_df))
            return fused_df

        # Create a dictionary to map document IDs to their original SearchHit objects
        hit_map = {hit.id: hit for hit in embedding_hits + keyword_hits}

        # Create a list of SearchHit objects from the fused results
        fused_hits = []
        for _, row in fused_df.iterrows():
            doc_id = row['docid']
            if doc_id in hit_map:
                # Use the original SearchHit object but update the score
                hit = hit_map[doc_id]
                # Create a new SearchHit with the fused score
                fused_hits.append(SearchHit(
                    id=hit.id,
                    score=row['score'],
                    metadata=hit.metadata,
                    retrieval_model=hit.retrieval_model
                ))

        self.log.debug("Fusion query completed",
                       fused_hits_count=len(fused_hits),
                       original_hits_count=len(embedding_hits) + len(keyword_hits))
        return fused_hits

    def get_docs(self, doc_ids: List[str], size_per_doc: int = 20, combined=True) -> List[SearchHit]:
        result = self.opensearch_service.get_docs(
            doc_ids, size_per_doc, combined)
        return [search_hit_from_opensearch(hit) for hit in result.hits]

    def _hits_to_trecrun(self, query_id: str, hits: List[SearchHit], tag: str) -> TrecRun:
        """
        Helper function to convert SearchHit objects to trectools TrecRun object.
        """
        # Create rows for the TrecRun dataframe
        rows = [{
            'query': query_id,
            'q0': 'q0',
            'docid': h.id,
            'rank': i+1,
            'score': h.score,
            'system': tag
        } for i, h in enumerate(hits)]

        # Create a TrecRun object
        query_run = TrecRun(None)
        df = pd.DataFrame(rows)
        df.columns = ["query", "q0", "docid", "rank", "score", "system"]
        query_run.load_run_from_dataframe(df)
        return query_run


# Example usage
if __name__ == "__main__":
    # Initialize the service
    service = QueryService()
    log = get_logger("indicies:main")

    # Example 1: Query using embedding-based search (Pinecone)
    log.info("=== Example 1: Embedding-Based Search (Pinecone) ===")
    embedding_results = service.query_embedding("What is a second brain?", k=3)
    log.info(
        f"Found {len(embedding_results)} results using embedding-based search")

    # Example 2: Query using keyword-based search (OpenSearch)
    log.info("\n=== Example 2: Keyword-Based Search (OpenSearch) ===")
    keyword_results = service.query_keywords("What is a second brain?", k=3)
    log.info(
        f"Found {len(keyword_results)} results using keyword-based search")

    # Example 3: Query using fusion (combining Pinecone and OpenSearch)
    log.info("\n=== Example 3: Fusion Search (Pinecone + OpenSearch) ===")
    fusion_results = service.query_fusion("What is a second brain?", k=5)
    log.info(f"Found {len(fusion_results)} results using fusion search")

    # Example 4: Return raw DataFrame from fusion
    log.info("\n=== Example 4: Fusion Search with Raw DataFrame ===")
    fusion_df = service.query_fusion(
        "What is artificial intelligence?", k=3, return_raw_df=True)
    fusion_df.info()
    log.info("Raw DataFrame from fusion search:")
    log.info(f"{fusion_df}")
    # Print first row as a dictionary
    log.info("First row as dict:")
    log.info(fusion_df.iloc[0].to_dict())

    # Example 5: Get documents by IDs
    log.info("\n=== Example 5: Get Documents by IDs ===")
    doc_ids = ["<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>", "<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>", "<urn:uuid:32d4d757-52c6-4a26-a038-9eb45f29389a>", "<urn:uuid:4cb3fec5-366c-4f66-a78e-3390c4b8fcc2>", "<urn:uuid:56a0db0c-f984-4f4b-bad4-36e317725375>", "<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>",
               "<urn:uuid:75931ce4-3825-4ef9-9673-b07dfc319d66>", "<urn:uuid:fc1fd791-7a5c-4d05-a1be-0e1bc92fd342>", "<urn:uuid:75931ce4-3825-4ef9-9673-b07dfc319d66>", "<urn:uuid:164b81ba-7a4b-449c-8232-41c809ef65db>", "<urn:uuid:57031123-cef2-4f51-adbe-4b6070379a8d>", "<urn:uuid:54236e11-cfe0-4870-8679-dcca522c66bc>", "<urn:uuid:380337c6-3957-4c63-9583-ba486d4994c1>"]
    log.info(f"Getting documents by IDs: {doc_ids}")
    docs = service.get_docs(doc_ids, size_per_doc=5)
    log.info(f"Found {len(docs)} documents by IDs")
    for doc in docs:
        log.info(f"Document ID: {doc.id}, Score: {doc.score}")
        log.info(f"Document Full Text: \n{'-'*10}\n{doc.metadata.text}")

    # combined false
    docs = service.get_docs(doc_ids, size_per_doc=5, combined=False)
    log.info(f"Found {len(docs)} documents by IDs")
    for doc in docs:
        log.info(f"Document ID: {doc.id}, Score: {doc.score}")
        log.info(f"Document Full Text: \n{'-'*10}\n{doc.metadata.text}")

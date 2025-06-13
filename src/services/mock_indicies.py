"""
Mock index service module for testing purposes.

This module provides a MockQueryService that returns dummy data with "I don't know." content
to allow the system to run when real vector services are unavailable.
"""
from typing import Optional, List
import pandas as pd

# Import utilities
from utils.logging_utils import get_logger

# Import from the main indicies module
from services.indicies import SearchHit
from services.live_rag_metadata import LiveRAGMetadata


class MockQueryService:
    """
    Mock index service
    """

    # Class logger
    log = get_logger("MockQueryService")

    def __init__(self, **kwargs):
        """Initialize the mock service. Ignores all parameters."""
        self.log.info("Initialized MockQueryService - returning mock data only")

    def query_embedding(
        self,
        query: str,
        k: int = 10,
        namespace: Optional[str] = None,
        **kwargs
    ) -> List[SearchHit]:
        """
        Mock embedding query that returns a single document with "I don't know." content.
        
        Args:
            query: The query string (ignored)
            k: Number of results requested (ignored, always returns 1)
            namespace: Namespace (ignored)
            
        Returns:
            List containing a single SearchHit with mock data
        """
        self.log.debug("Mock embedding query", query=query, k=k)
        
        mock_metadata = LiveRAGMetadata(
            chunk_order=1.0,
            doc_id="<urn:uuid:mock-document-id>",
            is_first_chunk=True,
            is_last_chunk=True,
            text="I don't know.",
            total_doc_chunks=1.0
        )
        
        mock_hit = SearchHit(
            id="doc-<urn:uuid:mock-document-id>::chunk-0",
            score=1.0,
            metadata=mock_metadata,
            retrieval_model="mock-embedding"
        )
        
        return [mock_hit]

    def query_keywords(
        self,
        query: str,
        k: int = 10,
        **kwargs
    ) -> List[SearchHit]:
        """
        Mock keyword query that returns a single document with "I don't know." content.
        
        Args:
            query: The query string (ignored)
            k: Number of results requested (ignored, always returns 1)
            
        Returns:
            List containing a single SearchHit with mock data
        """
        self.log.debug("Mock keyword query", query=query, k=k)
        
        mock_metadata = LiveRAGMetadata(
            chunk_order=1.0,
            doc_id="<urn:uuid:mock-document-id>",
            is_first_chunk=True,
            is_last_chunk=True,
            text="I don't know.",
            total_doc_chunks=1.0
        )
        
        mock_hit = SearchHit(
            id="doc-<urn:uuid:mock-document-id>::chunk-0",
            score=1.0,
            metadata=mock_metadata,
            retrieval_model="mock-bm25"
        )
        
        return [mock_hit]

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
        Mock fusion query that returns a single document with "I don't know." content.
        
        Args:
            query: The query string (ignored)
            k: Number of results requested (ignored, always returns 1)
            per_source_k: Per-source results (ignored)
            return_raw_df: Whether to return DataFrame (respected)
            pk_namespace: Pinecone namespace (ignored)
            rrf_k: RRF parameter (ignored)
            rrf_max_docs: Max docs for RRF (ignored)
            
        Returns:
            List of SearchHit objects or DataFrame with mock data
        """
        self.log.debug("Mock fusion query", query=query, k=k)
        
        mock_metadata = LiveRAGMetadata(
            chunk_order=1.0,
            doc_id="<urn:uuid:mock-document-id>",
            is_first_chunk=True,
            is_last_chunk=True,
            text="I don't know.",
            total_doc_chunks=1.0
        )
        
        mock_hit = SearchHit(
            id="doc-<urn:uuid:mock-document-id>::chunk-0",
            score=1.0,
            metadata=mock_metadata,
            retrieval_model="mock-fusion"
        )
        
        if return_raw_df:
            # Return a DataFrame for fusion queries when requested
            mock_df = pd.DataFrame([{
                'query': query,
                'q0': 'q0',
                'docid': mock_hit.id,
                'rank': 1,
                'score': mock_hit.score,
                'system': 'mock'
            }])
            return mock_df
        
        return [mock_hit]

    def get_docs(self, doc_ids: List[str], size_per_doc: int = 20, combined=True) -> List[SearchHit]:
        """
        Mock get_docs that returns a single document with "I don't know." content for each requested ID.
        
        Args:
            doc_ids: List of document IDs to retrieve (used for count)
            size_per_doc: Maximum docs per ID (used to limit results)
            combined: Whether to combine results (ignored)
            
        Returns:
            List of SearchHit objects with mock data
        """
        self.log.debug("Mock get_docs", doc_ids=doc_ids[:3], total_count=len(doc_ids))
        
        results = []
        for i, doc_id in enumerate(doc_ids[:size_per_doc]):  # Limit to size_per_doc
            mock_metadata = LiveRAGMetadata(
                chunk_order=1.0,
                doc_id=doc_id,  # Use the actual requested doc_id
                is_first_chunk=True,
                is_last_chunk=True,
                text="I don't know.",
                total_doc_chunks=1.0
            )
            
            mock_hit = SearchHit(
                id=f"doc-{doc_id}::chunk-0",
                score=1.0,
                metadata=mock_metadata,
                retrieval_model="mock-retrieval"
            )
            
            results.append(mock_hit)
        
        return results

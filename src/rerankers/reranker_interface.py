"""
Common interface for RAG systems.

This module defines the abstract base class that all RAG systems should implement.
"""
from abc import ABC, abstractmethod
from typing import List

from services.indicies import SearchHit


class RerankerInterface(ABC):
    """
    Abstract base class for rerankers.
    """

    @abstractmethod
    def rerank(self, query: str, hits: List[SearchHit], k: int = None) -> List[SearchHit]:
        """
        Rerank the retrieved documents.

        Args:
            query: The user query
            hits: List of retrieved documents
            k: Number of documents to return after reranking (default: all)

        Returns:
            Reranked list of SearchHit objects
        """
        pass

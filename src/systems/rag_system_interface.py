"""
Common interface for RAG systems.

This module defines the abstract base class that all RAG systems should implement.
"""
from abc import ABC, abstractmethod
from typing import Optional

from systems.rag_result import RAGResult


class RAGSystemInterface(ABC):
    """
    Abstract base class for RAG systems.
    
    All RAG systems should implement this interface to ensure consistency
    and interoperability across different implementations.
    
    Refer to basic_rag_system.py for an example implementation.
    """
    
    @abstractmethod
    def process_question(self, question: str, qid: Optional[str] = None) -> RAGResult:
        """
        Process a question and generate an answer using RAG.
        
        Args:
            question: The user's question
            qid: Optional query ID, will be populated in RAGResult
            
        Returns:
            A RAGResult containing the answer and metadata
        """
        pass

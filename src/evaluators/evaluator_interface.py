"""
Common interface for RAG evaluators.

This module defines the abstract base class that all RAG evaluators should implement.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair
from evaluators.evaluation_result import EvaluationResult


class EvaluatorInterface(ABC):
    """
    Abstract base class for RAG evaluators.
    
    All RAG evaluators should implement this interface to ensure consistency
    and interoperability across different implementations.
    
    Refer to basic_evaluator/edit_distance_evaluator.py for an example implementation.
    
    This interface supports evaluation of multiple RAG results against reference QA pairs.
    """
    
    @abstractmethod
    def evaluate(self, rag_results: List[RAGResult], references: List[QAPair]) -> EvaluationResult:
        """
        Evaluate a list of RAG results against a list of reference QA pairs.
        
        The implementation should match RAG results with references using their qid fields.
        
        Args:
            rag_results: List of RAG results to evaluate
            references: List of reference QA pairs to compare against
            
        Returns:
            An EvaluationResult containing evaluation metrics and optionally row-level results
        """
        pass

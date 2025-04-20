"""
Interface for RAG system results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class RAGResult:
    """
    Interface for RAG system results.
    Contains the query, answer, context, document IDs, and performance metrics.
    Also includes generated queries (rewritten queries from given query) and
    rewritten documents (context, but each document is rewritten).
    """
    query: str
    answer: str
    context: List[str]
    doc_ids: List[str]
    query_words_count: int
    answer_words_count: int
    total_time_ms: float
    qid: str = None
    timestamp: datetime = field(default_factory=datetime.now)
    generated_queries: List[str] = None
    rewritten_docs: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "query": self.query,
            "answer": self.answer,
            "context": self.context,
            "doc_ids": self.doc_ids,
            "query_words_count": self.query_words_count,
            "answer_words_count": self.answer_words_count,
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "generated_queries": self.generated_queries,
            "rewritten_docs": self.rewritten_docs,
            "qid": self.qid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResult':
        """
        Create a RAGResult instance from a dictionary.
        
        Args:
            data: Dictionary containing result fields
            
        Returns:
            RAGResult instance with values from the dictionary
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            context=data.get("context", []),
            doc_ids=data.get("doc_ids", []),
            query_words_count=data.get("query_words_count", 0),
            answer_words_count=data.get("answer_words_count", 0),
            total_time_ms=data.get("total_time_ms", 0.0),
            timestamp=timestamp or datetime.now(),
            generated_queries=data.get("generated_queries", None),
            rewritten_docs=data.get("rewritten_docs", None),
            qid=data.get("qid", None)
        )

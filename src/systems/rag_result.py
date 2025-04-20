"""
Interface for RAG system results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class RAGResult:
    """
    Interface for RAG system results.
    Contains the question, answer, context, document IDs, and performance metrics.
    Also includes generated queries (rewritten queries from given question) and
    rewritten documents (context, but each document is rewritten).
    """
    question: str
    answer: str
    context: List[str]
    doc_ids: List[str]
    total_time_ms: float
    qid: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    generated_queries: Optional[List[str]] = None
    rewritten_docs: Optional[List[str]] = None
    question_words_count: Optional[int] = None
    answer_words_count: Optional[int] = None
    
    def __post_init__(self):
        """
        Calculate word counts if not provided.
        """
        if self.question_words_count is None:
            self.question_words_count = self.calculate_question_words_count()
        
        if self.answer_words_count is None:
            self.answer_words_count = self.calculate_answer_words_count()
    
    def calculate_question_words_count(self) -> int:
        """
        Calculate the number of words in the question.
        
        Returns:
            Number of words in the question
        """
        return len(self.question.split())
    
    def calculate_answer_words_count(self) -> int:
        """
        Calculate the number of words in the answer.
        
        Returns:
            Number of words in the answer
        """
        return len(self.answer.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "doc_ids": self.doc_ids,
            "question_words_count": self.question_words_count,
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
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            context=data.get("context", []),
            doc_ids=data.get("doc_ids", []),
            question_words_count=data.get("question_words_count", 0),
            answer_words_count=data.get("answer_words_count", 0),
            total_time_ms=data.get("total_time_ms", 0.0),
            timestamp=timestamp or datetime.now(),
            generated_queries=data.get("generated_queries", None),
            rewritten_docs=data.get("rewritten_docs", None),
            qid=data.get("qid", None)
        )

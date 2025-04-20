"""
Interface for evaluator results.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class EvaluationResultRow:
    """
    Interface for individual row-level evaluator results.
    Contains the evaluation metrics and metadata for a single RAG result.
    """
    qid: str
    metrics: Dict[str, Any]
    evaluator_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation result row to a dictionary.
        
        Returns:
            Dictionary representation of the evaluation result row
        """
        return {
            "qid": self.qid,
            "metrics": self.metrics,
            "evaluator_name": self.evaluator_name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResultRow':
        """
        Create an EvaluationResultRow instance from a dictionary.
        
        Args:
            data: Dictionary containing evaluation result fields
            
        Returns:
            EvaluationResultRow instance with values from the dictionary
        """
        return cls(
            qid=data.get("qid", ""),
            metrics=data.get("metrics", {}),
            evaluator_name=data.get("evaluator_name", ""),
        )


@dataclass
class EvaluationResult:
    """
    Interface for aggregated evaluator results.
    Contains the evaluation metrics and metadata for a collection of RAG results.
    Can optionally include row-level results for individual evaluations.
    """
    metrics: Dict[str, Any]
    evaluator_name: str
    sample_count: int
    system_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    rows: Optional[List[EvaluationResultRow]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation result to a dictionary.
        
        Returns:
            Dictionary representation of the evaluation result
        """
        result = {
            "metrics": self.metrics,
            "evaluator_name": self.evaluator_name,
            "sample_count": self.sample_count,
            "system_name": self.system_name,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.rows:
            result["rows"] = [row.to_dict() for row in self.rows]
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """
        Create an EvaluationResult instance from a dictionary.
        
        Args:
            data: Dictionary containing evaluation result fields
            
        Returns:
            EvaluationResult instance with values from the dictionary
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        rows_data = data.get("rows")
        rows = None
        if rows_data:
            rows = [EvaluationResultRow.from_dict(row) for row in rows_data]
        
        return cls(
            metrics=data.get("metrics", {}),
            evaluator_name=data.get("evaluator_name", ""),
            sample_count=data.get("sample_count", 0),
            system_name=data.get("system_name"),
            timestamp=timestamp or datetime.now(),
            rows=rows
        )

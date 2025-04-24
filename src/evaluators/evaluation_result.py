"""
Interface for evaluator results.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class SystemAnalysis:
    """
    Analysis of RAG system performance based on evaluation results.
    Contains insights, patterns, and recommendations for improving the system.
    """
    analysis: str
    samples_analyzed: int
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "analysis": self.analysis,
            "samples_analyzed": self.samples_analyzed
        }
        
        if self.token_usage is not None:
            result["token_usage"] = self.token_usage
            
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemAnalysis':
        return cls(
            analysis=data.get("analysis", ""),
            samples_analyzed=data.get("samples_analyzed", 0),
            token_usage=data.get("token_usage"),
            cost_usd=data.get("cost_usd")
        )


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
    total_time_ms: Optional[float] = None
    total_cost: Optional[float] = None
    system_analysis: Optional[SystemAnalysis] = None
    
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
            
        if self.total_time_ms is not None:
            result["total_time_ms"] = self.total_time_ms
            
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
            
        if self.system_analysis is not None:
            result["system_analysis"] = self.system_analysis.to_dict()
            
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
        
        system_analysis_data = data.get("system_analysis")
        system_analysis = None
        if system_analysis_data:
            system_analysis = SystemAnalysis.from_dict(system_analysis_data)
        
        return cls(
            metrics=data.get("metrics", {}),
            evaluator_name=data.get("evaluator_name", ""),
            sample_count=data.get("sample_count", 0),
            system_name=data.get("system_name"),
            timestamp=timestamp or datetime.now(),
            rows=rows,
            total_time_ms=data.get("total_time_ms"),
            total_cost=data.get("total_cost"),
            system_analysis=system_analysis
        )

import numpy as np
from typing import List, Dict, Optional

from utils.logging_utils import get_logger


class QPPService:
    """
    Query Performance Prediction service that implements various QPP methods.
    """
    
    log = get_logger("qpp_service")
    
    def __init__(self, default_k: int = 10):
        """
        Initialize the QPP service.
        
        Args:
            default_k: Default number of top documents to use for QPP calculation
        """
        self.default_k = default_k
        self.log.info("QPP Service initialized", default_k=default_k)
    
    def calculate_entropy_qpp(self, scores: List[float], k: Optional[int] = None) -> float:
        """
        Calculate entropy-based QPP score.
        
        Lower entropy indicates better query performance (more certainty in results).
        We return negative entropy so that higher values indicate better performance.
        
        Args:
            scores: List of retrieval scores
            k: Number of top documents to use (defaults to self.default_k)
            
        Returns:
            Negative entropy score (higher is better)
        """
        k = k or self.default_k
        # Use only top k scores
        top_scores = scores[:k] if len(scores) > k else scores
        
        if not top_scores:
            self.log.warning("No scores provided for entropy calculation")
            return 0.0
        
        # Apply softmax normalization to convert scores to probabilities
        probabilities = self._softmax_normalize(top_scores)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = 0.0
        for p in probabilities:
            if p > 0:  # Avoid log(0)
                entropy -= p * np.log(p)
        
        # Return negative entropy (higher is better)
        return -entropy
    
    def calculate_sd_qpp(self, scores: List[float], k: Optional[int] = None) -> float:
        """
        Calculate standard deviation based QPP score.
        
        Higher standard deviation indicates better query performance.
        
        Args:
            scores: List of retrieval scores
            k: Number of top documents to use (defaults to self.default_k)
            
        Returns:
            Standard deviation score
        """
        k = k or self.default_k
        # Use only top k scores
        top_scores = scores[:k] if len(scores) > k else scores
        
        if not top_scores or len(top_scores) < 2:
            self.log.warning("Insufficient scores provided for SD calculation")
            return 0.0
        
        # Calculate standard deviation
        return np.std(top_scores)
    
    def calculate_mean_qpp(self, scores: List[float], k: Optional[int] = None) -> float:
        """
        Calculate mean-based QPP score.
        
        Higher mean indicates better query performance.
        
        Args:
            scores: List of retrieval scores
            k: Number of top documents to use (defaults to self.default_k)
            
        Returns:
            Mean score
        """
        k = k or self.default_k
        # Use only top k scores
        top_scores = scores[:k] if len(scores) > k else scores
        
        if not top_scores:
            self.log.warning("No scores provided for mean calculation")
            return 0.0
        
        # Calculate mean
        return np.mean(top_scores)
    
    def calculate_confidence_score(self, qpp_scores: Dict[str, float]) -> float:
        """
        Calculate an overall confidence score based on multiple QPP scores.
        
        Args:
            qpp_scores: Dictionary of QPP method names and their scores
            
        Returns:
            Confidence score between 0 and 1
        """
        # If no scores provided, return 0
        if not qpp_scores:
            return 0.0
        
        # Normalize each score to [0, 1] range based on typical ranges
        normalized_scores = {}
        
        # Entropy typically ranges from -log(k) to 0, normalize to [0, 1]
        if 'entropy' in qpp_scores:
            # Entropy is already negative, so higher values are better
            # Normalize assuming typical range is [-5, 0]
            entropy = qpp_scores['entropy']
            normalized_scores['entropy'] = min(1.0, max(0.0, (entropy + 5) / 5))
        
        # SD typically ranges from 0 to some positive value, normalize to [0, 1]
        if 'sd' in qpp_scores:
            # Normalize assuming typical range is [0, 2]
            sd = qpp_scores['sd']
            normalized_scores['sd'] = min(1.0, max(0.0, sd / 2))
        
        # Mean typically ranges from some negative to positive value, normalize to [0, 1]
        if 'mean' in qpp_scores:
            # Normalize assuming typical range is [-2, 2]
            mean = qpp_scores['mean']
            normalized_scores['mean'] = min(1.0, max(0.0, (mean + 2) / 4))
        
        # Calculate weighted average of normalized scores
        # Weights based on empirical performance from the paper
        weights = {
            'entropy': 0.5,
            'sd': 0.3,
            'mean': 0.2
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for method, score in normalized_scores.items():
            if method in weights:
                weighted_sum += score * weights[method]
                total_weight += weights[method]
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _softmax_normalize(self, scores: List[float]) -> List[float]:
        """
        Apply softmax normalization to convert scores to probabilities.
        
        Args:
            scores: List of retrieval scores
            
        Returns:
            List of probabilities that sum to 1
        """
        # Shift scores to avoid numerical issues with large exponentials
        shifted_scores = np.array(scores) - np.max(scores)
        exp_scores = np.exp(shifted_scores)
        return exp_scores / np.sum(exp_scores)

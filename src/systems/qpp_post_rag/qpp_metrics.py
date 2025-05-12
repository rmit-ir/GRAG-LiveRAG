"""
QPP metrics calculation module for post-retrieval prediction.
"""
from typing import List, Tuple
import numpy as np
from scipy import stats

def calculate_pearson(relevance_scores: List[float], oracle_scores: List[float]) -> float:
    """Calculate Pearson correlation coefficient between predicted and oracle scores."""
    if len(relevance_scores) < 2:
        return 0.0
    correlation, _ = stats.pearsonr(relevance_scores, oracle_scores)
    return correlation

def calculate_spearman(relevance_scores: List[float], oracle_scores: List[float]) -> float:
    """Calculate Spearman correlation coefficient between predicted and oracle scores."""
    if len(relevance_scores) < 2:
        return 0.0
    correlation, _ = stats.spearmanr(relevance_scores, oracle_scores)
    return correlation

def calculate_kendall(relevance_scores: List[float], oracle_scores: List[float]) -> float:
    """Calculate Kendall Tau correlation coefficient between predicted and oracle scores."""
    if len(relevance_scores) < 2:
        return 0.0
    correlation, _ = stats.kendalltau(relevance_scores, oracle_scores)
    return correlation

def calculate_sare(relevance_scores: List[float], oracle_scores: List[float], k: int = 5) -> float:
    """
    Calculate scaled Aggregate Ranking  Error (sARE) metric.
    Lower sARE values indicate better prediction quality.
    """
    if len(relevance_scores) < k:
        return 1.0  # Maximum error
        
    # Get top-k indices for both predicted and oracle scores
    pred_top_k = np.argsort(relevance_scores)[-k:]
    oracle_top_k = np.argsort(oracle_scores)[-k:]
    
    # Calculate relevance error
    pred_sum = sum(oracle_scores[i] for i in pred_top_k)
    oracle_sum = sum(oracle_scores[i] for i in oracle_top_k)
    
    if oracle_sum == 0:
        return 1.0
        
    sare = 1 - (pred_sum / oracle_sum)
    return max(0.0, min(1.0, sare))  # Clip between 0 and 1


def calculate_overlap_coefficient(relevance_scores: List[float], oracle_scores: List[float], k: int = 5) -> float:
    """
    Calculate the overlap coefficient between predicted and oracle top-k items.
    
    The overlap coefficient measures the overlap between two sets and is defined as:
    overlap(A,B) = |A ∩ B| / min(|A|, |B|)
    
    Args:
        relevance_scores: List of predicted relevance scores
        oracle_scores: List of true/oracle scores
        k: Number of top items to consider
    
    Returns:
        float: Overlap coefficient between 0 and 1, where 1 indicates perfect overlap
    """
    if len(relevance_scores) < k or len(oracle_scores) < k:
        k = min(len(relevance_scores), len(oracle_scores))
    
    # Get top-k indices for both predicted and oracle scores
    pred_top_k = set(np.argsort(relevance_scores)[-k:])
    oracle_top_k = set(np.argsort(oracle_scores)[-k:])
    
    # Calculate overlap coefficient
    intersection = len(pred_top_k.intersection(oracle_top_k))
    denominator = min(len(pred_top_k), len(oracle_top_k))
    
    if denominator == 0:
        return 0.0
    
    return intersection / denominator

def calculate_jaccard_index(relevance_scores: List[float], oracle_scores: List[float], k: int = 5) -> float:
    """
    Calculate the Jaccard index between predicted and oracle top-k items.
    
    The Jaccard index measures similarity between finite sets and is defined as:
    J(A,B) = |A ∩ B| / |A ∪ B|
    
    Args:
        relevance_scores: List of predicted relevance scores
        oracle_scores: List of true/oracle scores
        k: Number of top items to consider
    
    Returns:  
        float: Jaccard index between 0 and 1, where 1 indicates identical sets
    """
    if len(relevance_scores) < k or len(oracle_scores) < k:
        k = min(len(relevance_scores), len(oracle_scores))
    
    # Get top-k indices for both predicted and oracle scores
    pred_top_k = set(np.argsort(relevance_scores)[-k:])
    oracle_top_k = set(np.argsort(oracle_scores)[-k:])
    
    # Calculate Jaccard index
    intersection = len(pred_top_k.intersection(oracle_top_k))
    union = len(pred_top_k.union(oracle_top_k))
    
    if union == 0:  # This shouldn't happen with non-empty sets, but just in case
        return 0.0
    
    return intersection / union

def calculate_rbo(relevance_scores: List[float], oracle_scores: List[float], p: float = 0.9) -> float:
    """
    Calculate Rank-Biased Overlap (RBO) between predicted and oracle rankings.
    
    RBO measures the similarity between two rankings, with emphasis on agreement 
    at higher ranks. It's particularly useful for comparing incomplete or 
    non-conjoint rankings.
    
    Args:
        relevance_scores: List of predicted relevance scores
        oracle_scores: List of true/oracle scores
        p: Persistence parameter between 0 and 1, controlling the top-weightedness
           (higher values emphasize agreement in the full list, lower values 
            emphasize agreement at the very top ranks)
    
    Returns:
        float: RBO score between 0 and 1, where 1 indicates identical rankings
               and 0 indicates completely different rankings
    """
    if len(relevance_scores) == 0 or len(oracle_scores) == 0:
        return 0.0

    # Convert scores to rankings (argsort returns indices that would sort the array)
    # We want highest scores to have rank 0, so we reverse with [::-1]
    pred_ranking = np.argsort(relevance_scores)[::-1]
    oracle_ranking = np.argsort(oracle_scores)[::-1]
    
    # Convert rankings to dictionaries with position
    pred_dict = {item: pos for pos, item in enumerate(pred_ranking)}
    oracle_dict = {item: pos for pos, item in enumerate(oracle_ranking)}
    
    # Get all unique items from both rankings
    all_items = set(pred_dict.keys()).union(set(oracle_dict.keys()))
    
    # Calculate RBO
    rbo = 0.0
    sum_overlap = 0.0
    
    # Get max possible depth (length of the longer ranking)
    max_depth = max(len(relevance_scores), len(oracle_scores))
    
    for d in range(1, max_depth + 1):
        # Get the overlap at depth d
        pred_items = set([pred_ranking[i] for i in range(min(d, len(pred_ranking)))])
        oracle_items = set([oracle_ranking[i] for i in range(min(d, len(oracle_ranking)))])
        
        overlap_at_d = len(pred_items.intersection(oracle_items))
        agreement_at_d = overlap_at_d / d
        
        # RBO formula component: agreement at depth d weighted by p^(d-1)
        sum_overlap += agreement_at_d * (p ** (d - 1))
    
    # Calculate the final RBO score
    rbo = (1 - p) * sum_overlap
    
    return rbo

def calculate_smare(relevance_scores: List[float], oracle_scores: List[float], k: int = None) -> float:
    """
    Calculate scaled Mean Absolute Rank Error (sMARE) metric.
    Lower sMARE values indicate better prediction quality.
    
    Args:
        relevance_scores: List of predicted relevance scores
        oracle_scores: List of true/oracle scores
        k: Optional parameter to consider only top-k rankings. If None, uses all ranks.
    
    Returns:
        float: sMARE score between 0 and 1, where 0 indicates perfect rank correlation
    """
    if len(relevance_scores) < 2:
        return 1.0  # Maximum error
        
    n = len(relevance_scores)
    k = k or n  # If k is None, use full length
    k = min(k, n)  # Ensure k doesn't exceed length of lists
    
    # Get rankings (with 0 being the highest rank)
    pred_ranks = n - 1 - np.argsort(np.argsort(relevance_scores))
    oracle_ranks = n - 1 - np.argsort(np.argsort(oracle_scores))
    
    # Calculate absolute rank differences for top-k items by oracle ranking
    top_k_indices = np.argsort(oracle_scores)[-k:]
    rank_diffs = np.abs(pred_ranks[top_k_indices] - oracle_ranks[top_k_indices])
    
    # Scale by maximum possible rank difference (n-1)
    smare = np.mean(rank_diffs) / (n - 1)
    
    return float(max(0.0, min(1.0, smare)))  # Clip between 0 and 1

def calculate_qpp_scores(
    query: str,
    passages: List[str],
    relevance_assessor,
    batch_size: int = 5
) -> Tuple[List[float], List[float]]:
    """
    Calculate QPP scores for a set of passages using LLM-based relevance assessment.
    Returns predicted relevance scores and oracle scores.
    """
    relevance_scores = []
    oracle_scores = []
    
    # Process passages in batches to be more efficient with LLM calls
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        batch_scores = relevance_assessor.assess_batch(query, batch)
        relevance_scores.extend(batch_scores)
        
        # For oracle scores, we use binary relevance based on exact match
        # In a real system, this would be replaced with human annotations or a stronger oracle
        oracle_batch = [1.0 if query.lower() in p.lower() else 0.0 for p in batch]
        oracle_scores.extend(oracle_batch)
    
    return relevance_scores, oracle_scores
"""
Utility functions for fusion of search results.
"""
from typing import List, Dict
import pandas as pd
from trectools import TrecRun, fusion

from utils.logging_utils import get_logger
from services.indicies import SearchHit
from utils.namedtuple_utils import update_tuple

logger = get_logger("fusion_utils")

def rrf_fusion(hits_list: List[List[SearchHit]], max_documents: int, query_id: str = "query", rrf_k: int = 60) -> List[SearchHit]:
    """
    Applies reciprocal rank fusion to multiple lists of SearchHit objects using trectools.
    
    TODO: support weights...
    
    Args:
        hits_list: List of lists of SearchHit objects to fuse
        max_documents: Maximum number of documents to return
        query_id: Query ID to use for the TrecRun objects
        rrf_k: Parameter for reciprocal rank fusion (default: 60)
        
    Returns:
        List of SearchHit objects containing the top max_documents documents after fusion
    """
    logger.debug("Preparing for fusion", 
                hits_list_count=len(hits_list), 
                max_documents=max_documents)
    
    if not hits_list or all(not hits for hits in hits_list):
        logger.warning("No hits to fuse")
        return []
    
    # Convert each list of hits to a TrecRun object
    trec_runs = []
    all_hits = []
    
    for i, hits in enumerate(hits_list):
        if hits:
            trec_run = _hits_to_trecrun(query_id, hits, f"run_{i}")
            trec_runs.append(trec_run)
            all_hits.extend(hits)
    
    if not trec_runs:
        logger.warning("No valid TrecRun objects created")
        return []
    
    # Apply reciprocal rank fusion
    fused_run = fusion.reciprocal_rank_fusion(trec_runs, k=rrf_k, max_docs=max_documents)
    
    # Get the top max_documents results from the fused run
    fused_df = fused_run.run_data.head(max_documents)
    
    # Create a dictionary to map document IDs to their original SearchHit objects
    hit_map = {hit.id: hit for hit in all_hits}
    
    # Directly create a list of SearchHit objects from the fused results
    result = []
    for _, row in fused_df.iterrows():
        doc_id = row['docid']
        if doc_id in hit_map:
            # Use the original SearchHit object but update the score
            hit = hit_map[doc_id]
            # Create a new SearchHit with the fused score
            result.append(update_tuple(hit, score=row['score']))
    
    logger.debug("Fusion completed", result_count=len(result))
    return result

def _hits_to_trecrun(query_id: str, hits: List[SearchHit], tag: str) -> TrecRun:
    """
    Helper function to convert SearchHit objects to trectools TrecRun object.
    """
    # Create rows for the TrecRun dataframe
    rows = [{
        'query': query_id,
        'q0': 'q0',
        'docid': h.id,
        'rank': i+1,
        'score': h.score,
        'system': tag
    } for i, h in enumerate(hits)]

    # Create a TrecRun object
    query_run = TrecRun(None)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.columns = ["query", "q0", "docid", "rank", "score", "system"]
        query_run.load_run_from_dataframe(df)
    return query_run

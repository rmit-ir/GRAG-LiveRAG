"""
Utility functions for working with trectools in the context of RAG evaluation.

This module provides functions to convert RAG results and references to trectools format
and calculate metrics like NDCG@k.
"""
import os
import tempfile
from typing import Dict, List, Set, Tuple, Any

from trectools import TrecEval, TrecRun, TrecQrel
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair
from utils.logging_utils import get_logger

logger = get_logger("trec_utils")


def extract_document_id(chunk_id: str) -> str:
    """
    Extract the document UUID from a chunk ID.
    
    Args:
        chunk_id: The chunk ID in format 'doc-<urn:uuid:UUID>::chunk-N'
        
    Returns:
        The extracted document UUID in format '<urn:uuid:UUID>'
    """
    import re
    # Use regex to extract the UUID part
    match = re.search(r'<(urn:uuid:[^>]+)>', chunk_id)
    if match:
        return f"<{match.group(1)}>"
    return chunk_id  # Return original if no match


def create_trec_files(rag_results: List[RAGResult], references: Dict[str, QAPair]) -> Tuple[str, str]:
    """
    Create temporary TREC run and qrel files from RAG results and references.
    
    Args:
        rag_results: List of RAG results
        references: Dictionary mapping qids to reference QA pairs
        
    Returns:
        Tuple of (run_file_path, qrel_file_path)
    """
    # Create temporary files for trectools
    run_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    qrel_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    
    run_path = run_file.name
    qrel_path = qrel_file.name
    
    try:
        # Write run file (system results)
        # Format: qid Q0 docid rank score system
        for rag_result in rag_results:
            qid = rag_result.qid
            if qid not in references:
                continue
                
            for rank, doc_id in enumerate(rag_result.doc_ids, 1):
                # Use original doc_id as identifier
                run_file.write(f"{qid} Q0 {doc_id} {rank} {1000-rank} system\n")
        
        # Write qrel file (relevance judgments)
        # Format: qid 0 docid relevance
        for rag_result in rag_results:
            qid = rag_result.qid
            if qid not in references:
                continue
                
            reference = references[qid]
            gold_doc_ids = set(reference.document_ids)
            
            # For each document in the run, check if it's in the gold set
            for doc_id in rag_result.doc_ids:
                extracted_id = extract_document_id(doc_id)
                # kun: if not found in gold_doc_ids, 0
                relevance = 1 if extracted_id in gold_doc_ids else 0
                qrel_file.write(f"{qid} 0 {doc_id} {relevance}\n")
    
    finally:
        run_file.close()
        qrel_file.close()
    
    return run_path, qrel_path


def calculate_ndcg(run_path: str, qrel_path: str, k: int = 10) -> Dict[str, float]:
    """
    Calculate NDCG@k using trectools.
    
    Args:
        run_path: Path to the TREC run file
        qrel_path: Path to the TREC qrel file
        k: The cutoff for NDCG calculation (default: 10)
        
    Returns:
        Dictionary with NDCG scores
    """
    try:
        # Load run and qrel files
        run = TrecRun(run_path)
        qrels = TrecQrel(qrel_path)
        
        # Create a TrecEval object for evaluation
        te = TrecEval(run, qrels)
        
        # Calculate NDCG@k for overall score
        ndcg_value = te.get_ndcg(depth=k)
        
        # # Get per-query NDCG scores
        # per_query_ndcg = te.get_ndcg(depth=k, per_query=True)
        
        return ndcg_value
    
    except Exception as e:
        logger.error("Error calculating NDCG", error=str(e))
        return 0.0
    
    finally:
        # Clean up temporary files
        try:
            os.remove(run_path)
            os.remove(qrel_path)
        except Exception as e:
            logger.warning("Error removing temporary files", error=str(e))


def calculate_ndcg_for_results(rag_results: List[RAGResult], references: List[QAPair], k: int = 10) -> Dict[str, Any]:
    """
    Calculate NDCG@k for a list of RAG results and references.
    
    Args:
        rag_results: List of RAG results
        references: List of reference QA pairs
        k: The cutoff for NDCG calculation (default: 10)
        
    Returns:
        Dictionary with NDCG scores
    """
    # Create a mapping of QA pair qids to QA pairs for faster lookup
    reference_map = {ref.qid: ref for ref in references}
    
    # Filter out results without matching references
    valid_results = [r for r in rag_results if r.qid and r.qid in reference_map]
    
    if not valid_results:
        logger.warning("No valid results found for NDCG calculation")
        return {f"ndcg_{k}": 0.0}
    
    # Create TREC files
    run_path, qrel_path = create_trec_files(valid_results, reference_map)
    
    # Calculate NDCG
    return calculate_ndcg(run_path, qrel_path, k)

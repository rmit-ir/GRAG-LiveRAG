"""
Context Recall Evaluator for RAG systems.

uv run scripts/evaluate.py --evaluator evaluators.context_recall.evaluator.ContextRecall --help
"""
import re
import time
import statistics
from typing import Dict, Any, List

from evaluators.context_recall import trec_utils

from utils.logging_utils import get_logger
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair
from evaluators.evaluator_interface import EvaluatorInterface
from evaluators.evaluation_result import EvaluationResult, EvaluationResultRow


class ContextRecall(EvaluatorInterface):
    """
    Evaluator that calculates the recall of retrieved documents compared to gold context documents.
    
    This evaluator extracts document IDs from retrieved chunks and compares them with gold context
    document IDs to calculate recall metrics. It handles the different formats of IDs between
    retrieved chunks and gold context.
    
    Retrieved chunks format: doc-<urn:uuid:27c682dc-c97b-4c3e-b41c-999c65264a6d>::chunk-0
    Gold context format: ['<urn:uuid:417fbc69-80f9-4b53-8b5c-1c9cdd2bb0e9>']
    """
    
    log = get_logger("context_recall_evaluator")
    
    def __init__(self):
        """
        Initialize the ContextRecall.
        """
        self.log.info("ContextRecall initialized")
    
    def _extract_document_id(self, chunk_id: str) -> str:
        """
        Extract the document UUID from a chunk ID.
        
        Args:
            chunk_id: The chunk ID in format 'doc-<urn:uuid:UUID>::chunk-N'
            
        Returns:
            The extracted document UUID in format '<urn:uuid:UUID>'
        """
        # Use regex to extract the UUID part
        match = re.search(r'<(urn:uuid:[^>]+)>', chunk_id)
        if match:
            return f"<{match.group(1)}>"
        return chunk_id  # Return original if no match
    
    def _evaluate_single(self, rag_result: RAGResult, reference: QAPair) -> Dict[str, Any]:
        """
        Evaluate a single RAG result against a reference QA pair.
        
        Args:
            rag_result: The RAG result to evaluate
            reference: The reference QA pair to compare against
            
        Returns:
            A dictionary containing the recall metrics
        """
        # Extract document IDs from retrieved chunks
        retrieved_doc_ids = set()
        for chunk_id in rag_result.doc_ids:
            doc_id = self._extract_document_id(chunk_id)
            retrieved_doc_ids.add(doc_id)
        
        # Get gold context document IDs
        gold_doc_ids = set(reference.document_ids)
        
        # Calculate intersection and metrics
        correct_docs = retrieved_doc_ids.intersection(gold_doc_ids)
        
        # Calculate recall metrics
        recall = len(correct_docs) / len(gold_doc_ids) if gold_doc_ids else 0.0
        precision = len(correct_docs) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Create metrics dictionary
        metrics = {
            "context_recall": recall,
            "context_precision": precision,
            "context_f1": f1,
            "retrieved_docs_count": len(retrieved_doc_ids),
            "gold_docs_count": len(gold_doc_ids),
            "correct_docs_count": len(correct_docs)
        }
        
        return metrics
    
    def evaluate(self, rag_results: List[RAGResult], references: List[QAPair]) -> EvaluationResult:
        """
        Evaluate a list of RAG results against a list of reference QA pairs.
        
        Args:
            rag_results: List of RAG results to evaluate
            references: List of reference QA pairs to compare against
            
        Returns:
            An EvaluationResult containing the aggregated recall metrics and row-level results
        """
        start_time = time.time()
        self.log.info("Evaluating RAG results", result_count=len(rag_results), reference_count=len(references))
        
        # Create a mapping of QA pair qids to QA pairs for faster lookup
        reference_map = {ref.qid: ref for ref in references}
        
        # Evaluate each RAG result and create row-level results
        rows = []
        recalls = []
        precisions = []
        f1_scores = []
        records_with_at_least_one_ref = 0
        records_with_all_refs = 0
        
        for rag_result in rag_results:
            if rag_result.qid and rag_result.qid in reference_map:
                reference = reference_map[rag_result.qid]
                
                # Evaluate the single result
                metrics = self._evaluate_single(rag_result, reference)
                
                # Add to aggregation lists
                recalls.append(metrics["context_recall"])
                precisions.append(metrics["context_precision"])
                f1_scores.append(metrics["context_f1"])
                
                # Track records with at least one reference document
                if metrics["correct_docs_count"] > 0:
                    records_with_at_least_one_ref += 1
                
                # Track records with all reference documents
                if metrics["correct_docs_count"] == metrics["gold_docs_count"]:
                    records_with_all_refs += 1
                
                # Create row-level result
                row = EvaluationResultRow(
                    qid=rag_result.qid,
                    metrics=metrics,
                    evaluator_name=self.__class__.__name__
                )
                rows.append(row)
                
                # Calculate NDCG@10 for individual result if not already in metrics
                if "ndcg_10" not in metrics and len(rag_result.doc_ids) > 0:
                    # Create a single-item list for this result
                    single_result_ndcg = trec_utils.calculate_ndcg_for_results([rag_result], [reference], k=10)
                    metrics["ndcg_10"] = single_result_ndcg
                
                self.log.info("Evaluated result", 
                             qid=rag_result.qid, 
                             context_recall=metrics["context_recall"],
                             context_precision=metrics["context_precision"],
                             context_f1=metrics["context_f1"],
                             ndcg_10=metrics.get("ndcg_10", 0.0))
            else:
                self.log.warning("No matching reference found for result", qid=rag_result.qid)
        
        # Calculate aggregated metrics
        aggregated_metrics = {}
        
        if recalls:
            aggregated_metrics["avg_context_recall"] = statistics.mean(recalls)
            aggregated_metrics["avg_context_precision"] = statistics.mean(precisions)
            aggregated_metrics["avg_context_f1"] = statistics.mean(f1_scores)
            
            # Calculate percentage of records with at least one reference document
            aggregated_metrics["pct_records_with_at_least_one_ref"] = (records_with_at_least_one_ref / len(rows)) * 100
            
            # Calculate percentage of records with all reference documents
            aggregated_metrics["pct_records_with_all_refs"] = (records_with_all_refs / len(rows)) * 100
            
            # Calculate NDCG@10 using trectools
            self.log.info("Calculating NDCG@10 using trectools")
            aggregated_ndcg = trec_utils.calculate_ndcg_for_results(rag_results, references, k=10)
            aggregated_metrics["ndcg_10"] = aggregated_ndcg
        
        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000
        aggregated_metrics["processing_time_ms"] = total_time_ms
        
        self.log.info("Evaluation completed", 
                     evaluated_count=len(rows),
                     avg_context_recall=aggregated_metrics.get("avg_context_recall"),
                     avg_context_precision=aggregated_metrics.get("avg_context_precision"),
                     avg_context_f1=aggregated_metrics.get("avg_context_f1"),
                     ndcg_10=aggregated_metrics.get("ndcg_10"),
                     pct_records_with_at_least_one_ref=aggregated_metrics.get("pct_records_with_at_least_one_ref"),
                     pct_records_with_all_refs=aggregated_metrics.get("pct_records_with_all_refs"),
                     processing_time_ms=total_time_ms)
        
        # Get system_name from the first RAG result if available
        system_name = None
        if rag_results and hasattr(rag_results[0], 'system_name'):
            system_name = rag_results[0].system_name
        
        # Create and return the evaluation result
        return EvaluationResult(
            metrics=aggregated_metrics,
            evaluator_name=self.__class__.__name__,
            sample_count=len(rows),
            rows=rows,
            system_name=system_name,
            total_time_ms=total_time_ms
        )

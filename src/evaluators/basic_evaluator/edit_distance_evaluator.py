"""
Edit Distance Evaluator for RAG systems.

This evaluator calculates the minimum edit distance (Levenshtein distance)
between the generated answer and the reference answer using the Levenshtein package.
"""
import time
import statistics
import Levenshtein
from typing import Dict, Any, List

from utils.logging_utils import get_logger
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair
from evaluators.evaluator_interface import EvaluatorInterface
from evaluators.evaluation_result import EvaluationResult, EvaluationResultRow


class EditDistanceEvaluator(EvaluatorInterface):
    """
    Evaluator that calculates the minimum edit distance between the generated answer
    and the reference answer.
    
    The edit distance is a measure of how many single-character edits (insertions,
    deletions, or substitutions) are needed to change one string into another.
    A lower edit distance indicates higher similarity between the strings.
    """
    
    log = get_logger("edit_distance_evaluator")
    
    def __init__(self, normalize: bool = True):
        """
        Initialize the EditDistanceEvaluator.
        
        Args:
            normalize: Whether to normalize the edit distance by the length of the
                      longer string. This results in a value between 0 and 1, where
                      0 means the strings are identical and 1 means they are completely
                      different.
        """
        self.normalize = normalize
        self.log.info("EditDistanceEvaluator initialized", normalize=normalize)
    
    def _evaluate_single(self, rag_result: RAGResult, reference: QAPair) -> Dict[str, Any]:
        """
        Evaluate a single RAG result against a reference QA pair using edit distance.
        
        Args:
            rag_result: The RAG result to evaluate
            reference: The reference QA pair to compare against
            
        Returns:
            A dictionary containing the edit distance metrics
        """
        # Calculate the edit distance using the Levenshtein package
        edit_distance = Levenshtein.distance(rag_result.answer, reference.answer)
        
        # Normalize if requested
        normalized_distance = None
        if self.normalize:
            max_length = max(len(rag_result.answer), len(reference.answer))
            normalized_distance = edit_distance / max_length if max_length > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            "edit_distance": edit_distance
        }
        
        if normalized_distance is not None:
            metrics["normalized_distance"] = normalized_distance
        
        return metrics
    
    def evaluate(self, rag_results: List[RAGResult], references: List[QAPair]) -> EvaluationResult:
        """
        Evaluate a list of RAG results against a list of reference QA pairs using edit distance.
        RAGResult and QAPair are matched using their qid fields.
        
        Args:
            rag_results: List of RAG results to evaluate
            references: List of reference QA pairs to compare against
            
        Returns:
            An EvaluationResult containing the aggregated edit distance metrics and row-level results
        """
        start_time = time.time()
        self.log.info("Evaluating RAG results", result_count=len(rag_results), reference_count=len(references))
        
        # Create a mapping of QA pair qids to QA pairs for faster lookup
        reference_map = {ref.qid: ref for ref in references}
        
        # Evaluate each RAG result and create row-level results
        rows = []
        edit_distances = []
        normalized_distances = []
        
        for rag_result in rag_results:
            if rag_result.qid and rag_result.qid in reference_map:
                reference = reference_map[rag_result.qid]
                
                # Evaluate the single result
                metrics = self._evaluate_single(rag_result, reference)
                
                # Add to aggregation lists
                edit_distances.append(metrics["edit_distance"])
                if "normalized_distance" in metrics:
                    normalized_distances.append(metrics["normalized_distance"])
                
                # Create row-level result
                row = EvaluationResultRow(
                    qid=rag_result.qid,
                    metrics=metrics,
                    evaluator_name=self.__class__.__name__
                )
                rows.append(row)
                
                self.log.info("Evaluated result", 
                             qid=rag_result.qid, 
                             edit_distance=metrics["edit_distance"],
                             normalized_distance=metrics.get("normalized_distance"))
            else:
                self.log.warning("No matching reference found for result", qid=rag_result.qid)
        
        # Calculate aggregated metrics
        aggregated_metrics = {}
        
        if edit_distances:
            aggregated_metrics["avg_edit_distance"] = statistics.mean(edit_distances)
            aggregated_metrics["min_edit_distance"] = min(edit_distances)
            aggregated_metrics["max_edit_distance"] = max(edit_distances)
            aggregated_metrics["median_edit_distance"] = statistics.median(edit_distances)
        
        if normalized_distances:
            aggregated_metrics["avg_normalized_distance"] = statistics.mean(normalized_distances)
            aggregated_metrics["min_normalized_distance"] = min(normalized_distances)
            aggregated_metrics["max_normalized_distance"] = max(normalized_distances)
            aggregated_metrics["median_normalized_distance"] = statistics.median(normalized_distances)
        
        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000
        aggregated_metrics["processing_time_ms"] = total_time_ms
        
        self.log.info("Evaluation completed", 
                     evaluated_count=len(rows),
                     avg_edit_distance=aggregated_metrics.get("avg_edit_distance"),
                     avg_normalized_distance=aggregated_metrics.get("avg_normalized_distance"),
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
            system_name=system_name
        )


# Main entry point for testing the evaluator
# Just for demonstration purposes, no need to have __main__ in other evaluators
if __name__ == "__main__":
    from systems.rag_result import RAGResult
    
    # Create sample RAG results
    rag_result1 = RAGResult(
        question="What is retrieval-augmented generation?",
        answer="Retrieval-augmented generation (RAG) is a technique that combines retrieval of relevant documents with text generation.",
        context=["Retrieval-augmented generation (RAG) is an AI framework that combines information retrieval with text generation."],
        doc_ids=["doc1"],
        total_time_ms=100.0,
        qid="1",
        system_name="TestRAGSystem"
    )
    
    rag_result2 = RAGResult(
        question="How does RAG work?",
        answer="RAG works by retrieving relevant documents and then using them to generate text.",
        context=["RAG works by first retrieving relevant documents from a corpus and then using them to condition a language model."],
        doc_ids=["doc2"],
        total_time_ms=120.0,
        qid="2",
        system_name="TestRAGSystem"
    )
    
    # Create sample reference QA pairs
    reference1 = QAPair(
        question="What is retrieval-augmented generation?",
        answer="Retrieval-augmented generation (RAG) is an AI framework that enhances large language models by retrieving external knowledge.",
        context=["Retrieval-augmented generation (RAG) is an AI framework that enhances large language models by retrieving external knowledge."],
        question_categories=[],
        user_categories=[],
        document_ids=["doc1"],
        qid="1"
    )
    
    reference2 = QAPair(
        question="How does RAG work?",
        answer="RAG works by first retrieving relevant documents from a corpus and then using them to condition a language model.",
        context=["RAG works by first retrieving relevant documents from a corpus and then using them to condition a language model."],
        question_categories=[],
        user_categories=[],
        document_ids=["doc2"],
        qid="2"
    )
    
    # Initialize the evaluator
    evaluator = EditDistanceEvaluator(normalize=True)
    
    # Evaluate the RAG results
    result = evaluator.evaluate([rag_result1, rag_result2], [reference1, reference2])
    
    # Print the aggregated result
    print("\n" + "="*80)
    print("Aggregated Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print row-level results
    if result.rows:
        print("\nRow-level Results:")
        for row in result.rows:
            print(f"\nQID: {row.qid}")
            for key, value in row.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print("="*80 + "\n")

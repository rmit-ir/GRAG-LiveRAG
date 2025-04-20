"""
Vector Similarity Evaluator for RAG systems.

This evaluator calculates the cosine similarity between the embeddings of the
generated answer and the reference answer using the EmbeddingUtils.
"""
import time
import statistics
import numpy as np
from typing import Dict, Any, List

from utils.logging_utils import get_logger
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair
from services.embedding_utils import EmbeddingUtils
from evaluators.evaluator_interface import EvaluatorInterface
from evaluators.evaluation_result import EvaluationResult, EvaluationResultRow


class VectorSimEvaluator(EvaluatorInterface):
    """
    Evaluator that calculates the cosine similarity between the embeddings of the
    generated answer and the reference answer.
    
    The cosine similarity measures the cosine of the angle between two vectors,
    providing a similarity score between -1 and 1, where 1 means the vectors are
    identical, 0 means they are orthogonal, and -1 means they are diametrically opposed.
    A higher cosine similarity indicates higher semantic similarity between the texts.
    """
    
    log = get_logger("vector_sim_evaluator")
    
    def __init__(self, 
                 embedding_model_name: str = "intfloat/e5-base-v2",
                 pooling: str = "avg",
                 normalize: bool = True):
        """
        Initialize the VectorSimEvaluator.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            pooling: Pooling strategy to use (cls or avg)
            normalize: Whether to normalize the embeddings
        """
        self.embedding_utils = EmbeddingUtils(embedding_model_name=embedding_model_name)
        self.pooling = pooling
        self.normalize = normalize
        self.log.info("VectorSimEvaluator initialized", 
                     embedding_model=embedding_model_name,
                     pooling=pooling,
                     normalize=normalize)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity between the vectors
        """
        # Convert to numpy arrays for easier calculation
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1_np, vec2_np)
        norm_vec1 = np.linalg.norm(vec1_np)
        norm_vec2 = np.linalg.norm(vec2_np)
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def _evaluate_single(self, rag_result: RAGResult, reference: QAPair) -> Dict[str, Any]:
        """
        Evaluate a single RAG result against a reference QA pair using vector similarity.
        
        Args:
            rag_result: The RAG result to evaluate
            reference: The reference QA pair to compare against
            
        Returns:
            A dictionary containing the vector similarity metrics
        """
        # Generate embeddings for the generated answer and reference answer
        generated_embedding = self.embedding_utils.embed_query(
            rag_result.answer, 
            query_prefix="answer: ", 
            pooling=self.pooling, 
            normalize=self.normalize
        )
        
        reference_embedding = self.embedding_utils.embed_query(
            reference.answer, 
            query_prefix="answer: ", 
            pooling=self.pooling, 
            normalize=self.normalize
        )
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(generated_embedding, reference_embedding)
        
        # Create metrics dictionary
        metrics = {
            "cosine_similarity": similarity
        }
        
        return metrics
    
    def evaluate(self, rag_results: List[RAGResult], references: List[QAPair]) -> EvaluationResult:
        """
        Evaluate a list of RAG results against a list of reference QA pairs using vector similarity.
        RAGResult and QAPair are matched using their qid fields.
        
        Args:
            rag_results: List of RAG results to evaluate
            references: List of reference QA pairs to compare against
            
        Returns:
            An EvaluationResult containing the aggregated vector similarity metrics and row-level results
        """
        start_time = time.time()
        self.log.info("Evaluating RAG results", result_count=len(rag_results), reference_count=len(references))
        
        # Create a mapping of QA pair qids to QA pairs for faster lookup
        reference_map = {ref.qid: ref for ref in references}
        
        # Evaluate each RAG result and create row-level results
        rows = []
        similarities = []
        
        for rag_result in rag_results:
            if rag_result.qid and rag_result.qid in reference_map:
                reference = reference_map[rag_result.qid]
                
                # Evaluate the single result
                metrics = self._evaluate_single(rag_result, reference)
                
                # Add to aggregation lists
                similarities.append(metrics["cosine_similarity"])
                
                # Create row-level result
                row = EvaluationResultRow(
                    qid=rag_result.qid,
                    metrics=metrics,
                    evaluator_name=self.__class__.__name__
                )
                rows.append(row)
                
                self.log.info("Evaluated result", 
                             qid=rag_result.qid, 
                             cosine_similarity=metrics["cosine_similarity"])
            else:
                self.log.warning("No matching reference found for result", qid=rag_result.qid)
        
        # Calculate aggregated metrics
        aggregated_metrics = {}
        
        if similarities:
            aggregated_metrics["avg_cosine_similarity"] = statistics.mean(similarities)
            aggregated_metrics["min_cosine_similarity"] = min(similarities)
            aggregated_metrics["max_cosine_similarity"] = max(similarities)
            aggregated_metrics["median_cosine_similarity"] = statistics.median(similarities)
        
        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000
        aggregated_metrics["processing_time_ms"] = total_time_ms
        
        self.log.info("Evaluation completed", 
                     evaluated_count=len(rows),
                     avg_cosine_similarity=aggregated_metrics.get("avg_cosine_similarity"),
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

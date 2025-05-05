"""QPP Post-retrieval RAG system that uses sARE metrics to select the best query variant."""
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
from utils.logging_utils import get_logger
from services.indicies import QueryService, SearchHit 
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.qpp_post_rag.qpp_metrics import calculate_sare
from systems.qpp_post_rag.relevance_assessor import RelevanceAssessor

class QPPPostRAG(RAGSystemInterface):
    log = get_logger("qpp_post_rag")

    def __init__(self, llm_client='ai71', k=10, relevance_threshold=0.6):
        model_id = "tiiuae/falcon3-10b-instruct"
        if llm_client == 'ai71':
            self.rag_llm_client = AI71Client(model_id=model_id)
            self.qgen_llm_client = AI71Client(model_id=model_id)
        elif llm_client == 'ec2_llm':
            self.rag_llm_client = EC2LLMClient(model_id=model_id)
            self.qgen_llm_client = EC2LLMClient(model_id=model_id)

        self.k = k
        self.query_service = QueryService()
        self.relevance_assessor = RelevanceAssessor(llm_client=llm_client)
        self.relevance_threshold = relevance_threshold
        self.rag_system_prompt = "Answer the question based on the provided documents."
        self.hyde_system_prompt = "Given the question, write a short hypothetical answer that could be true. Be brief and concise."

    def _evaluate_query(self, query: str, original_hits: List[SearchHit], query_type: str) -> Tuple[List[SearchHit], float]:
        if query_type == 'bm25':
            hits = self.query_service.query_keywords(query, k=self.k)
        else:  # vector
            hits = self.query_service.query_embedding(query, k=self.k)
        sare = calculate_sare([h.score for h in hits], [h.score for h in original_hits], k=self.k)
        return hits, sare
    def _generate_hyde_answer(self, question: str) -> str:
        """Generate a hypothetical answer for HyDE."""
        prompt = f"Question: {question}\n\nHypothetical answer:"
        hyde_answer, _ = self.qgen_llm_client.complete_chat_once(prompt, self.hyde_system_prompt)
        return hyde_answer.strip()

    def _evaluate_query_hyde(self, question: str, hyde_answer: str, original_hits: List[SearchHit], query_type: str) -> Tuple[List[SearchHit], float]:
        """Evaluate query using HyDE method."""
        combined_query = f"Question: {question}\nAnswer: {hyde_answer}"
        if query_type == 'bm25':
            hits = self.query_service.query_keywords(combined_query, k=self.k)
        else:  # vector
            hits = self.query_service.query_embedding(combined_query, k=self.k)
        sare = calculate_sare([h.score for h in hits], [h.score for h in original_hits], k=self.k)
        return hits, sare
        
    def _filter_context_by_relevance(self, question: str, context: List[str]) -> List[str]:
        """Filter context based on relevance scores."""
        if not context:
            return []
            
        # Assess relevance of each context passage
        relevance_scores = self.relevance_assessor.assess_batch(question, context)
        
        # Filter context based on relevance threshold
        filtered_context = []
        for text, score in zip(context, relevance_scores):
            if score >= self.relevance_threshold:
                filtered_context.append(text)
                self.log.info(f"Keeping context with relevance score: {score:.2f}")
            else:
                self.log.info(f"Filtering out context with low relevance score: {score:.2f}")
        
        # If nothing passes the threshold, keep the top scoring passage
        if not filtered_context and context:
            best_idx = relevance_scores.index(max(relevance_scores))
            filtered_context.append(context[best_idx])
            self.log.info(f"No context passed threshold, keeping top passage with score: {relevance_scores[best_idx]:.2f}")
            
        return filtered_context

    def process_question(self, question: str, qid: Optional[str] = None) -> RAGResult:
        start_time = time.time()
        
        # Get baseline results separately for BM25 and vector
        original_bm25_hits = self.query_service.query_keywords(question, k=self.k)
        original_vector_hits = self.query_service.query_embedding(question, k=self.k)
        
        # Generate hypothetical answer for HyDE
        hyde_answer = self._generate_hyde_answer(question)
        
        # Track queries and HyDE results
        tried_queries_bm25 = [question]
        tried_queries_vector = [question]
        best_hits = original_bm25_hits + original_vector_hits
        best_sare_bm25 = 1.0
        best_sare_vector = 1.0
        
        # Try HyDE and query rewrites
        for _ in range(5):
            # BM25 with HyDE
            # Evaluate both BM25 and Vector with HyDE
            for query_type, original_hits, best_sare, tried_queries in [
                ('bm25', original_bm25_hits, best_sare_bm25, tried_queries_bm25),
                ('vector', original_vector_hits, best_sare_vector, tried_queries_vector)
            ]:
                hits, sare = self._evaluate_query_hyde(question, hyde_answer, original_hits, query_type)
                if sare < best_sare:
                    if query_type == 'bm25':
                        best_hits = hits + best_hits[self.k:]
                        best_sare_bm25 = sare
                    else:  # vector
                        best_hits = best_hits[:self.k] + hits
                        best_sare_vector = sare
                    tried_queries.append(f"HyDE:{hyde_answer}")

            # Generate new hypothetical answer if needed
            if best_sare_bm25 >= 0.2 or best_sare_vector >= 0.2:
                hyde_answer = self._generate_hyde_answer(question)
            else:
                break

        # Prepare context
        seen = set()
        context = []
        doc_ids = []
        for hit in best_hits:
            if hit.metadata.text not in seen:
                context.append(hit.metadata.text)
                doc_ids.append(hit.id)
                seen.add(hit.metadata.text)
                
        # Filter context by relevance
        filtered_context = self._filter_context_by_relevance(question, context)
        filtered_doc_ids = []
        
        # Rebuild doc_ids to match filtered context
        for hit in best_hits:
            if hit.metadata.text in filtered_context and hit.id not in filtered_doc_ids:
                filtered_doc_ids.append(hit.id)

        # Generate answer using only relevant context
        prompt = f"Documents:\n\n{chr(10).join(filtered_context)}\n\nQuestion: {question}\n\nAnswer:"
        answer, _ = self.rag_llm_client.complete_chat_once(prompt, self.rag_system_prompt)

        result = RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            context=context,
            doc_ids=doc_ids,
            generated_queries=tried_queries_bm25 + tried_queries_vector,
            total_time_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now(),
            system_name="QPPPostRAG",
            metadata={
                "best_query_bm25": tried_queries_bm25[-1],
                "best_query_vector": tried_queries_vector[-1],
                "best_sare_bm25": best_sare_bm25,
                "best_sare_vector": best_sare_vector,
                "queries_tried_bm25": len(tried_queries_bm25),
                "queries_tried_vector": len(tried_queries_vector),
                "original_context_count": len(context),
                "filtered_context_count": len(filtered_context),
                "relevance_threshold": self.relevance_threshold,
                "filtered_doc_ids": filtered_doc_ids
            }
        )

        self.log.info("Generated answer",
                     answer_length=result.answer_words_count,
                     processing_time_ms=result.total_time_ms,
                     best_sare_bm25=best_sare_bm25,
                     best_sare_vector=best_sare_vector,
                     queries_tried_bm25=len(tried_queries_bm25),
                     queries_tried_vector=len(tried_queries_vector),
                     original_context=len(context),
                     filtered_context=len(filtered_context),
                     qid=qid)

        return result

if __name__ == "__main__":
    rag_system = QPPPostRAG()
    result = rag_system.process_question(
        "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?",
        qid=1
    )

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Context:", result.context)
    print("Document IDs:", result.doc_ids)
    print("Total Time (s):", result.total_time_ms)
    print("Generated Queries:", result.generated_queries)
    print("Original Context Count:", result.metadata["original_context_count"])
    print("Filtered Context Count:", result.metadata["filtered_context_count"])
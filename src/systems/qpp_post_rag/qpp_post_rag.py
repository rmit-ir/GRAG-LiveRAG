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

class QPPPostRAG(RAGSystemInterface):
    log = get_logger("qpp_post_rag")

    def __init__(self, llm_client='ai71', k=10):
        model_id = "tiiuae/falcon3-10b-instruct"
        if llm_client == 'ai71':
            self.rag_llm_client = AI71Client(model_id=model_id)
            self.qgen_llm_client = AI71Client(model_id=model_id)
        elif llm_client == 'ec2_llm':
            self.rag_llm_client = EC2LLMClient(model_id=model_id)
            self.qgen_llm_client = EC2LLMClient(model_id=model_id)

        self.k = k
        self.query_service = QueryService()
        self.qgen_system_prompt = "Generate a better search query variant. Focus on key concepts and ways to express the main idea. Only output the query text."
        self.rag_system_prompt = "Answer the question based on the provided documents."

    def _evaluate_query(self, query: str, original_hits: List[SearchHit]) -> Tuple[List[SearchHit], float]:
        hits = self.query_service.query_embedding(query, k=self.k) + self.query_service.query_keywords(query, k=self.k)
        sare = calculate_sare([h.score for h in hits], [h.score for h in original_hits], k=self.k)
        return hits, sare

    def _generate_improved_query(self, question: str, prev_queries: List[str]) -> str:
        context = "Previous queries:\n" + "\n".join(prev_queries) if prev_queries else ""
        prompt = f"{context}\n\nQuestion: {question}\n\nGenerate better query:"
        query, _ = self.qgen_llm_client.complete_chat_once(prompt, self.qgen_system_prompt)
        return query.strip()

    def process_question(self, question: str, qid: Optional[str] = None) -> RAGResult:
        start_time = time.time()
        
        # Get baseline results
        original_hits = self.query_service.query_embedding(question, k=self.k) + self.query_service.query_keywords(question, k=self.k)
        
        # Iteratively improve query
        current_query = question
        tried_queries = [current_query]
        best_hits = original_hits
        best_sare = 1.0
        
        # Try up to 5 query rewrites
        for _ in range(4):
            new_query = self._generate_improved_query(question, tried_queries)
            if new_query in tried_queries:
                continue
                
            tried_queries.append(new_query)
            hits, sare = self._evaluate_query(new_query, original_hits)
            
            if sare < best_sare:
                best_hits = hits
                best_sare = sare
                current_query = new_query
            
            if best_sare < 0.2:
                break

        # Prepare final context
        seen = set()
        context = []
        doc_ids = []
        for hit in best_hits:
            if hit.metadata.text not in seen:
                context.append(hit.metadata.text)
                doc_ids.append(hit.id)
                seen.add(hit.metadata.text)

        # Generate answer
        prompt = f"Documents:\n\n{chr(10).join(context)}\n\nQuestion: {question}\n\nAnswer:"
        answer, _ = self.rag_llm_client.complete_chat_once(prompt, self.rag_system_prompt)

        result = RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            context=context,
            doc_ids=doc_ids,
            generated_queries=tried_queries,
            total_time_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now(),
            system_name="QPPPostRAG",
            metadata={
                "best_query": current_query,
                "best_sare": best_sare,
                "queries_tried": len(tried_queries)
            }
        )

        self.log.info("Generated answer",
                     answer_length=result.answer_words_count,
                     processing_time_ms=result.total_time_ms,
                     best_sare=best_sare,
                     queries_tried=len(tried_queries),
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
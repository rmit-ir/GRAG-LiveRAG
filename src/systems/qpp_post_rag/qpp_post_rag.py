"""
QPP Post-retrieval RAG system that uses correlation coefficients and sARE for query selection.
"""
from typing import List, Dict, Tuple
import time
from systems.vanilla_rag.vanilla_rag import VanillaRAG
from systems.rag_result import RAGResult
from services.indicies import SearchHit
from .qpp_metrics import (
    calculate_pearson, calculate_spearman,
    calculate_kendall, calculate_sare,
    calculate_qpp_scores
)
from .relevance_assessor import RelevanceAssessor

class QPPPostRAG(VanillaRAG):
    def __init__(self, llm_client='ai71', module_query_gen='with_number'):
        super().__init__(llm_client, module_query_gen)
        self.relevance_assessor = RelevanceAssessor(llm_client)

    def _calculate_query_metrics(self, query: str, documents: List[SearchHit]) -> Tuple[Dict[str, float], List[float]]:
        """Calculate QPP metrics for a query and its retrieved documents."""
        passages = [doc.metadata.text for doc in documents]
        relevance_scores, oracle_scores = calculate_qpp_scores(query, passages, self.relevance_assessor)
        
        metrics = {
            'pearson': calculate_pearson(relevance_scores, oracle_scores),
            'spearman': calculate_spearman(relevance_scores, oracle_scores),
            'kendall': calculate_kendall(relevance_scores, oracle_scores),
            'sare': calculate_sare(relevance_scores, oracle_scores)
        }
        
        return metrics, relevance_scores

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """Process question with QPP-based post-retrieval filtering."""
        start_time = time.time()
        
        # Get query variants like vanilla RAG
        queries = self._create_query_variants(question)

        # Retrieve initial documents
        documents: List[SearchHit] = []
        doc_ids = set()
        for query in queries:
            embed_results = self.query_service.query_embedding(query, k=3)
            keyword_results = self.query_service.query_keywords(query, k=3)
            results = embed_results + keyword_results
            for doc in results:
                if doc.id not in doc_ids:
                    documents.append(doc)
                    doc_ids.add(doc.id)

        # Filter documents using QPP
        filtered_documents = []
        filtered_doc_ids = set()
        for doc in documents:
            if self.relevance_assessor.assess_relevance(question, doc.metadata.text):
                filtered_documents.append(doc)
                filtered_doc_ids.add(doc.id)

        # If no documents pass QPP, fall back to original documents
        if not filtered_documents:
            filtered_documents = documents
            filtered_doc_ids = doc_ids

        # Create context from filtered documents
        context = "Documents: \n\n"
        context += "\n\n".join([doc.metadata.text for doc in filtered_documents])

        # Generate answer using filtered context
        prompt = context + "\n\nQuestion: " + question + "\n\nAnswer: "
        answer, _ = self.rag_llm_client.complete_chat_once(prompt, self.rag_system_prompt)

        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            context=[doc.metadata.text for doc in filtered_documents],
            doc_ids=list(filtered_doc_ids),
            generated_queries=queries,
            total_time_ms=(time.time() - start_time) * 1000,
            system_name="QPPPostRAG",
        )

if __name__ == "__main__":
    # Test the QPPPostRAG system
    rag_system = QPPPostRAG()
    result = rag_system.process_question(
        "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?",
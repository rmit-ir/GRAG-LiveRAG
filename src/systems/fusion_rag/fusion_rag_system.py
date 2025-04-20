import time
from typing import List
from datetime import datetime

from utils.logging_utils import get_logger
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.fusion_rag.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class FusionRAGSystem(RAGSystemInterface):
    """
    Fusion RAG system implementation.
    """
    
    log = get_logger("fusion_rag_system")
    
    def __init__(self):
        self.query_service = QueryService()
        
        self.llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=SYSTEM_PROMPT
        )
        
        self.max_documents = 10
        self.log.info("FusionRAGSystem initialized", 
                     llm_model="tiiuae/falcon3-10b-instruct")
    
    def generate_queries(self, question: str) -> List[str]:
        # Reformulate question to generate search queries
        return [question]
    
    def process_question(self, question: str, qid: str = None) -> RAGResult:
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)
        
        # Generate search queries
        queries = self.generate_queries(question)
        self.log.debug("Generated queries", queries=queries, count=len(queries))
        
        # Search for documents using each query and collect all results
        all_hits = []
        for query in queries:
            # Use fusion search to combine embedding and keyword search
            hits = self.query_service.query_fusion(query, k=self.max_documents)
            all_hits.extend(hits)
            self.log.debug("Retrieved documents for query", 
                          query=query, 
                          hits_count=len(hits))
        
        # Take the first max_documents unique documents
        unique_docs = {}
        for hit in all_hits:
            if hit.id not in unique_docs and len(unique_docs) < self.max_documents:
                unique_docs[hit.id] = hit
        
        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in unique_docs.values()]
        doc_ids = [hit.id for hit in unique_docs.values()]
        
        self.log.debug("Selected documents for context", 
                      doc_count=len(doc_contents),
                      doc_ids=doc_ids)
        
        # Create context for the LLM
        context = "\n\n".join(doc_contents)
        
        # Generate prompt for the LLM
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Generate answer using the LLM
        _, answer = self.llm_client.query(prompt)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        result = RAGResult(
            query=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=queries if queries != [question] else None,
            rewritten_docs=None,
            qid=qid
        )
        
        self.log.info("Generated answer", 
                     answer_length=result.answer_words_count,
                     processing_time_ms=total_time_ms,
                     qid=qid)
        
        return result

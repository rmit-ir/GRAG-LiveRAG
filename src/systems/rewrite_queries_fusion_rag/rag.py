"""
uv run scripts/run.py --system systems.rewrite_queries_fusion_rag.rag.FusionRAGSystem --help
"""
import time
from typing import List
from datetime import datetime

from utils.logging_utils import get_logger
from utils.fusion_utils import apply_fusion_to_hits
from utils.query_utils import generate_query_id
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from .prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE, QUERY_GENERATION_PROMPT


class FusionRAGSystem(RAGSystemInterface):
    """
    This RAG system rewrites the question to generate search queries and uses a fusion search strategy
    """
    
    log = get_logger("fusion_rag_system")
    
    def __init__(self, max_documents: int = 10, max_queries: int = 3):
        """
        Initialize the FusionRAGSystem.
        
        Args:
            max_documents: Maximum number of documents to retrieve per query
            max_queries: Maximum number of search queries to generate (including the original)
        """
        self.query_service = QueryService()
        
        self.llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=SYSTEM_PROMPT
        )
        
        self.max_documents = max_documents
        self.max_queries = max_queries
        self.log.info("FusionRAGSystem initialized", 
                     llm_model="tiiuae/falcon3-10b-instruct",
                     max_documents=max_documents,
                     max_queries=max_queries)
    
    def generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple search queries from the original question using the LLM.
        
        Args:
            question: The original user question
            
        Returns:
            A list of search queries derived from the original question
        """
        # Always include the original question as one of the queries
        queries = [question]
        
        try:
            # Generate additional queries using the LLM
            # Adjust the prompt to specify how many queries we want (max_queries - 1 since we already include the original)
            prompt = QUERY_GENERATION_PROMPT.format(question=question)
            if self.max_queries > 1:
                prompt = f"Generate {self.max_queries - 1} alternative search queries for the following question.\n{prompt}"
            _, response = self.llm_client.query(prompt)
            
            # Parse the response to extract the generated queries
            # Expecting one query per line
            generated_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Add unique queries to the list (avoid duplicates)
            for query in generated_queries:
                if query not in queries and len(queries) < self.max_queries:
                    queries.append(query)
                    
            self.log.debug("Generated search queries", 
                          original_question=question,
                          generated_queries=queries)
        except Exception as e:
            self.log.error("Failed to generate search queries", 
                          error=str(e),
                          question=question)
        
        return queries
    
    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question using the fusion RAG approach:

        1. Generate multiple search queries from the original question
        2. Use fusion search for each query to retrieve relevant documents
        3. Combine and deduplicate the results
        4. Generate an answer using the retrieved documents as context
        
        Args:
            question: The user's question
            qid: Optional query ID
            
        Returns:
            A RAGResult containing the answer and metadata
        """
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)
        
        # Generate search queries
        queries = self.generate_queries(question)
        self.log.debug("Generated queries", queries=queries, count=len(queries))
        
        # Search for documents using each query and collect results per query
        hits_per_query = []
        for query in queries:
            # Use fusion search to combine embedding and keyword search
            hits = self.query_service.query_fusion(query, k=self.max_documents)
            hits_per_query.append(hits)
            self.log.debug("Retrieved documents for query", 
                          query=query, 
                          hits_count=len(hits))
        
        # Apply fusion to get the top documents
        qid = qid or generate_query_id(question)
        fused_docs = apply_fusion_to_hits(hits_per_query, self.max_documents, query_id=qid)
        
        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in fused_docs.values()]
        doc_ids = [hit.id for hit in fused_docs.values()]
        
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
            question=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=queries if len(queries) > 1 else None,
            rewritten_docs=None,
            qid=qid,
            system_name="FusionRAGSystem"
        )
        
        self.log.info("Generated answer", 
                     answer_length=result.answer_words_count,
                     processing_time_ms=total_time_ms,
                     qid=qid)
        
        return result

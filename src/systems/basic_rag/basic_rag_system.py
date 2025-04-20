"""
Basic RAG system implementation.

This module implements a basic RAG (Retrieval-Augmented Generation) system that:
1. Takes a question
2. Generates search queries
3. Searches each query to get documents
4. Performs fusion of results
5. Generates an answer using the retrieved documents
"""
import time
from typing import List, Optional
from datetime import datetime

from utils.logging_utils import get_logger
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.basic_rag.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class BasicRAGSystem(RAGSystemInterface):
    """
    Basic RAG system implementation.
    
    This system takes a question, generates search queries, retrieves documents,
    performs fusion, and generates an answer using the retrieved documents.
    """
    
    log = get_logger("basic_rag_system")
    
    def __init__(
        self,
        embedding_model_name: str = "intfloat/e5-base-v2",
        pinecone_index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        pinecone_namespace: str = "default",
        opensearch_index_name: str = "fineweb10bt-512-0w-e5-base-v2",
        region_name: Optional[str] = None,
        max_documents: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the BasicRAGSystem.
        
        Args:
            embedding_model_name: The embedding model name to use for query embedding
            pinecone_index_name: The Pinecone index name to use
            pinecone_namespace: The Pinecone namespace to use
            opensearch_index_name: The OpenSearch index name to use
            region_name: The AWS region name to use
            max_documents: Maximum number of documents to retrieve
            temperature: The temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
        """
        self.query_service = QueryService(
            pinecone_embedding_model_name=embedding_model_name,
            pinecone_index_name=pinecone_index_name,
            pinecone_namespace=pinecone_namespace,
            opensearch_index_name=opensearch_index_name,
            region_name=region_name
        )
        
        self.llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.max_documents = max_documents
        self.log.info("BasicRAGSystem initialized", 
                     llm_model="tiiuae/falcon3-10b-instruct", 
                     embedding_model=embedding_model_name,
                     max_documents=max_documents)
    
    def generate_queries(self, question: str) -> List[str]:
        """
        Generate search queries from the question.
        
        Args:
            question: The user's question
            
        Returns:
            A list of search queries
        """
        # For a basic implementation, we'll just use the original question
        # In a more advanced system, we could use an LLM to generate multiple queries
        self.log.debug("Generating queries", question=question)
        return [question]
    
    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question and generate an answer using RAG.
        
        Args:
            question: The user's question
            qid: Optional query ID, will be populated in RAGResult
            
        Returns:
            A RAGResult containing the answer and metadata
        """
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
        
        # Calculate metrics
        query_words_count = len(question.split())
        answer_words_count = len(answer.split())
        total_time_ms = (time.time() - start_time) * 1000
        
        self.log.info("Generated answer", 
                     answer_length=answer_words_count,
                     processing_time_ms=total_time_ms,
                     qid=qid)
        
        # Create and return the result
        result = RAGResult(
            query=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            query_words_count=query_words_count,
            answer_words_count=answer_words_count,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=queries if queries != [question] else None,
            rewritten_docs=None,
            qid=qid
        )
        
        return result


# Main entry point for testing the basic RAG system
if __name__ == "__main__":
    import json
    import os
    from datetime import datetime
    from dotenv import load_dotenv
    from utils.path_utils import get_data_dir

    # Load environment variables
    load_dotenv()
    
    # Define test question
    question = "What is retrieval-augmented generation?"
    
    # Initialize the RAG system
    log = get_logger("basic_rag_main")
    log.info("Initializing RAG system")
    rag_system = BasicRAGSystem(
        max_documents=10,
        temperature=0.7,
        max_tokens=1024
    )
    
    # Process the question
    qid = "test-1"
    log.info("Processing question", question=question, qid=qid)
    result = rag_system.process_question(question, qid=qid)
    
    # Print the result
    print("\n" + "="*80)
    print(f"Query ID: {result.qid}")
    print(f"Question: {result.query}")
    print(f"Answer: {result.answer}")
    print("-"*80)
    print(f"Query words: {result.query_words_count}")
    print(f"Answer words: {result.answer_words_count}")
    print(f"Processing time: {result.total_time_ms:.2f} ms")
    print(f"Documents used: {len(result.doc_ids)}")
    print(f"Document IDs: {', '.join(result.doc_ids[:3])}...")
    print("="*80 + "\n")
    
    # Save the result
    output_dir = os.path.join(get_data_dir(), "rag_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_result_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save the result as JSON
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    log.info("Result saved", filepath=filepath, qid=result.qid)

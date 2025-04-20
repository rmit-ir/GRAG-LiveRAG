import time
from typing import List
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
    """
    
    log = get_logger("basic_rag_system")
    
    def __init__(self):
        self.query_service = QueryService()
        
        self.llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=SYSTEM_PROMPT
        )
        
        self.max_documents = 10
        self.log.info("BasicRAGSystem initialized", 
                     llm_model="tiiuae/falcon3-10b-instruct")
    
    def generate_queries(self, question: str) -> List[str]:
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
            hits = self.query_service.query_keywords(query, k=self.max_documents)
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
    rag_system = BasicRAGSystem()
    
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

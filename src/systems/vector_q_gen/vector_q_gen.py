"""
Vector Q-Gen RAG System that decomposes queries into components and generates queries for each component.
uv run scripts/run.py --system systems.vector_q_gen.vector_q_gen.VectorQGen --help
"""
import time
from typing import List, Dict, Any
from datetime import datetime

from utils.logging_utils import get_logger
from utils.fusion_utils import rrf_fusion
from utils.query_utils import generate_query_id
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from services.llms.bedrock_client import BedrockClient
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.vector_q_gen.prompts import (
    SYSTEM_PROMPT, 
    ANSWER_PROMPT_TEMPLATE
)
from systems.vector_q_gen.query_processor import QueryProcessor


class VectorQGen(RAGSystemInterface):
    """
    This RAG system decomposes the question into components, generates queries for each component,
    performs fusion within each component, and takes top documents from each component for final generation.
    """

    log = get_logger("vector_q_gen_system")

    def __init__(self,
                 max_documents: int = 10,
                 max_component_documents: int = 5,
                 max_components: int = 3,
                 max_queries_per_component: int = 3,
                 q_gen_model_id: str = "tiiuae/falcon3-10b-instruct",
                 rag_model_id: str = "tiiuae/falcon3-10b-instruct",
                 q_gen_llm_client: str = "ai71_client",
                 rag_llm_client: str = "ai71_client"):
        """
        Initialize the VectorQGen.

        Args:
            max_documents: Maximum number of documents to stuff to RAG generation (total across all components)
            max_component_documents: Maximum number of documents to select from each component
            max_components: Maximum number of components to decompose the question into
            max_queries_per_component: Maximum number of queries to generate for each component
            q_gen_model_id: Model ID for query generation LLM
            rag_model_id: Model ID for RAG generation LLM 
            q_gen_llm_client: Client type for query generation LLM, ai71_client, general_openai_client, bedrock_client
            rag_llm_client: Client type for RAG generation LLM, ai71_client, general_openai_client, bedrock_client
        """
        self.query_service = QueryService()
        self.max_documents = max_documents
        self.max_component_documents = max_component_documents
        
        # Initialize the query processor
        self.query_processor = QueryProcessor(
            max_components=max_components,
            max_queries_per_component=max_queries_per_component,
            q_gen_model_id=q_gen_model_id,
            q_gen_llm_client=q_gen_llm_client
        )
        
        # Initialize the RAG answer generation client
        self.load_rag_client(rag_llm_client, rag_model_id)

        self.log.info("VectorQGen initialized",
                      q_gen_model_id=q_gen_model_id,
                      rag_model_id=rag_model_id,
                      q_gen_llm_client=q_gen_llm_client,
                      rag_llm_client=rag_llm_client,
                      max_documents=max_documents,
                      max_component_documents=max_component_documents,
                      max_components=max_components,
                      max_queries_per_component=max_queries_per_component)

    def load_rag_client(self, rag_llm_client: str, rag_model_id: str):
        """
        Load LLM client for RAG answer generation.
        
        Args:
            rag_llm_client: Client type for RAG answer generation ("bedrock_client", "general_openai_client", or "ai71_client")
            rag_model_id: Model ID for RAG answer generation
        """
        # Initialize RAG answer generation client
        # If model contains "sonnet" or "claude", use BedrockClient regardless of rag_llm_client setting
        if 'sonnet' in rag_model_id.lower() or 'claude' in rag_model_id.lower():
            self.llm_client = BedrockClient(
                model_id=rag_model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0
            )
        elif rag_llm_client == "general_openai_client":
            self.llm_client = GeneralOpenAIClient(
                model_id=rag_model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0
            )
        elif rag_llm_client == "bedrock_client":
            self.llm_client = BedrockClient(
                model_id=rag_model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0
            )
        else:  # Default to AI71Client
            self.llm_client = AI71Client(
                model_id=rag_model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0
            )

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question using the Vector Q-Gen approach:
        1. Decompose the question into components
        2. Generate queries for each component
        3. Retrieve documents for each query
        4. Perform fusion within each component
        5. Select top documents from each component
        6. Generate answer using the selected documents
        
        Args:
            question: The user's question
            qid: Optional query ID
            
        Returns:
            A RAGResult containing the answer and metadata
        """
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)
        
        # Generate a query ID if not provided
        qid = qid or generate_query_id(question)
        
        # Step 1: Decompose the question into components
        components = self.query_processor.decompose_question(question)
        
        # Limit the number of components if needed
        if len(components) > self.query_processor.max_components:
            self.log.info("Limiting number of components",
                         original_count=len(components),
                         max_components=self.query_processor.max_components)
            components = components[:self.query_processor.max_components]
        
        # Store component data
        component_data = []
        all_selected_docs = []
        
        # Step 2-4: For each component, generate queries, retrieve documents, and perform fusion
        for i, component in enumerate(components):
            component_info = {
                "component": component,
                "queries": [],
                "doc_count": 0
            }
            
            # Generate queries for this component
            queries = self.query_processor.generate_queries_for_component(component, question)
            component_info["queries"] = queries
            
            # Retrieve documents for each query
            hits_per_query = []
            for query in queries:
                hits = self.query_service.query_embedding(query, k=self.max_component_documents * 2)
                if hits:
                    hits_per_query.append(hits)
                    self.log.debug("Retrieved documents for query",
                                  component=component,
                                  query=query,
                                  hits_count=len(hits))
                else:
                    self.log.warning("No hits for query",
                                    component=component,
                                    query=query)
            
            # Perform fusion within this component to get the top documents
            if hits_per_query:
                component_id = f"{qid}_component_{i}"
                fused_docs = rrf_fusion(hits_per_query, self.max_component_documents, query_id=component_id)
                component_info["doc_count"] = len(fused_docs)
                
                # Add the selected documents to our collection
                all_selected_docs.extend(fused_docs)
                
                self.log.debug("Selected documents for component",
                              component=component,
                              doc_count=len(fused_docs))
            else:
                self.log.warning("No documents retrieved for component",
                               component=component)
            
            component_data.append(component_info)
        
        # Step 5: If we have more documents than our limit, select the top ones
        # We take an equal number from each component if possible
        if len(all_selected_docs) > self.max_documents:
            self.log.info("Limiting total number of documents",
                         original_count=len(all_selected_docs),
                         max_documents=self.max_documents)
            
            # Sort all documents by score
            all_selected_docs.sort(key=lambda x: x.score, reverse=True)
            all_selected_docs = all_selected_docs[:self.max_documents]
        
        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in all_selected_docs]
        doc_ids = [hit.id for hit in all_selected_docs]
        
        self.log.debug("Final selected documents for context",
                      doc_count=len(doc_contents),
                      doc_ids=doc_ids)
        
        # Step 6: Create context for the LLM
        context = "\n\n".join(doc_contents)
        
        # Generate prompt for the LLM
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context, question=question)
        
        # Generate answer using the LLM
        _, answer = self.llm_client.query(prompt)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Create metadata for the result
        metadata = {
            "components": [
                {
                    "component": comp["component"],
                    "queries": comp["queries"],
                    "doc_count": comp["doc_count"]
                } for comp in component_data
            ],
            "total_components": len(components),
            "total_queries": sum(len(comp["queries"]) for comp in component_data),
            "total_docs": len(doc_contents),
            "final_prompt": prompt
        }
        
        # Collect all generated queries
        all_queries = []
        for comp in component_data:
            all_queries.extend(comp["queries"])
        
        result = RAGResult(
            question=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=all_queries if len(all_queries) > 1 else None,
            rewritten_docs=None,
            qid=qid,
            system_name="VectorQGen",
            metadata=metadata
        )
        
        self.log.info("Generated answer",
                     answer_length=result.answer_words_count,
                     processing_time_ms=total_time_ms,
                     component_count=len(components),
                     query_count=len(all_queries),
                     doc_count=len(doc_contents),
                     qid=qid)
        
        return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    from systems.rag_system_interface import test_rag_system
    
    load_dotenv()
    
    # Test the RAG system
    result = test_rag_system(VectorQGen(
        rag_llm_client="ai71_client"
    ), "form u4 arbitration provisions compared transportation worker arbitration agreements legal status differences")
    
    print(f"Components: {len(result.metadata['components'])}")
    for i, comp in enumerate(result.metadata['components']):
        print(f"Component {i+1}: {comp['component']}")
        print(f"  Queries: {comp['queries']}")
        print(f"  Documents: {comp['doc_count']}")

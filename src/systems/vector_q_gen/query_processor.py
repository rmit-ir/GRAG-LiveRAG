"""
Query processing utilities for the Vector Q-Gen RAG System.
"""
import re
import time
from typing import List

from utils.logging_utils import get_logger
from services.llms.ai71_client import AI71Client
from services.llms.bedrock_client import BedrockClient
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.vector_q_gen.prompts import (
    QUERY_DECOMPOSITION_PROMPT,
    COMPONENT_QUERY_GENERATION_PROMPT
)


class QueryProcessor:
    """
    Handles query decomposition and query generation for the Vector Q-Gen RAG System.
    """
    
    log = get_logger("vector_q_gen_query_processor")
    components_tag_pattern = re.compile(r'<components>(.*?)</components>', re.DOTALL)
    queries_tag_pattern = re.compile(r'<queries>(.*?)</queries>', re.DOTALL)
    
    def __init__(self,
                 max_components: int = 3,
                 max_queries_per_component: int = 3,
                 q_gen_model_id: str = "tiiuae/falcon3-10b-instruct",
                 q_gen_llm_client: str = "ai71_client"):
        """
        Initialize the QueryProcessor.
        
        Args:
            max_components: Maximum number of components to decompose the question into
            max_queries_per_component: Maximum number of queries to generate for each component
            q_gen_model_id: Model ID for query generation LLM
            q_gen_llm_client: Client type for query generation LLM
        """
        self.max_components = max_components
        self.max_queries_per_component = max_queries_per_component
        
        self.load_llm_clients(q_gen_llm_client, q_gen_model_id)
        
        self.log.info("QueryProcessor initialized",
                     q_gen_model_id=q_gen_model_id,
                     q_gen_llm_client=q_gen_llm_client,
                     max_components=max_components,
                     max_queries_per_component=max_queries_per_component)
    
    def load_llm_clients(self, q_gen_llm_client: str, q_gen_model_id: str):
        """
        Load LLM clients for query generation.
        
        Args:
            q_gen_llm_client: Client type for query generation ("bedrock_client", "general_openai_client", or "ai71_client")
            q_gen_model_id: Model ID for query generation
        """
        # Prepare query decomposition prompt
        self.decomposition_prompt = QUERY_DECOMPOSITION_PROMPT.format(max_components=self.max_components)
        
        # Initialize query generation clients
        # If model contains "sonnet" or "claude", use BedrockClient regardless of q_gen_llm_client setting
        if 'sonnet' in q_gen_model_id.lower() or 'claude' in q_gen_model_id.lower():
            self.query_decomposer = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
            self.query_generator = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        elif q_gen_llm_client == "general_openai_client":
            self.query_decomposer = GeneralOpenAIClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
            self.query_generator = GeneralOpenAIClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        elif q_gen_llm_client == "bedrock_client":
            self.query_decomposer = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
            self.query_generator = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        else:  # Default to AI71Client
            self.query_decomposer = AI71Client(
                model_id=q_gen_model_id,
                temperature=0.5
            )
            self.query_generator = AI71Client(
                model_id=q_gen_model_id,
                temperature=0.5
            )
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing surrounding quotes.
        
        Args:
            text: The text string to sanitize
            
        Returns:
            Sanitized text string
        """
        # Remove surrounding double quotes if present
        text = text.strip()
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        return text
    
    def _extract_components(self, response: str) -> List[str]:
        """
        Extract components from LLM response that contains HTML tags.
        Only extracts lines within <components> and </components> tags.
        
        Args:
            response: LLM response text containing components in HTML tags
            
        Returns:
            List of extracted components
        """
        # Extract content between <components> tags
        components_match = re.search(r'<components>(.*?)</components>', response, re.DOTALL)
        if not components_match:
            self.log.warning("No components found in HTML tags")
            return []
        
        components_content = components_match.group(1)
        
        # Extract numbered components (1. Component text)
        components = []
        for line in components_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try to match numbered format (e.g., "1. Component text")
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                component = self._sanitize_text(match.group(1).strip())
                if component:
                    components.append(component)
            elif line:  # If not numbered but not empty, add it anyway
                components.append(self._sanitize_text(line))
        
        return components
    
    def _extract_queries(self, response: str) -> List[str]:
        """
        Extract queries from LLM response that contains HTML tags.
        Only extracts lines within <queries> and </queries> tags.
        
        Args:
            response: LLM response text containing queries in HTML tags
            
        Returns:
            List of extracted queries
        """
        # Extract content between <queries> tags
        queries_match = re.search(r'<queries>(.*?)</queries>', response, re.DOTALL)
        if not queries_match:
            self.log.warning("No queries found in HTML tags")
            return []
        
        queries_content = queries_match.group(1)
        
        # Extract numbered queries (1. Query text)
        queries = []
        for line in queries_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try to match numbered format (e.g., "1. Query text")
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                query = self._sanitize_text(match.group(1).strip())
                if query:
                    queries.append(query)
            elif line:  # If not numbered but not empty, add it anyway
                queries.append(self._sanitize_text(line))
        
        return queries
    
    def decompose_question(self, question: str, max_retries: int = 2) -> List[str]:
        """
        Decompose the question into components.
        
        Args:
            question: The user's question
            max_retries: Maximum number of retries if decomposition fails
            
        Returns:
            List of question components
        """
        for attempt in range(max_retries + 1):
            try:
                # Pass the question to the decomposer
                prompt = f"User question: {question}"
                response, _ = self.query_decomposer.complete_chat_once(prompt, self.decomposition_prompt)
                
                # Extract components from the response
                components = self._extract_components(response)
                
                if components:
                    self.log.debug("Decomposed question into components",
                                  attempt=attempt,
                                  component_count=len(components),
                                  components=components)
                    return components
                else:
                    self.log.warning("No components extracted, retrying",
                                    attempt=attempt)
            except Exception as e:
                self.log.error("Failed to decompose question",
                              error=str(e),
                              attempt=attempt)
            
            # If we reach here, we need to retry
            if attempt < max_retries:
                self.log.info("Retrying question decomposition",
                             attempt=attempt+1,
                             max_retries=max_retries)
                time.sleep(1)  # Small delay before retry
        
        # If all retries failed, return the original question as a single component
        self.log.error("All question decomposition attempts failed, using original question")
        return [question]
    
    def generate_queries_for_component(self, component: str, original_question: str, max_retries: int = 2) -> List[str]:
        """
        Generate queries for a specific component.
        
        Args:
            component: The component to generate queries for
            original_question: The original user question
            max_retries: Maximum number of retries if query generation fails
            
        Returns:
            List of generated queries
        """
        # Set the system message for this component
        component_prompt = COMPONENT_QUERY_GENERATION_PROMPT.format(
            original_question=original_question,
            component=component,
            max_queries=self.max_queries_per_component
        )
        
        for attempt in range(max_retries + 1):
            try:
                # Pass the component as the query
                prompt = f"Component: {component}"
                response, _ = self.query_generator.complete_chat_once(prompt, component_prompt)
                
                # Extract queries from the response
                queries = self._extract_queries(response)
                
                if queries:
                    self.log.debug("Generated queries for component",
                                  component=component,
                                  attempt=attempt,
                                  query_count=len(queries),
                                  queries=queries)
                    return queries
                else:
                    self.log.warning("No queries extracted for component, retrying",
                                    component=component,
                                    attempt=attempt)
            except Exception as e:
                self.log.error("Failed to generate queries for component",
                              component=component,
                              error=str(e),
                              attempt=attempt)
            
            # If we reach here, we need to retry
            if attempt < max_retries:
                self.log.info("Retrying query generation for component",
                             component=component,
                             attempt=attempt+1,
                             max_retries=max_retries)
                time.sleep(1)  # Small delay before retry
        
        # If all retries failed, return the component as a single query
        self.log.error("All query generation attempts failed for component, using component as query",
                      component=component)
        return [component]

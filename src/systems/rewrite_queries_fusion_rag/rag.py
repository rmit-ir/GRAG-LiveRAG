"""
uv run scripts/run.py --system systems.rewrite_queries_fusion_rag.rag.FusionRAGSystem --help
"""
import re
import time
from typing import List
from datetime import datetime

from utils.logging_utils import get_logger
from utils.fusion_utils import rrf_fusion
from utils.query_utils import generate_query_id
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.rewrite_queries_fusion_rag.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE, SPARSE_QUERY_GENERATION_PROMPT, DENSE_QUERY_GENERATION_PROMPT


class FusionRAGSystem(RAGSystemInterface):
    """
    This RAG system rewrites the question to generate search queries and uses a fusion search strategy
    """

    log = get_logger("fusion_rag_system")
    query_tag_pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
    queries_tag_pattern = re.compile(r'<queries>(.*?)</queries>', re.DOTALL)

    def __init__(self, max_documents: int = 10, max_queries: int = 3):
        """
        Initialize the FusionRAGSystem.

        Args:
            max_documents: Maximum number of documents to retrieve per query
            max_queries: Maximum number of search queries to generate (including the original)
        """
        self.query_service = QueryService()

        # LLM client for answer generation
        self.llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=SYSTEM_PROMPT
        )

        # LLM client for sparse query generation (keyword-based search)
        sparse_prompt = SPARSE_QUERY_GENERATION_PROMPT.format(max_queries=max_queries)
        self.sparse_query_generator = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=sparse_prompt,
            temperature=0.5  # Set temperature for diversity in retry scenarios
        )

        # LLM client for dense query generation (semantic search)
        dense_prompt = DENSE_QUERY_GENERATION_PROMPT.format(max_queries=max_queries)
        self.dense_query_generator = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
            system_message=dense_prompt,
            temperature=0.5  # Set temperature for diversity in retry scenarios
        )

        self.max_documents = max_documents
        self.max_queries = max_queries
        self.log.info("FusionRAGSystem initialized",
                      llm_model="tiiuae/falcon3-10b-instruct",
                      max_documents=max_documents,
                      max_queries=max_queries)

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize a query by removing surrounding double quotes and <query> tags.
        Example input queries:
        
        > 1. "<queries>How does the artwork 'For Proctor Silex' manipulate perspective to engage viewers as they move closer?"</queries>
        
        Args:
            query: The query string to sanitize
            
        Returns:
            Sanitized query string
        """
        # Remove surrounding double quotes if present
        query = query.strip()
        if (query.startswith('"') and query.endswith('"')) or \
           (query.startswith("'") and query.endswith("'")):
            query = query[1:-1].strip()
            
        # Remove <query></query> tags if present
        match = self.query_tag_pattern.match(query)
        if match:
            query = match.group(1).strip()
        match = self.queries_tag_pattern.match(query)
        if match:
            query = match.group(1).strip()
            
        return query
        
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
        queries_match = re.search(
            r'<queries>(.*?)</queries>', response, re.DOTALL)
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
                query = self._sanitize_query(match.group(1).strip())
                if query:
                    queries.append(query)
            elif line:  # If not numbered but not empty, add it anyway
                queries.append(self._sanitize_query(line))

        return queries

    def _generate_queries(self, question: str, engine_type: str, max_retries: int = 2) -> List[str]:
        """
        Generate queries optimized for a specific search engine type.

        Args:
            question: The user's question
            engine_type: Type of search engine ('sparse' or 'dense')
            max_retries: Maximum number of retries if query generation fails

        Returns:
            List of generated queries
        """
        client = self.sparse_query_generator if engine_type == 'sparse' else self.dense_query_generator

        for attempt in range(max_retries + 1):
            try:
                # Just pass the question as the query
                prompt = f"User question: {question}"
                _, response = client.query(prompt)

                # Extract queries from the response
                queries = self._extract_queries(response)

                if queries:
                    self.log.debug(f"Generated {engine_type} queries",
                                   attempt=attempt,
                                   query_count=len(queries),
                                   queries=queries)
                    return queries
                else:
                    self.log.warning(f"No {engine_type} queries extracted, retrying",
                                     attempt=attempt)
            except Exception as e:
                self.log.error(f"Failed to generate {engine_type} queries",
                               error=str(e),
                               attempt=attempt)

            # If we reach here, we need to retry
            if attempt < max_retries:
                self.log.info(f"Retrying {engine_type} query generation",
                              attempt=attempt+1,
                              max_retries=max_retries)
                time.sleep(1)  # Small delay before retry

        # If all retries failed, return empty list
        self.log.error(f"All {engine_type} query generation attempts failed")
        return []

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question using the fusion RAG approach:

        1. Generate multiple search queries from the original question using different generators
           - Sparse queries for keyword-based search
           - Dense queries for embedding-based search
        2. Use appropriate search method for each query type
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

        # Always include the original question
        original_query = question

        # Generate sparse queries for keyword-based search
        sparse_queries = self._generate_queries(question, 'sparse')
        self.log.debug("Generated sparse queries",
                       queries=sparse_queries, count=len(sparse_queries))

        # Generate dense queries for embedding-based search
        dense_queries = self._generate_queries(question, 'dense')
        self.log.debug("Generated dense queries",
                       queries=dense_queries, count=len(dense_queries))

        # Search for documents using each query and collect results per query
        hits_per_query = []

        # Use the original query with fusion search
        hits = self.query_service.query_fusion(
            original_query, k=self.max_documents)
        hits_per_query.append(hits)
        self.log.debug("Retrieved documents for original query",
                       query=original_query,
                       hits_count=len(hits))

        # Use sparse queries with keyword search
        for query in sparse_queries:
            hits = self.query_service.query_keywords(
                query, k=self.max_documents)
            hits_per_query.append(hits)
            self.log.debug("Retrieved documents for sparse query",
                           query=query,
                           hits_count=len(hits))

        # Use dense queries with embedding search
        for query in dense_queries:
            hits = self.query_service.query_embedding(
                query, k=self.max_documents)
            hits_per_query.append(hits)
            self.log.debug("Retrieved documents for dense query",
                           query=query,
                           hits_count=len(hits))

        # Apply fusion to get the top documents
        qid = qid or generate_query_id(question)
        fused_docs = rrf_fusion(
            hits_per_query, self.max_documents, query_id=qid)

        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in fused_docs]
        doc_ids = [hit.id for hit in fused_docs]

        self.log.debug("Selected documents for context",
                       doc_count=len(doc_contents),
                       doc_ids=doc_ids)

        # Create context for the LLM
        context = "\n\n".join(doc_contents)

        # Generate prompt for the LLM
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context, question=question)

        # Generate answer using the LLM
        _, answer = self.llm_client.query(prompt)

        total_time_ms = (time.time() - start_time) * 1000

        # Collect all generated queries for the result
        all_generated_queries = [original_query] + \
            sparse_queries + dense_queries

        result = RAGResult(
            question=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=all_generated_queries if len(
                all_generated_queries) > 1 else None,
            rewritten_docs=None,
            qid=qid,
            system_name="FusionRAGSystem"
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      qid=qid)

        return result


if __name__ == "__main__":
    system = FusionRAGSystem(max_queries=3)
    test_question = "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?"

    print(f"Original question: {test_question}")

    # Generate sparse queries
    sparse_queries = system._generate_queries(test_question, 'sparse')
    print("\nSparse queries (keyword-based search):")
    for i, query in enumerate(sparse_queries, 1):
        print(f"{i}. {query}")

    # Generate dense queries
    dense_queries = system._generate_queries(test_question, 'dense')
    print("\nDense queries (semantic search):")
    for i, query in enumerate(dense_queries, 1):
        print(f"{i}. {query}")

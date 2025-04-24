"""
uv run scripts/run.py --system systems.qpp_fusion.qpp_rag.FusionQPPSystem --help
"""
import re
import time
from typing import List
from datetime import datetime

from evaluators.evaluator_interface import test_evaluator
from evaluators.llm_evaluator.llm_evaluator import LLMEvaluator
from services.ds_data_morgana import QAPair
from utils.logging_utils import get_logger
from utils.fusion_utils import rrf_fusion
from utils.query_utils import generate_query_id
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from services.qpp import QPPService
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface, test_rag_system
from systems.qpp_fusion.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE, SPARSE_QUERY_GENERATION_PROMPT, DENSE_QUERY_GENERATION_PROMPT


class QPPFusionSystem(RAGSystemInterface):
    """
    This RAG system rewrites the question to generate search queries and uses a fusion search strategy
    """

    log = get_logger("fusion_rag_system")
    query_tag_pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
    queries_tag_pattern = re.compile(r'<queries>(.*?)</queries>', re.DOTALL)

    def __init__(self,
                 max_documents: int = 5,
                 max_query_documents: int = 200,
                 max_queries: int = 5,
                 qpp_k: int = 10,
                 max_effective_queries: int = 5,
                 llm_client: str = "ai71_client"):
        """
        Initialize the FusionQPPSystem.

        Args:
            max_documents: Maximum number of documents to stuff to RAG generation
            max_queries: Maximum number of search queries to generate (including the original)
            qpp_k: Number of top documents to use for QPP calculation
            max_effective_queries: Maximum number of most effective queries to use
            llm_client: LLM client to use: ai71_client, general_openai_client
        """
        self.query_service = QueryService()
        self.qpp_service = QPPService(default_k=qpp_k)

        model_id = "tiiuae/falcon3-10b-instruct"
        client_type = llm_client

        # Prepare prompts
        sparse_prompt = SPARSE_QUERY_GENERATION_PROMPT.format(
            max_queries=max_queries)
        dense_prompt = DENSE_QUERY_GENERATION_PROMPT.format(
            max_queries=max_queries)

        # Initialize the appropriate LLM client type
        if llm_client == "general_openai_client":
            # LLM client for answer generation
            self.llm_client = GeneralOpenAIClient(
                model_id=model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
            )

            # LLM client for sparse query generation (keyword-based search)
            self.sparse_query_generator = GeneralOpenAIClient(
                model_id=model_id,
                system_message=sparse_prompt,
                temperature=0.5  # Set temperature for diversity in retry scenarios
            )

            # LLM client for dense query generation (semantic search)
            self.dense_query_generator = GeneralOpenAIClient(
                model_id=model_id,
                system_message=dense_prompt,
                temperature=0.5  # Set temperature for diversity in retry scenarios
            )
        else:
            # LLM client for answer generation
            self.llm_client = AI71Client(
                model_id=model_id,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
            )

            # LLM client for sparse query generation (keyword-based search)
            self.sparse_query_generator = AI71Client(
                model_id=model_id,
                system_message=sparse_prompt,
                temperature=0.5  # Set temperature for diversity in retry scenarios
            )

            # LLM client for dense query generation (semantic search)
            self.dense_query_generator = AI71Client(
                model_id=model_id,
                system_message=dense_prompt,
                temperature=0.5  # Set temperature for diversity in retry scenarios
            )

        self.max_documents = max_documents
        self.max_query_documents = max_query_documents
        self.max_queries = max_queries
        self.max_effective_queries = max_effective_queries
        self.log.info("QPPFusionSystem initialized",
                      llm_model=model_id,
                      llm_client=client_type,
                      max_documents=max_documents,
                      max_queries=max_queries,
                      qpp_k=qpp_k,
                      max_effective_queries=max_effective_queries)

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

    def _calculate_qpp_scores(self, scores: List[float]) -> dict:
        """
        Calculate QPP scores using different methods.

        Args:
            scores: List of retrieval scores

        Returns:
            Dictionary of QPP method names and their scores
        """
        qpp_scores = {}

        # Calculate entropy-based QPP score
        qpp_scores['entropy'] = self.qpp_service.calculate_entropy_qpp(scores)

        # Calculate standard deviation based QPP score
        qpp_scores['sd'] = self.qpp_service.calculate_sd_qpp(scores)

        # Calculate mean-based QPP score
        qpp_scores['mean'] = self.qpp_service.calculate_mean_qpp(scores)

        return qpp_scores

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

        # Collect all generated queries
        all_queries = [original_query] + sparse_queries + dense_queries

        # Store query effectiveness data
        query_effectiveness = []
        hits_per_query = []

        # Process each query to get hits and calculate QPP scores
        for i, query in enumerate(all_queries):
            # Determine search method based on query type
            if i == 0:  # Original query
                hits = self.query_service.query_fusion(
                    query, k=self.max_query_documents)
                query_type = "original"
            elif i <= len(sparse_queries):  # Sparse queries
                hits = self.query_service.query_keywords(
                    query, k=self.max_query_documents)
                query_type = "sparse"
            else:  # Dense queries
                hits = self.query_service.query_embedding(
                    query, k=self.max_query_documents)
                query_type = "dense"

            # Calculate QPP scores for this query's results
            if hits:
                scores = [hit.score for hit in hits]
                qpp_scores = self._calculate_qpp_scores(scores)
                confidence_score = self.qpp_service.calculate_confidence_score(
                    # qpp_scores, weights={'entropy': 1.0})
                    qpp_scores, weights={'entropy': 1.0})

                # Store query effectiveness data
                query_effectiveness.append({
                    'query': query,
                    'query_type': query_type,
                    'qpp_scores': qpp_scores,
                    'confidence_score': confidence_score,
                    'index': i  # Keep original index for stable sorting
                })

                # Store hits for this query
                hits_per_query.append(hits)

                self.log.debug(f"Retrieved documents for {query_type} query",
                               query=query,
                               hits_count=len(hits),
                               confidence_score=confidence_score,
                               entropy=qpp_scores.get('entropy'),
                               sd=qpp_scores.get('sd'),
                               mean=qpp_scores.get('mean'))
            else:
                self.log.warning(
                    f"No hits for {query_type} query", query=query)

        # Sort queries by effectiveness (confidence score)
        query_effectiveness.sort(
            key=lambda x: x['confidence_score'], reverse=True)

        # Take only the top effective queries
        total_queries = len(query_effectiveness)
        effective_queries = query_effectiveness[:self.max_effective_queries]
        removed_queries = total_queries - len(effective_queries)

        self.log.info("Selected most effective queries",
                      total_queries=total_queries,
                      selected_queries=len(effective_queries),
                      removed_queries=removed_queries)

        # Get hits only for the most effective queries
        effective_hits = [hits_per_query[q['index']]
                          for q in effective_queries]

        # Apply fusion to get the top documents using only the most effective queries
        qid = qid or generate_query_id(question)
        fused_docs = rrf_fusion(
            effective_hits, self.max_documents, query_id=qid)

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

        # Extract the effective queries for the result
        effective_query_list = [q['query'] for q in effective_queries]

        # Create QPP metadata for the result
        qpp_metadata = {
            'total_queries': total_queries,
            'selected_queries': len(effective_queries),
            'removed_queries': removed_queries,
            'effective_queries': effective_query_list,
            'query_effectiveness': [
                {
                    'query': q['query'],
                    'query_type': q['query_type'],
                    'confidence_score': q['confidence_score'],
                    'qpp_scores': q['qpp_scores']
                } for q in effective_queries
            ]
        }

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
            system_name="QPPFusionSystem",
            metadata=qpp_metadata
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      effective_queries=len(effective_queries),
                      removed_queries=removed_queries,
                      qid=qid)

        return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # RAG
    result = test_rag_system(QPPFusionSystem(
        llm_client="ai71_client"
    ), "effect of others apathy vs conformity pressure academic performance motivation")

    # Evaluate
    result.qid = "17"
    ref_qa_pair = QAPair.from_dict({
        'question': 'effect of others apathy vs conformity pressure academic performance motivation',
        'answer': "When exposed to others' apathy in academic situations, people with low commitment to academic goals perform worse and give up more easily. In contrast, when exposed to conformity pressure from others, people tend to align with the group's responses even if they know they're wrong, with studies showing people agree with incorrect majority answers about 37% of the time. The key difference is that apathy specifically decreases motivation in uncommitted individuals, while conformity pressure leads people to change their responses regardless of their original motivation level.",
        'context': [],
        'question_categories': [],
        'user_categories': [],
        'document_ids': ['<urn:uuid:0d6cc5da-72f9-426d-a303-627f1184c17c>', '<urn:uuid:34dcfdd9-4d11-46eb-83c3-f86b540b92b4>'],
        'qid': '17',
    })
    test_evaluator(LLMEvaluator(), [result], [ref_qa_pair])

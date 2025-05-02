"""
uv run scripts/run.py --system systems.qpp_pre.qpp_pre_rag.QPPPreSystem --help
"""
import re
import time
from typing import List
from datetime import datetime

from services.ds_data_morgana import QAPair
from utils.logging_utils import get_logger
from utils.query_utils import generate_query_id
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient
from services.llms.bedrock_client import BedrockClient
from services.qpp import QPPService
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface , test_rag_system
from systems.qpp_pre.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE, SPARSE_QUERY_GENERATION_PROMPT


class QPPPreSystem(RAGSystemInterface):
    """
    This RAG system uses query performance prediction for pre-retrieval optimization,
    focusing only on sparse (keyword-based) retrieval.
    """

    log = get_logger("qpp_pre_rag_system")
    query_tag_pattern = re.compile(r'<query>(.*?)</query>', re.DOTALL)
    queries_tag_pattern = re.compile(r'<queries>(.*?)</queries>', re.DOTALL)

    def __init__(self,
                 max_documents: int = 10,
                 max_query_documents: int = 200,
                 max_queries: int = 5,
                 qpp_k: int = 10,
                 max_effective_queries: int = 3,
                 q_gen_model_id: str = "tiiuae/falcon3-10b-instruct",
                 rag_model_id: str = "tiiuae/falcon3-10b-instruct",
                 q_gen_llm_client: str = "ai71_client",
                 rag_llm_client: str = "ai71_client"):
        """
        Initialize the QPPPreSystem.

        Args:
            max_documents: Maximum number of documents to stuff to RAG generation
            max_queries: Maximum number of search queries to generate (including the original)
            qpp_k: Number of top documents to use for QPP calculation
            max_effective_queries: Maximum number of most effective queries to use
            q_gen_model_id: Model ID for query generation LLM
            rag_model_id: Model ID for RAG generation LLM 
            q_gen_llm_client: Client type for query generation LLM, ai71_client, ec2_llm, bedrock_client
            rag_llm_client: Client type for RAG generation LLM, ai71_client, ec2_llm, bedrock_client
        """
        self.query_service = QueryService()
        self.qpp_service = QPPService(default_k=qpp_k)
        self.max_documents = max_documents
        self.max_query_documents = max_query_documents
        self.max_queries = max_queries
        self.max_effective_queries = max_effective_queries

        self.load_llm_clients(q_gen_llm_client, rag_llm_client,
                              q_gen_model_id, rag_model_id)

        self.log.info("QPPPreSystem initialized",
                      q_gen_model_id=q_gen_model_id,
                      rag_model_id=rag_model_id,
                      q_gen_llm_client=q_gen_llm_client,
                      rag_llm_client=rag_llm_client,
                      max_documents=max_documents,
                      max_queries=max_queries,
                      qpp_k=qpp_k,
                      max_effective_queries=max_effective_queries)

    def load_llm_clients(self, q_gen_llm_client: str, rag_llm_client: str, q_gen_model_id: str, rag_model_id: str):
        """
        Load LLM clients for query generation and RAG answer generation.
        
        Args:
            q_gen_llm_client: Client type for query generation
            rag_llm_client: Client type for RAG answer generation
            q_gen_model_id: Model ID for query generation
            rag_model_id: Model ID for RAG answer generation
        """
        # Prepare query generation prompt
        sparse_prompt = SPARSE_QUERY_GENERATION_PROMPT.format(max_queries=self.max_queries)
        
        # Initialize query generation client
        if 'sonnet' in q_gen_model_id.lower() or 'claude' in q_gen_model_id.lower():
            self.query_generator = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        elif q_gen_llm_client == "ec2_llm":
            self.query_generator = EC2LLMClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        elif q_gen_llm_client == "bedrock_client":
            self.query_generator = BedrockClient(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        else:  # Default to AI71Client
            self.query_generator = AI71Client(
                model_id=q_gen_model_id,
                temperature=0.5
            )
        
        # Store system prompt
        self.sparse_prompt = sparse_prompt

        # Initialize RAG answer generation client
        if 'sonnet' in rag_model_id.lower() or 'claude' in rag_model_id.lower():
            self.llm_client = BedrockClient(
                model_id=rag_model_id,
                max_tokens=200,
                temperature=0.0
            )
        elif rag_llm_client == "ec2_llm":
            self.llm_client = EC2LLMClient(
                model_id=rag_model_id,
                max_tokens=200,
                temperature=0.0
            )
        elif rag_llm_client == "bedrock_client":
            self.llm_client = BedrockClient(
                model_id=rag_model_id,
                max_tokens=200,
                temperature=0.0
            )
        else:  # Default to AI71Client
            self.llm_client = AI71Client(
                model_id=rag_model_id,
                max_tokens=200,
                temperature=0.0
            )

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize a query by removing surrounding double quotes and <query> tags.

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

    def _generate_queries(self, question: str, max_retries: int = 2) -> List[str]:
        """
        Generate sparse queries optimized for keyword search.

        Args:
            question: The user's question
            max_retries: Maximum number of retries if query generation fails

        Returns:
            List of generated queries
        """
        for attempt in range(max_retries + 1):
            try:
                prompt = f"User question: {question}"
                response, _ = self.query_generator.complete_chat_once(prompt, self.sparse_prompt)

                # Extract queries from the response
                queries = self._extract_queries(response)

                if queries:
                    self.log.debug(f"Generated sparse queries",
                                   attempt=attempt,
                                   query_count=len(queries),
                                   queries=queries)
                    return queries
                else:
                    self.log.warning(f"No queries extracted, retrying",
                                     attempt=attempt)
            except Exception as e:
                self.log.error(f"Failed to generate queries",
                               error=str(e),
                               attempt=attempt)

            # If we reach here, we need to retry
            if attempt < max_retries:
                self.log.info(f"Retrying query generation",
                              attempt=attempt+1,
                              max_retries=max_retries)
                time.sleep(1)  # Small delay before retry

        # If all retries failed, return empty list
        self.log.error(f"All query generation attempts failed")
        return []

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question using the QPP pre-retrieval approach.

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
        sparse_queries = self._generate_queries(question)
        self.log.debug("Generated sparse queries",
                       queries=sparse_queries, count=len(sparse_queries))

        # Collect all generated queries
        all_queries = [original_query] + sparse_queries

        # Store query effectiveness data
        query_effectiveness = []
        hits_per_query = []

        # Process each query to get hits and calculate QPP scores
        for i, query in enumerate(all_queries):
            # Use keyword search for all queries
            hits = self.query_service.query_keywords(
                query, k=self.max_query_documents)
            query_type = "original" if i == 0 else "sparse"

            # Calculate QPP scores for this query's results
            if hits:
                scores = [hit.score for hit in hits]
                qpp_scores = self._calculate_qpp_scores(scores)
                confidence_score = self.qpp_service.calculate_confidence_score(
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

        # Take only the top effective query
        total_queries = len(query_effectiveness)
        
        if total_queries == 0:
            self.log.error("No effective queries found")
            return RAGResult(
                question=question,
                answer="I'm sorry, I couldn't find relevant information to answer your question.",
                context=[],
                doc_ids=[],
                total_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                generated_queries=all_queries if len(all_queries) > 1 else None,
                rewritten_docs=None,
                qid=qid or generate_query_id(question),
                system_name="QPPPreSystem",
                metadata={"error": "No effective queries found"}
            )
        
        # Use just the best query according to QPP
        best_query = query_effectiveness[0]
        best_query_index = best_query['index']
        best_docs = hits_per_query[best_query_index][:self.max_documents]

        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in best_docs]
        doc_ids = [hit.id for hit in best_docs]

        self.log.debug("Selected documents for context",
                       doc_count=len(doc_contents),
                       doc_ids=doc_ids)

        # Create context for the LLM
        context = "\n\n".join(doc_contents)

        # Generate prompt for the LLM
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context, question=question)

        # Generate answer using the LLM
        answer, _ = self.llm_client.complete_chat_once(prompt, SYSTEM_PROMPT)

        total_time_ms = (time.time() - start_time) * 1000
        qid = qid or generate_query_id(question)

        # Create metadata for the result
        metadata = {
            'total_queries': total_queries,
            'selected_query': best_query['query'],
            'query_type': best_query['query_type'],
            'confidence_score': best_query['confidence_score'],
            'qpp_scores': best_query['qpp_scores'],
            'final_prompt': prompt
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
            system_name="QPPPreSystem",
            metadata=metadata
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      selected_query=best_query['query'],
                      confidence_score=best_query['confidence_score'],
                      qid=qid)

        return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    from evaluators.evaluator_interface import test_evaluator
    from evaluators.llm_evaluator.llm_evaluator import LLMEvaluator
    from services.ds_data_morgana import QAPair
    
    load_dotenv()

    # RAG
    result = test_rag_system(QPPPreSystem(
        rag_llm_client="ai71_client"
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

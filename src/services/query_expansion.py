from typing import List, NamedTuple, Optional

from services.llms.general_openai_client import GeneralOpenAIClient
from services.query_variants import QueryVariantsGenerator, extract_queries
from services.query_expansion_prompts import (
    IS_SIMPLE_SYS_PROMPT,
    QUERY_DECOMPOSITION_SYS_PROMPT,
    QUERY_REPHRASING_SYS_PROMPT,
    SUB_QUERIES_SYS_PROMPT
)
from utils.logging_utils import get_logger

logger = get_logger("query_expansion")


class ExpandedQueries(NamedTuple):
    """
    NamedTuple containing all intermediary results from the query expansion process.
    """
    original_query: str
    components: str  # Decomposition components
    rephrased_query: str
    is_simple: bool
    sub_queries: List[str]


class QueryExpansion:
    """
    Class for expanding queries into components, rephrased queries, and sub-queries.
    """

    def __init__(
        self,
        llm_client: GeneralOpenAIClient,
        is_simple_sys_prompt: Optional[str] = None,
        query_decomposition_sys_prompt: Optional[str] = None,
        query_rephrasing_sys_prompt: Optional[str] = None,
        sub_queries_sys_prompt: Optional[str] = None,
        query_variants_generator: Optional[QueryVariantsGenerator] = None
    ):
        """
        Initialize the QueryExpansion.

        Args:
            llm_client: LLM client to use for query expansion
            is_simple_sys_prompt: System prompt for determining if a question is simple
            query_decomposition_sys_prompt: System prompt for decomposing a question
            query_rephrasing_sys_prompt: System prompt for rephrasing a question
            sub_queries_sys_prompt: System prompt for generating sub-queries
            query_variants_generator: QueryVariantsGenerator instance to use
        """
        self.llm_client = llm_client

        # Use provided prompts or defaults from query_expansion_prompts.py
        self.is_simple_sys_prompt = is_simple_sys_prompt or IS_SIMPLE_SYS_PROMPT
        self.query_decomposition_sys_prompt = query_decomposition_sys_prompt or QUERY_DECOMPOSITION_SYS_PROMPT
        self.query_rephrasing_sys_prompt = query_rephrasing_sys_prompt or QUERY_REPHRASING_SYS_PROMPT
        self.sub_queries_sys_prompt = sub_queries_sys_prompt or SUB_QUERIES_SYS_PROMPT

        # Use provided QueryVariantsGenerator or create a new one
        self.query_variants_generator = query_variants_generator or QueryVariantsGenerator(
            llm_client)

        # Log which prompts are being used
        if is_simple_sys_prompt is None:
            logger.info("Using default is_simple_sys_prompt",
                        prompt=self.is_simple_sys_prompt)
        if query_decomposition_sys_prompt is None:
            logger.info("Using default query_decomposition_sys_prompt",
                        prompt=self.query_decomposition_sys_prompt)
        if query_rephrasing_sys_prompt is None:
            logger.info("Using default query_rephrasing_sys_prompt",
                        prompt=self.query_rephrasing_sys_prompt)
        if sub_queries_sys_prompt is None:
            logger.info("Using default sub_queries_sys_prompt",
                        prompt=self.sub_queries_sys_prompt)

    def is_simple_query(self, question: str) -> bool:
        """
        Check if a question is simple or composite.

        Args:
            question: The question to check

        Returns:
            True if the question is simple, False otherwise
        """
        is_simple, _ = self.llm_client.complete_chat_once(
            system_message=self.is_simple_sys_prompt, message="Question: " + question)
        logger.debug(f"Is simple question", input=question, output=is_simple)
        return "SIMPLE" in is_simple.upper()

    def expand_composite_query(self, question: str) -> List[str]:
        """
        Generate sub-queries for composite questions.

        Args:
            question: The rephrased query

        Returns:
            List of sub-queries
        """
        qs_str, _ = self.llm_client.complete_chat_once(
            system_message=self.sub_queries_sys_prompt, message=f"Question: {question}")

        logger.debug(f"Generated composite sub-queries",
                     input=question, output=qs_str)

        return extract_queries(qs_str)

    def expand_queries(self, question: str, n_queries: int = 5) -> ExpandedQueries:
        """
        Expand a query into components, rephrased query, and sub-queries.

        Args:
            question: The original query
            n_queries: Number of query variants to generate for simple queries

        Returns:
            ExpandedQueries object containing all intermediary results
        """
        # Query decomposition
        components_str, _ = self.llm_client.complete_chat_once(
            system_message=self.query_decomposition_sys_prompt, message="Question: " + question)
        logger.debug(f"Decomposed question",
                     input=question, output=components_str)

        # Query rephrasing
        rephrased_q, _ = self.llm_client.complete_chat_once(
            system_message=self.query_rephrasing_sys_prompt, message=f"Question: {question}\n\n{components_str}")
        logger.debug(f"Rephrased question", input=question, output=rephrased_q)

        # Check if the query is simple
        simple = self.is_simple_query(question)

        if simple:
            # For simple queries, generate query variants
            sub_queries = self.query_variants_generator.generate_variants(
                question, n_queries)
            if not sub_queries:  # Fallback if extraction fails
                sub_queries = [question]
        else:
            # For composite queries, generate sub-queries
            sub_queries = self.expand_composite_query(rephrased_q)
            if not sub_queries:  # Fallback if extraction fails
                sub_queries = [rephrased_q]

        return ExpandedQueries(
            original_query=question,
            components=components_str,
            rephrased_query=rephrased_q,
            is_simple=simple,
            sub_queries=sub_queries
        )

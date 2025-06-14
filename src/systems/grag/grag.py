import time
import json
import os
from typing import List, Literal, Optional
from services.answer_utils import condense_answer
# from services.indicies import QueryService, SearchHit
from services.indicies import SearchHit
from services.mock_indicies import MockQueryService
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient
from services.query_variants import QueryVariantsGenerator
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from services.llms.mini_tgi_client import MiniTGIClient
from services.logits_reranker import LogitsReranker
from services.query_expansion import QueryExpansion
from utils.doc_listing_utils import truncate_doc_listings
from utils.logging_utils import get_logger


class GRAG(RAGSystemInterface):
    # autopep8: off
    def __init__(self,
                 llm_client: Literal['ai71', 'ec2_llm'] = 'ai71',
                 query_gen_prompt_level: Literal['naive', 'medium', 'advanced'] = 'naive',
                 rag_prompt_level: Literal['naive', 'medium', 'advanced'] = 'naive',
                 qpp: Literal['no'] = 'no',
                 enable_hyde: bool = False,
                 first_step_ranker: Literal['keywords', 'embedding', 'both_concat', 'both_fusion'] = 'both_fusion',
                 reranker: Literal['no', 'logits'] = 'logits',
                 context_words_limit: int = 15_000,
                 n_queries: int = 8,
                 initial_retrieval_k_docs: int = 50,
                 query_expansion_mode: Literal['none', 'variants', 'decomposition'] = 'none'):
    # autopep8: on
        """
        Initialize the GRAG.

        Args:
            llm_client: Client for the LLM. Options are 'ai71' or 'ec2_llm', default is 'ai71'.
            query_gen_prompt_level: The level of the query generation prompt to use. Options are 'naive', 'medium', or 'advanced'. Default is 'naive'.
            rag_prompt_level: The level of the RAG prompt to use. Options are 'naive', 'medium', or 'advanced'. Default is 'naive'.
            qpp: Query performance prediction algorithm. Default is 'no'.
            first_step_ranker: The first step ranker to use. Options are 'keywords', 'embedding', 'both_concat', or 'both_fusion'. Default is 'both_concat'.
            reranker: The reranker to use. Options are 'no' or 'logits'. Default is 'no'.
            context_words_limit: Maximum number of words to include in the context. Default is 15,000.
            n_queries: Number of query variants to generate. Default is 5.
            initial_retrieval_k_docs: Total number of documents to retrieve initially. Default is 50.
            query_expansion_mode: Mode for query expansion. Options are 'none' (no expansion), 'variants' (only use query variants), 
                                 or 'decomposition' (full query expansion). Default is 'none'.
        """

        if llm_client == 'ai71':
            self.rag_llm_client = AI71Client()
            self.qgen_llm_client = AI71Client()
        elif llm_client == 'ec2_llm':
            self.rag_llm_client = EC2LLMClient()
            self.qgen_llm_client = EC2LLMClient()

        self.logger = get_logger('grag')

        # Controllers set up
        self.qpp = qpp
        self.first_step_ranker = first_step_ranker
        self.reranker = reranker
        self.n_queries = int(n_queries)
        self.initial_retrieval_k_docs = int(initial_retrieval_k_docs)
        self.context_words_limit = int(context_words_limit)
        self.query_gen_prompt_level = query_gen_prompt_level
        self.query_expansion_mode = query_expansion_mode
        self.enable_hyde = enable_hyde

        if self.reranker != 'no':
            self.logits_llm = MiniTGIClient()
            self.reranker_client = LogitsReranker(
                logits_llm=self.logits_llm, prompt_version="can_answer")

        # Load prompts from JSON files
        current_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_dir, 'answer_gen_prompts.json'), 'r') as f:
            rag_prompts = json.load(f)
        with open(os.path.join(current_dir, 'query_gen_prompts.json'), 'r') as f:
            query_gen_prompts = json.load(f)

        # Store system prompts and primary prompts
        self.rag_system_prompt = rag_prompts[rag_prompt_level]['system_prompt']
        self.rag_primary_prompt = rag_prompts[rag_prompt_level]['primary_prompt']
        self.query_gen_sys_prompt = query_gen_prompts[query_gen_prompt_level]['system_prompt']

        # self.query_service = QueryService()  # LiveRAG services not available - using mock service instead
        self.query_service = MockQueryService()

    def process_question(self, question: str, qid: Optional[str] = None) -> RAGResult:
        """
        Process the question and return the answer.
        """
        start_time = time.time()

        query_variants_generator = QueryVariantsGenerator(
            llm_client=self.qgen_llm_client,
            system_prompt=self.query_gen_sys_prompt
        )

        rephrased_query = question
        # Handle different query expansion modes
        if self.query_expansion_mode == 'none':
            # No query expansion, just use the original question
            queries = [question]
        elif self.query_expansion_mode == 'variants':
            # Only use query variants generator
            queries = query_variants_generator.generate_variants(
                question, self.n_queries)
        elif self.query_expansion_mode == 'decomposition':  # 'decomposition'
            # Use the full query expansion
            query_expansion = QueryExpansion(
                llm_client=self.qgen_llm_client,
                query_variants_generator=query_variants_generator
            )
            qs_res = query_expansion.expand_queries(question, self.n_queries)
            queries = qs_res.sub_queries
            rephrased_query = qs_res.rephrased_query
        else:
            raise ValueError(
                f"Invalid query expansion mode: {self.query_expansion_mode}")
        if self.enable_hyde:
            hyde_system_prompt = "Given the question, write a short hypothetical answer that could be true. Be brief and concise."
            hyde_answer, _ = self.qgen_llm_client.complete_chat_once(
                question, hyde_system_prompt)
            queries.append(hyde_answer.strip())
        # Calculate k per query based on total initial retrieval docs
        k_per_query = int(self.initial_retrieval_k_docs / len(queries))

        # Retrieve documents based on first_step_ranker
        listings: List[List[SearchHit]] = []
        doc_ids_set = set()
        for query in queries:
            if self.first_step_ranker == 'both_concat':
                embed_results = self.query_service.query_embedding(
                    query, k=k_per_query // 2)
                keyword_results = self.query_service.query_keywords(
                    query, k=k_per_query // 2)
                results = embed_results + keyword_results
            elif self.first_step_ranker == 'both_fusion':
                results = self.query_service.query_fusion(
                    query, k=k_per_query, per_source_k=100)
            elif self.first_step_ranker == 'keywords':
                results = self.query_service.query_keywords(
                    query, k=k_per_query)
            elif self.first_step_ranker == 'embedding':
                results = self.query_service.query_embedding(
                    query, k=k_per_query)
            else:
                raise ValueError(
                    f"Invalid first step ranker: {self.first_step_ranker}. Options are 'both_concat', 'both_fusion', 'keywords', or 'embedding'.")

            unique_results: List[SearchHit] = []
            for result in results:
                # if not already added
                if result.metadata.doc_id not in doc_ids_set:
                    doc_ids_set.add(result.metadata.doc_id)
                    unique_results.append(result)
            listings.append(unique_results)

        # Flatten the listings to get all documents
        all_docs = [hit for listing in listings for hit in listing]

        # Apply different processing based on reranker
        if self.reranker == 'logits':
            # For logits reranker, pass all documents directly to reranker
            docs = self.reranker_client.rerank(
                all_docs, question=rephrased_query, words_limit=self.context_words_limit)

            # If reranking returns nothing, fallback to first 10
            if docs is None or len(docs) == 0:
                docs = all_docs[:10]
                self.logger.info(
                    "Logits reranker returned no results, falling back to first 10 documents")
        else:
            # When not using logits reranker, use truncate_doc_listings to protect context limit
            docs = truncate_doc_listings(
                listings=listings, context_word_limit=self.context_words_limit)

        # Prepare context
        context = "\n\n".join(
            [f"Document [{i+1}] {doc.metadata.text}" for i, doc in enumerate(docs)])
        agen_prompt = self.rag_primary_prompt.format(
            context=context, question=question)

        # Generate answer
        answer, _ = self.rag_llm_client.complete_chat_once(
            agen_prompt, self.rag_system_prompt)

        answer = condense_answer(llm_client=self.rag_llm_client,
                                 answer=answer, words_limit=300)

        final_prompt = str([
            {"role": "system", "content": self.rag_system_prompt},
            {"role": "user", "content": agen_prompt}
        ])

        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            metadata={"final_prompt": final_prompt},
            context=[doc.metadata.text for doc in docs],
            doc_ids=[doc.metadata.doc_id for doc in docs],
            generated_queries=queries,
            total_time_ms=(time.time() - start_time) * 1000,
            system_name="GRAG",
        )


if __name__ == "__main__":
    # Test the GRAG system
    rag_system = GRAG(initial_retrieval_k_docs=10, enable_hyde=True,
                              query_gen_prompt_level='naive', reranker='no', first_step_ranker='both_fusion')

    result = rag_system.process_question(
        "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?",
        qid="1"
    )

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Context:", result.context)
    print("Document IDs:", result.doc_ids)
    print("Total Time (s):", result.total_time_ms)
    print("Generated Queries:", result.generated_queries)

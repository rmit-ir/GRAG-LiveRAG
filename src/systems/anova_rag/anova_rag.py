"""
uv run scripts/run.py --system AnovaRAG --help
"""
import re
import time
from typing import List
from services.indicies import QueryService, SearchHit, truncate_docs
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from utils.logging_utils import get_logger
import argparse


class AnovaRAG(RAGSystemInterface):
    def __init__(self, llm_client='ai71', qgen_model_id='tiiuae/falcon3-10b-instruct', qgen_api_base=None, k_queries=5, sanitize_query: bool = False, first_step_ranker = 'bm25+embedding_model', num_first_retrieved_documents=3, expand_doc=False, expand_words_limit=15_000):
        """
        Initialize the AnovaRAG.

        Args:
            llm_client: Client for the LLM. Options are 'ai71' or 'ec2_llm', default is 'ai71'.
            module_query_gen: Select this module for query generation. Options are 'with_number'|'without_number', default is 'with_number'.
            expand_doc: If True, expand the chunks into full docs and preserve ranking. Default is False.
            expand_words_limit: The number of words to keep after expanded chunks to documents.
        """
        if llm_client == 'ai71':
            self.rag_llm_client = AI71Client()
            self.qgen_llm_client = AI71Client()
        elif llm_client == 'ec2_llm':
            self.rag_llm_client = EC2LLMClient()
            self.qgen_llm_client = EC2LLMClient(
                model_id=qgen_model_id, api_base=qgen_api_base)

        # if module_query_gen == 'with_num':
        self.logger = get_logger('anova_rag')
        
        # Controllers set up
        self.k_queries = int(k_queries)
        self.sanitize_query = sanitize_query
        self.first_step_ranker = first_step_ranker
        self.num_first_retrieved_documents = int(num_first_retrieved_documents)

        # Store system prompts
        self.rag_system_prompt = "You are a helpful assistant. Answer the question based on the provided documents."
        self.qgen_system_prompt = f"Generate a list of {k_queries} search query variants based on the user's question, give me one query variant per line. There are no spelling mistakes in the original question. Do not include any other text."
        self.query_service = QueryService()
        self.expand_doc = expand_doc
        self.expand_words_limit = int(expand_words_limit)

    def _create_query_variants(self, question: str) -> List[str]:
        resp_text, _ = self.qgen_llm_client.complete_chat_once(
            question, self.qgen_system_prompt)

        think = re.search(r'<think>(.*?)</think>(.*)', resp_text, re.DOTALL)
        if think:
            self.logger.info(f"Think: {think.group(1)}")
            # Use the second matched group (content after </think>)
            query_text = think.group(2).strip()
        else:
            # If no <think> block, use the entire response
            query_text = resp_text

        queries = query_text.split("\n")
        
        # Remove surrounding quotes, numbers, and prefixes
        if self.sanitize_query:
            queries = [self._sanitize_query(query) for query in queries]
            
        if len(queries) > self.k_queries:
            self.logger.warning(
                f"Number of generated queries ({len(queries)}) exceeds the limit ({self.k_queries}). Truncating.",
                source_queries=queries)
            queries = queries[:self.k_queries]
        # return queries + [question]
        return queries

    def _sanitize_query(self, query: str) -> str:
        query = query.strip()

        # Check for numbered list format (e.g., "1. query" or "11. query")
        import re
        numbered_pattern = re.match(r'^\d+\.\s+(.*)', query)
        if numbered_pattern:
            query = numbered_pattern.group(1)

        # Remove surrounding quotes if present
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            query = query[1:-1]

        # Replace escaped quotes with regular quotes
        query = query.replace("\\'", "'").replace('\\"', '"')

        return query

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process the question and return the answer.
        """

        start_time = time.time()
        queries = self._create_query_variants(question)

        documents: List[SearchHit] = []
        doc_ids = set()
        for query in queries:
            if self.first_step_ranker == 'bm25+embedding_model':
                embed_results = self.query_service.query_embedding(query, k=self.num_first_retrieved_documents)
                keyword_results = self.query_service.query_keywords(query, k=self.num_first_retrieved_documents)
                results = embed_results + keyword_results
            
            elif self.first_step_ranker == 'bm25':
                results = self.query_service.query_keywords(query, k=self.num_first_retrieved_documents)
            elif self.first_step_ranker == 'embedding_model':
                results = self.query_service.query_embedding(query, k=self.num_first_retrieved_documents)
            else:
                raise ValueError(f"Invalid first step ranker: {self.first_step_ranker}. Options are 'bm25+embedding_model', 'bm25', or 'embedding_model'.")
            
            for doc in results:
                if doc.id not in doc_ids:
                    documents.append(doc)
                    doc_ids.add(doc.id)

        # If expand_doc is True, expand the chunks using get_doc while preserving original ranking
        if self.expand_doc:
            ordered_doc_ids = [doc.id for doc in documents]
            full_docs = self.query_service.get_docs(ordered_doc_ids)
            documents = truncate_docs(full_docs, self.expand_words_limit)
            self.logger.info(f"Expanded documents", question=question,
                             taken_docs=len(documents),
                             original_docs=len(full_docs),
                             original_chunks=len(ordered_doc_ids))

        context = "Documents: \n\n"
        context += "\n\n".join([doc.metadata.text for doc in documents])

        prompt = context + "\n\nQuestion: " + question + "\n\nAnswer: "

        answer, _ = self.rag_llm_client.complete_chat_once(
            prompt, self.rag_system_prompt)

        final_prompt = str([
            {"role": "system", "content": self.rag_system_prompt},
            {"role": "user", "content": prompt}
        ])

        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            metadata={"final_prompt": final_prompt},
            context=[doc.metadata.text for doc in documents],
            doc_ids=list(doc_ids),
            generated_queries=queries,
            total_time_ms=(time.time() - start_time) * 1000,
            system_name="AnovaRAG",
        )

    
# def parse_args():
#     parser = argparse.ArgumentParser(description="AnovaRAG system")
    
#     parser.add_argument("--k_queries", 
#                        type=int, 
#                        default=5, 
#                        help="Number of query variants to generate (default: 5)")

#     return parser.parse_args()

if __name__ == "__main__":
    
    # # Parse command-line arguments
    # args = parse_args()
    # k_quires = args.k_queries
    # print("k_queries:", k_quires)
    
    # Create an instance of the AnovaRAG system
    rag_system = AnovaRAG()
    
    result = rag_system.process_question(
        "How does the artwork 'For Proctor Silex' create an interesting visual illusion for viewers as they approach it?",
        qid=1
    )

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Context:", result.context)
    print("Document IDs:", result.doc_ids)
    print("Total Time (s):", result.total_time_ms)
    # print generated queries
    print("Generated Queries:", result.generated_queries)

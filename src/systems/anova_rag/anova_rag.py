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
from services.llms.mini_tgi_client import MiniTGIClient
from utils.logging_utils import get_logger


class AnovaRAG(RAGSystemInterface):
    def __init__(self, 
                 llm_client='ai71', 
                 qgen_model_id='tiiuae/falcon3-10b-instruct', 
                 qgen_api_base=None, 
                 original_question_inlcuded=False,
                 k_queries=5, 
                 sanitize_query: bool = False, 
                 qpp = None,
                 num_first_retrieved_documents=3, 
                 first_step_ranker = 'keywords+embedding_model', 
                 fusion_method='concatenation',
                 reranker='pointwise',
                 num_reranked_documents=15,):
        """
        Initialize the AnovaRAG.

        Args:
            llm_client: Client for the LLM. Options are 'ai71' or 'ec2_llm', default is 'ai71'.
            qgen_model_id: Model ID for the query generation LLM. Default is 'tiiuae/falcon3-10b-instruct'            
            
            original_question_inlcuded: If True, include the original question in the generated queries. Default is False.
            k_queries: Number of query variants to generate. Default is 5.
            sanitize_query: If True, sanitize the generated queries (remove qoutes, numbers, etc.). Default is False.
            qpp: Query preformance prediction algorithm. Not in use now.
            num_first_retrieved_documents: Number of documents to retrieve in the first step. Default is 3.
            first_step_ranker: The first step ranker to use. Options are 'keywords+embedding_model', 'keywords', or 'embedding_model'. Default is 'bm25+embedding_model'.
            fusion_method: The method to use for gathering the first step retrieval results. Options are 'concatenation'. Default is 'concatenation'.
            reranker: The reranker to use. Options are 'pointwise'. Default is 'pointwise'.
            num_reranked_documents: Number of documents returned from reranker. Better be less than num of quries * num of first step retrieved documents. Default is 15.
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
        self.logits_llm = MiniTGIClient()
        
        # Controllers set up
        self.original_question_inlcuded = bool(original_question_inlcuded)
        self.k_queries = int(k_queries)
        self.sanitize_query = sanitize_query
        self.qpp = qpp
        self.first_step_ranker = first_step_ranker
        self.num_first_retrieved_documents = int(num_first_retrieved_documents)
        self.fusion_method = fusion_method
        self.reranker = reranker
        self.num_reranked_documents = num_reranked_documents
        

        # Store system prompts
        self.rag_system_prompt = "You are a helpful assistant. Answer the question based on the provided documents."
        self.qgen_system_prompt = f"Generate a list of {k_queries} search query variants based on the user's question, give me one query variant per line. There are no spelling mistakes in the original question. Do not include any other text."
        self.query_service = QueryService()

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
        if self.original_question_inlcuded:
            return queries + [question]
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
    
    def rerank_by_logits(self, documents: List[SearchHit], question: str, num_reranked_documents: int) -> List[SearchHit]:
        self.log.debug("Documents before reranking", documents=documents)
        id_doc_dict = {doc.id: doc for doc in documents}
        id_yes_prob = []
        for hit in documents:
            doc_text = hit.metadata.text.replace("\n", " ")
            prompt = f"Document: {doc_text}\n\nQuestion: {question}\n\nIs this question generated from this document (only answer 'Yes' or 'No')?\nAnswer:\n\n"
            logits = self.logits_llm.get_token_logits(
                prompt, tokens=['Yes', 'No'])
            yes_raw_prob = logits['raw_probabilities'].get('Yes', 0.0)
            id_yes_prob.append((hit.id, yes_raw_prob))
        self.log.debug("Logits for documents", logits=id_yes_prob)

        sorted_docs = sorted(id_yes_prob, key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, _ in sorted_docs:
            if doc_id in id_doc_dict:
                reranked_docs.append(id_doc_dict[doc_id])
        self.log.debug("Documents after reranking", documents=reranked_docs)

        return reranked_docs[:num_reranked_documents]

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process the question and return the answer.
        """

        start_time = time.time()
        queries = self._create_query_variants(question)

        documents: List[SearchHit] = []
        doc_ids = set()
        for query in queries:
            if self.first_step_ranker == 'keywords+embedding_model':
                embed_results = self.query_service.query_embedding(query, k=self.num_first_retrieved_documents)
                keyword_results = self.query_service.query_keywords(query, k=self.num_first_retrieved_documents)
                if self.fusion_method == 'concatenation':
                    results = embed_results + keyword_results
                else:
                    raise ValueError(f"Invalid fusion method: {self.fusion}. Options are 'concatenation'.")
            
            elif self.first_step_ranker == 'keywords':
                results = self.query_service.query_keywords(query, k=self.num_first_retrieved_documents)
            elif self.first_step_ranker == 'embedding_model':
                results = self.query_service.query_embedding(query, k=self.num_first_retrieved_documents)
            else:
                raise ValueError(f"Invalid first step ranker: {self.first_step_ranker}. Options are 'bm25+embedding_model', 'bm25', or 'embedding_model'.")
            
            for doc in results:
                if doc.id not in doc_ids:
                    documents.append(doc)
                    doc_ids.add(doc.id)

        # Rerank the documents using logits
        if self.reranker == 'no_reranker':
            pass
        elif self.reranker == 'pointwise':
            documents = self.rerank_by_logits(documents, question, self.num_reranked_documents)
        else:
            raise ValueError(f"Invalid reranker: {self.reranker}. Options are 'no_reranker' or 'pointwise'.")
        
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

if __name__ == "__main__":
    
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

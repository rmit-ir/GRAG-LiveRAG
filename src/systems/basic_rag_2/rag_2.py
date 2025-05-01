"""
uv run scripts/run.py --system systems.basic_rag_2.rag_2.BasicRAG2 --help
"""
import time
from typing import List
from services.indicies import QueryService, SearchHit
from services.llms.ai71_client import AI71Client
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface


class BasicRAG2(RAGSystemInterface):
    def __init__(self):
        self.rag_llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
        )
        # self.qgen_llm_client = BedrockClient(
        #     model_id="meta.llama3-1-8b-instruct-v1:0",
        # )
        self.qgen_llm_client = AI71Client(
            model_id="tiiuae/falcon3-10b-instruct",
        )
        
        # Store system prompts
        self.rag_system_prompt = "You are a helpful assistant. Answer the question based on the provided documents."
        self.qgen_system_prompt = "Generate a list of 5 search query variants based on the user's question, give me one query variant per line. Do not include any other text."
        self.query_service = QueryService()

    def _create_query_variants(self, question: str) -> List[str]:
        queries, _ = self.qgen_llm_client.complete_chat_once(question, self.qgen_system_prompt)
        queries = queries.split("\n")
        queries = [query.strip() for query in queries]
        return queries

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process the question and return the answer.
        """

        start_time = time.time()
        queries = self._create_query_variants(question)

        documents: List[SearchHit] = []
        doc_ids = set()
        for query in queries:
            embed_results = self.query_service.query_embedding(query, k=3)
            keyword_results = self.query_service.query_keywords(query, k=3)
            results = embed_results + keyword_results
            for doc in results:
                if doc.id not in doc_ids:
                    documents.append(doc)
                    doc_ids.add(doc.id)

        context = "Documents: \n\n"
        context += "\n\n".join([doc.metadata.text for doc in documents])

        prompt = context + "\n\nQuestion: " + question + "\n\nAnswer: "

        answer, _ = self.rag_llm_client.complete_chat_once(prompt, self.rag_system_prompt)

        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            context=[doc.metadata.text for doc in documents],
            doc_ids=list(doc_ids),
            total_time_ms=(time.time() - start_time) * 1000,
        )


if __name__ == "__main__":
    # Test the BasicRAG2 system
    rag_system = BasicRAG2()
    result = rag_system.process_question(
        "What are the different approaches being developed to target IL-6 in treating myeloma, and what kinds of medications are being tested?",
        qid="test_qid_1"
    )
    
    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Context:", result.context)
    print("Document IDs:", result.doc_ids)
    print("Total Time (s):", result.total_time_ms)

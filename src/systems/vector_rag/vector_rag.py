"""
uv run scripts/run.py --system systems.vector_rag.vector_rag.VectorRAG --help
"""
import time
from datetime import datetime

from utils.logging_utils import get_logger
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface, test_rag_system
from systems.vector_rag.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class VectorRAG(RAGSystemInterface):
    """
    Vector RAG system implementation.
    """

    log = get_logger("vector_rag_system")

    def __init__(self, llm_client="ai71_client"):
        """
        Initialize the VectorRAG.

        Args:
            llm_client: LLM client to use: ai71_client, general_openai_client
        """
        self.query_service = QueryService()

        model_id = "tiiuae/falcon3-10b-instruct"

        if llm_client == "general_openai_client":
            self.llm_client = GeneralOpenAIClient(
                model_id='tiiuae/Falcon3-10B-Instruct',
                system_message=SYSTEM_PROMPT
            )
        else:
            self.llm_client = AI71Client(
                model_id=model_id,
                system_message=SYSTEM_PROMPT
            )

        self.log.info("VectorRAG initialized",
                      llm_model=model_id,
                      llm_client=llm_client)

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)

        # Search for documents using vector embedding search
        hits = self.query_service.query_embedding(question, k=10)
        self.log.debug("Retrieved documents", hits_count=len(hits))

        # Extract document contents and IDs
        doc_contents = [hit.metadata.text for hit in hits]
        context = "\n\n".join(doc_contents)
        doc_ids = [hit.id for hit in hits]

        self.log.debug("Selected documents for context",
                       doc_count=len(doc_contents),
                       doc_ids=doc_ids)

        # Generate answer using the LLM
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context, question=question)
        _, answer = self.llm_client.query(prompt)

        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000

        result = RAGResult(
            question=question,
            answer=answer,
            context=doc_contents,
            doc_ids=doc_ids,
            total_time_ms=total_time_ms,
            timestamp=datetime.now(),
            generated_queries=None,
            rewritten_docs=None,
            qid=qid,
            system_name="VectorRAG"
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      qid=qid)

        return result


# Main entry point for testing the Vector RAG system
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    test_rag_system(VectorRAG(), "What is the capital of France?")

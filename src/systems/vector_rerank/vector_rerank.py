"""
uv run scripts/run.py --system systems.vector_rerank.vector_rerank.VectorRerank --help
"""
import time
from datetime import datetime

from evaluators.evaluator_interface import test_evaluator
from evaluators.llm_evaluator.llm_evaluator import LLMEvaluator
from services.ds_data_morgana import QAPair
from utils.logging_utils import get_logger
from services.indicies import QueryService
from rerankers.llm_reranker.llm_reranker import LLMReranker
from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface, test_rag_system
from systems.vector_rerank.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class VectorRerank(RAGSystemInterface):
    """
    Vector RAG system with LLM-based reranking implementation.
    """

    log = get_logger("vector_rerank_system")

    def __init__(
        self,
        rag_llm_client="ai71_client",
        reranker_strategy="setwise",
        reranker_k=10,
        role_playing=True,
        tone_words="Please",
        evidence_first=True
    ):
        """
        Initialize the VectorRerank.

        Args:
            llm_client: LLM client to use: ai71_client, general_openai_client
            reranker_strategy: Reranking strategy (pointwise, pairwise, listwise, setwise)
            reranker_k: Number of documents to return after reranking
            role_playing: Whether to use role playing in reranker prompts
            tone_words: Tone words to use in reranker prompts (None for no tone words)
            evidence_first: Whether to put evidence before instructions in reranker prompts
        """
        self.query_service = QueryService()
        self.reranker_strategy = reranker_strategy
        self.reranker_k = reranker_k
        self.role_playing = role_playing
        self.tone_words = tone_words
        self.evidence_first = evidence_first

        model_id = "tiiuae/falcon3-10b-instruct"

        if rag_llm_client == "general_openai_client":
            self.rag_llm_client = GeneralOpenAIClient(
                model_id='tiiuae/falcon3-10b-instruct',
                system_message=SYSTEM_PROMPT
            )
        else:
            self.rag_llm_client = AI71Client(
                model_id=model_id,
                system_message=SYSTEM_PROMPT
            )

        # Initialize reranker
        self.reranker = LLMReranker(
            llm_client=self.rag_llm_client,
            strategy=self.reranker_strategy,
            role_playing=self.role_playing,
            tone_words=self.tone_words,
            evidence_first=self.evidence_first
        )

        self.log.info("VectorRerank initialized",
                      llm_model=model_id,
                      rag_llm_client=rag_llm_client,
                      reranker_strategy=self.reranker_strategy,
                      role_playing=self.role_playing,
                      tone_words=self.tone_words,
                      evidence_first=self.evidence_first)

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)

        # Search for documents using vector embedding search
        hits = self.query_service.query_embedding(question, k=100)
        self.log.debug("Retrieved documents", hits_count=len(hits))

        # Apply reranking
        rerank_start_time = time.time()
        self.log.debug("Applying LLM reranking")
        hits = self.reranker.rerank(question, hits, k=self.reranker_k)
        rerank_time_ms = (time.time() - rerank_start_time) * 1000
        self.log.debug("Reranking completed",
                       reranked_hits_count=len(hits),
                       rerank_time_ms=rerank_time_ms)

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
        _, answer = self.rag_llm_client.query(prompt)

        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000

        # Prepare metadata
        metadata = {
            "reranker": {
                "strategy": self.reranker.strategy,
                "role_playing": self.reranker.role_playing,
                "evidence_first": self.reranker.evidence_first,
                "tone_words": self.reranker.tone_words,
                "rerank_time_ms": rerank_time_ms
            }
        }

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
            system_name="VectorRerank",
            metadata=metadata
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      qid=qid)

        return result


# Main entry point for testing the Vector Rerank system
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = test_rag_system(VectorRerank(
        rag_llm_client="general_openai_client",
    ), "I'm interested in crime-solving games but want something more educational - are there any games that teach you real skills while you investigate cases?")

    # Evaluate
    result.qid = "27"
    ref_qa_pair = QAPair.from_dict({
        'question': "I'm interested in crime-solving games but want something more educational - are there any games that teach you real skills while you investigate cases?",
        'answer': "Where in the World is Carmen Sandiego? is a notable example of an educational crime-solving game series. While solving cases, players need to use real history and geography knowledge and employ genuine research and deduction skills. The game follows a procedural pattern of heist, investigation, and capturing thieves, and while its crime-solving may not be as realistic as other police games, it requires players to learn and apply real knowledge to progress.",
        'context': [],
        'question_categories': [],
        'user_categories': [],
        'document_ids': ['<urn:uuid:0de73eca-9722-484e-a48b-b75a60516639>'],
        'qid': result.qid,
    })
    test_evaluator(LLMEvaluator(), [result], [ref_qa_pair])

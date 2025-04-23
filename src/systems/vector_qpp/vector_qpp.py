"""
uv run scripts/run.py --system systems.vector_qpp.vector_qpp.VectorQPP --help
"""
import time
from evaluators.evaluator_interface import test_evaluator
from evaluators.llm_evaluator.llm_evaluator import LLMEvaluator
from datetime import datetime
from typing import List, Dict

from services.ds_data_morgana import QAPair
from services.qpp import QPPService
from utils.logging_utils import get_logger
from services.indicies import QueryService
from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface, test_rag_system
from systems.vector_qpp.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


class VectorQPP(RAGSystemInterface):
    """
    RAG system implementation with vector search and Query Performance Prediction.
    """

    log = get_logger("vector_qpp_system")

    def __init__(self, llm_client="ai71_client", qpp_k: int = 10):
        """
        Initialize the VectorQPP.

        Args:
            llm_client: LLM client to use: ai71_client, general_openai_client
            qpp_k: Number of top documents to use for QPP calculation
        """
        self.query_service = QueryService()
        self.qpp_service = QPPService(default_k=qpp_k)

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

        self.log.info("VectorQPP initialized",
                      llm_model=model_id,
                      llm_client=llm_client,
                      qpp_k=qpp_k)

    def process_question(self, question: str, qid: str = None) -> RAGResult:
        """
        Process a question using vector search and QPP.

        Args:
            question: The user's question
            qid: Optional query ID

        Returns:
            RAGResult containing the answer, context, and QPP metadata
        """
        start_time = time.time()
        self.log.info("Processing question", question=question, qid=qid)

        # Search for documents using embedding search
        hits = self.query_service.query_embedding(question, k=10)
        self.log.debug("Retrieved documents", hits_count=len(hits))

        # Extract document contents, IDs, and scores
        doc_contents = [hit.metadata.text for hit in hits]
        doc_ids = [hit.id for hit in hits]
        doc_scores = [hit.score for hit in hits]

        # Calculate QPP scores
        qpp_scores = self._calculate_qpp_scores(doc_scores)
        confidence_score = self.qpp_service.calculate_confidence_score(
            qpp_scores)

        self.log.debug("QPP scores calculated",
                       entropy=qpp_scores.get('entropy'),
                       sd=qpp_scores.get('sd'),
                       mean=qpp_scores.get('mean'),
                       confidence=confidence_score)

        # Join document contents to create context
        context = "\n\n".join(doc_contents)

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
            system_name="VectorQPP",
            metadata={
                'qpp_scores': qpp_scores,
                'confidence_score': confidence_score
            }
        )

        self.log.info("Generated answer",
                      answer_length=result.answer_words_count,
                      processing_time_ms=total_time_ms,
                      confidence=confidence_score,
                      qid=qid)

        return result

    def _calculate_qpp_scores(self, scores: List[float]) -> Dict[str, float]:
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


# Main entry point for testing the basic RAG system
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # RAG
    result = test_rag_system(VectorQPP(
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

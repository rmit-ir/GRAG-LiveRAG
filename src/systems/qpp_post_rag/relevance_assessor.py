"""
LLM-based relevance assessor for QPP calculations.
"""
from typing import List
from services.llms.ai71_client import AI71Client
from services.llms.ec2_llm_client import EC2LLMClient

class RelevanceAssessor:
    def __init__(self, llm_client='ai71'):
        """Initialize relevance assessor with specified LLM client."""
        if llm_client == 'ai71':
            self.llm_client = AI71Client(
                model_id="tiiuae/falcon3-10b-instruct",
            )
        elif llm_client == 'ec2_llm':
            self.llm_client = EC2LLMClient(
                model_id="tiiuae/falcon3-10b-instruct",
            )
            
        self.system_prompt = """You are a relevance assessment expert.
Rate how relevant a passage is to answering a question on a scale from 0 to 1.
0 means completely irrelevant, 1 means highly relevant.
Only output a single number between 0 and 1."""

    def assess_relevance(self, question: str, passage: str) -> float:
        """Assess relevance of a single passage to the question."""
        prompt = f"""Rate how relevant this passage is to answering the question on a scale from 0 to 1.
Question: {question}
Passage: {passage}
Score:"""

        result, _ = self.llm_client.complete_chat_once(prompt, self.system_prompt)
        try:
            score = float(result.strip())
            return max(0.0, min(1.0, score))  # Clip between 0 and 1
        except ValueError:
            return 0.0

    def assess_batch(self, question: str, passages: List[str]) -> List[float]:
        """Assess relevance for a batch of passages."""
        return [self.assess_relevance(question, passage) for passage in passages]
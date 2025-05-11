"""
Classes for handling LiveRAG Challenge input and output formats.
"""
from dataclasses import dataclass
import re
from typing import List, Dict, Any, TypedDict
import json
from services.ds_data_morgana import QAPair
from systems.rag_result import RAGResult


@dataclass
class LiveRAGQuestion:
    """
    Represents a question in the LiveRAG Challenge format.

    Example:
    {
      "id": 17,
      "question": "What is the capital of France?" 
    }
    """
    id: int
    question: str

    @classmethod
    def from_json(cls, json_str: str) -> 'LiveRAGQuestion':
        """
        Create a LiveRAGQuestion from a JSON string.

        Args:
            json_str: JSON string containing question data

        Returns:
            LiveRAGQuestion instance
        """
        data = json.loads(json_str)
        return cls(
            id=data["id"],
            question=data["question"]
        )

    def to_qapair(self) -> QAPair:
        """
        Convert to QAPair object.

        Returns:
            QAPair instance
        """
        return QAPair(
            question=self.question,
            answer="",  # Empty as this is just a question
            context=[],
            question_categories=[],
            user_categories=[],
            document_ids=[],
            qid=str(self.id)
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiveRAGQuestion':
        return cls(
            # Default to hash of question
            id=data.get('id', hash(data["question"]) % 1000000),
            question=data.get('question')
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "question": self.question
        }


class LiveRAGPassage(TypedDict):
    """
    Represents a passage with related document IDs in the LiveRAG Challenge format.

    Example:
    {
      "passage": "Paris is the capital and most populous city of France.",
      "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>", "<urn:uuid:1234abcd-5678-efgh-9202-ijklmnopqrst>"]
    }
    """
    passage: str
    doc_IDs: List[str]


@dataclass
class LiveRAGAnswer:
    """
    Represents an answer in the LiveRAG Challenge format.

    Example:
    {
      "id": 17,
      "question": "What is the capital of France?",
      "passages": [
        {
          "passage": "Paris is the capital and most populous city of France.",
          "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>", "<urn:uuid:1234abcd-5678-efgh-9202-ijklmnopqrst>"]
        },
        {
          "passage": "France is located in Western Europe.",
          "doc_IDs": ["<urn:uuid:1234abcd-5678-efgh-9101-ijklmnopqrst>"]
        }
      ],
      "final_prompt": "Using the following - Paris is the capital and most populous city of France - and - France is located in Western Europe - answer the question: What is the capital of France?",
      "answer": "Paris" 
    }
    """
    id: int
    question: str
    passages: List[LiveRAGPassage]
    final_prompt: str
    answer: str

    @classmethod
    def from_rag_result(cls, result: RAGResult) -> 'LiveRAGAnswer':
        """
        Create a LiveRAGAnswer from a RAGResult object.

        Args:
            result: RAGResult object containing answer data

        Returns:
            LiveRAGAnswer instance
        """
        # Convert qid to int if possible, otherwise use a default
        try:
            qid = int(result.qid) if result.qid else 0
        except (ValueError, TypeError):
            qid = 0

        # Create passages from context and doc_ids
        passages = []

        # If context and doc_ids have the same length, pair them directly
        if len(result.context) == len(result.doc_ids):
            for passage, doc_id in zip(result.context, result.doc_ids):
                # transform doc-<urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>::chunk-1
                # to <urn:uuid:8cfe9f92-9499-422a-a4a5-55a7ae879410>
                doc_id = re.sub(r'doc-(.*)::chunk-\d+', r'\1', doc_id)
                passages.append(LiveRAGPassage(
                    passage=passage,
                    doc_IDs=[doc_id]
                ))

        # Get final prompt from the result's metadata, or construct a simple one if not available
        final_prompt = result.metadata.get('final_prompt', 'No final prompt provided in the result.')

        return cls(
            id=qid,
            question=result.question,
            passages=passages,
            final_prompt=final_prompt,
            answer=result.answer
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "question": self.question,
            "passages": [
                {
                    "passage": p["passage"],
                    "doc_IDs": p["doc_IDs"]
                } for p in self.passages
            ],
            "final_prompt": self.final_prompt,
            "answer": self.answer
        }

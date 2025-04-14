from typing import Dict, Any
from dataclasses import dataclass
from utils.logging_utils import get_logger

log = get_logger("live_rag_metadata")


@dataclass
class LiveRAGMetadata:
    """
    Metadata class for LiveRAG chunks.
    Contains information about chunk position, document ID, and the text itself.
    """
    chunk_order: float
    doc_id: str
    is_first_chunk: bool
    is_last_chunk: bool
    text: str
    total_doc_chunks: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiveRAGMetadata':
        """
        Create a LiveRAGMetadata instance from a dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            LiveRAGMetadata instance with values from the dictionary
        """
        log.debug(f"Creating LiveRAGMetadata from dict with doc_id: {data.get('doc_id', 'unknown')}")
        return cls(
            chunk_order=data.get('chunk_order', 0.0),
            doc_id=data.get('doc_id', ''),
            is_first_chunk=data.get('is_first_chunk', False),
            is_last_chunk=data.get('is_last_chunk', False),
            text=data.get('text', ''),
            total_doc_chunks=data.get('total_doc_chunks', 0.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metadata to a dictionary.

        Returns:
            Dictionary representation of the metadata
        """
        return {
            'chunk_order': self.chunk_order,
            'doc_id': self.doc_id,
            'is_first_chunk': self.is_first_chunk,
            'is_last_chunk': self.is_last_chunk,
            'text': self.text,
            'total_doc_chunks': self.total_doc_chunks
        }

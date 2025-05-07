"""
Setwise reranker implementation based on the paper:
"A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models"
by Shengyao Zhuang and Honglei Zhuang.

This module implements the setwise approach for document reranking using LLMs.
"""
from typing import List, Optional
import re

from rerankers.reranker_interface import RerankerInterface
from services.llms.llm_interface import LLMInterface
from utils.logging_utils import get_logger
from services.indicies import SearchHit
from utils.namedtuple_utils import update_tuple


class SetwiseReranker(RerankerInterface):
    """
    Setwise reranker that uses LLMs to compare multiple documents at once.
    
    This implementation is based on the paper:
    "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models"
    
    It supports both heap sort and bubble sort algorithms for efficient reranking.
    """

    log = get_logger("setwise_reranker")

    def __init__(
        self,
        llm_client: LLMInterface,
        algorithm: str = "heapsort",
        compare_size: int = 3,
        role_playing: bool = True,
        tone_words: Optional[str] = "Please",
        evidence_first: bool = True,
        max_tokens_per_doc: int = 200
    ):
        """
        Initialize the setwise reranker.

        Args:
            llm_client: LLM client to use for reranking
            algorithm: Sorting algorithm to use (heapsort or bubblesort)
            compare_size: Number of documents to compare at once (c parameter in the paper)
            role_playing: Whether to use role playing in prompts
            tone_words: Tone words to use in prompts (None for no tone words)
            evidence_first: Whether to put evidence before instructions in prompts
            max_tokens_per_doc: Maximum number of tokens per document
        """
        self.llm_client = llm_client
        self.algorithm = algorithm.lower()
        self.compare_size = compare_size
        self.role_playing = role_playing
        self.tone_words = tone_words
        self.evidence_first = evidence_first
        self.max_tokens_per_doc = max_tokens_per_doc

        # Validate algorithm
        valid_algorithms = ["heapsort", "bubblesort"]
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid sorting algorithm: {algorithm}. Must be one of {valid_algorithms}")

        # Validate compare_size
        if self.compare_size < 2:
            raise ValueError(
                f"Invalid compare_size: {compare_size}. Must be at least 2.")

        self.log.info("Setwise reranker initialized",
                      algorithm=self.algorithm,
                      compare_size=self.compare_size,
                      role_playing=self.role_playing,
                      tone_words=self.tone_words,
                      evidence_first=self.evidence_first)

    def rerank(self, query: str, hits: List[SearchHit], k: int = None) -> List[SearchHit]:
        """
        Rerank the retrieved documents using the setwise approach.

        Args:
            query: The user query
            hits: List of retrieved documents
            k: Number of documents to return after reranking (default: all)

        Returns:
            Reranked list of SearchHit objects
        """
        if not hits:
            self.log.warning("No documents to rerank")
            return hits

        if k is None:
            k = len(hits)

        self.log.info("Reranking documents",
                      algorithm=self.algorithm,
                      hits_count=len(hits),
                      k=k)

        # Truncate documents to max_tokens_per_doc
        truncated_hits = self._truncate_documents(hits)

        # Apply the appropriate sorting algorithm
        if self.algorithm == "heapsort":
            reranked_hits = self._heapsort_rerank(query, truncated_hits, k)
        elif self.algorithm == "bubblesort":
            reranked_hits = self._bubblesort_rerank(query, truncated_hits, k)
        else:
            # This should never happen due to validation in __init__
            self.log.error("Unknown sorting algorithm",
                           algorithm=self.algorithm)
            return hits[:k]

        self.log.info("Reranking completed",
                      original_hits=len(hits),
                      reranked_hits=len(reranked_hits))

        return reranked_hits[:k]

    def _truncate_documents(self, hits: List[SearchHit]) -> List[SearchHit]:
        """
        Truncate documents to max_tokens_per_doc.
        This is a simple word-based truncation, not token-based.

        Args:
            hits: List of retrieved documents

        Returns:
            List of SearchHit objects with truncated text
        """
        truncated_hits = []
        for hit in hits:
            # Simple word-based truncation (approximation of tokens)
            words = hit.metadata.text.split()
            if len(words) > self.max_tokens_per_doc:
                truncated_text = " ".join(
                    words[:self.max_tokens_per_doc]) + "..."
                # Create a new metadata object with truncated text
                new_metadata = type(hit.metadata)(
                    chunk_order=hit.metadata.chunk_order,
                    doc_id=hit.metadata.doc_id,
                    is_first_chunk=hit.metadata.is_first_chunk,
                    is_last_chunk=hit.metadata.is_last_chunk,
                    text=truncated_text,
                    total_doc_chunks=hit.metadata.total_doc_chunks
                )
                # Create a new hit with the new metadata
                truncated_hit = SearchHit(
                    id=hit.id,
                    score=hit.score,
                    metadata=new_metadata,
                    retrieval_model=hit.retrieval_model
                )
            else:
                # No truncation needed, use original hit
                truncated_hit = hit
                
            truncated_hits.append(truncated_hit)

        return truncated_hits

    def _build_prompt(self, query: str, docs: List[str], doc_ids: List[str] = None) -> str:
        """
        Build a prompt for the LLM to compare multiple documents.

        Args:
            query: The user query
            docs: List of document texts
            doc_ids: List of document IDs (optional)

        Returns:
            Prompt for the LLM
        """
        # Role playing component
        role_playing_text = ""
        if self.role_playing:
            role_playing_text = "You are an expert document selection system. "

        # Tone words component
        tone_text = ""
        if self.tone_words:
            tone_text = f"{self.tone_words} "

        # Task instruction
        task_instruction = f"Select the most relevant passage for the query: {query}"
        output_type = f"{tone_text}Return only the number of the most relevant passage."

        # Format documents with IDs
        if doc_ids is None:
            doc_ids = [str(i+1) for i in range(len(docs))]

        formatted_docs = []
        for i, (doc_id, doc) in enumerate(zip(doc_ids, docs)):
            formatted_docs.append(f"[{doc_id}] {doc}")

        evidence_text = "\n\n".join(formatted_docs)

        # Assemble the prompt based on evidence_first flag
        if self.evidence_first:
            prompt = f"{role_playing_text}{evidence_text}\n\n{task_instruction}\n\n{output_type}"
        else:
            prompt = f"{role_playing_text}{task_instruction}\n\n{evidence_text}\n\n{output_type}"

        return prompt

    def _parse_llm_response(self, response: str) -> int:
        """
        Parse the LLM response to extract the selected document index.

        Args:
            response: The LLM response

        Returns:
            The index of the selected document (0-based)
        """
        # Look for a number in the response
        match = re.search(r'\d+', response)
        if match:
            try:
                # Convert to 0-based index
                selected_idx = int(match.group()) - 1
                return selected_idx
            except ValueError:
                self.log.warning(
                    "Failed to parse response as integer", response=response)
                return 0
        else:
            self.log.warning("No number found in response", response=response)
            return 0

    def _compare_documents(self, query: str, docs: List[SearchHit]) -> int:
        """
        Compare a set of documents and return the index of the most relevant one.

        Args:
            query: The user query
            docs: List of documents to compare

        Returns:
            Index of the most relevant document (0-based)
        """
        # Extract document texts
        doc_texts = [doc.metadata.text for doc in docs]
        
        # Build prompt for setwise comparison
        prompt = self._build_prompt(query, doc_texts)
        
        # Get selection from LLM
        response, _ = self.llm_client.complete_chat_once(prompt, None)
        
        # Parse response to get the selected document index
        selected_idx = self._parse_llm_response(response)
        
        # Ensure the index is valid
        if selected_idx < 0 or selected_idx >= len(docs):
            self.log.warning("Invalid index from LLM response",
                           index=selected_idx,
                           valid_range=f"0-{len(docs)-1}",
                           response=response)
            selected_idx = 0
            
        return selected_idx

    def _heapsort_rerank(self, query: str, hits: List[SearchHit], k: int) -> List[SearchHit]:
        """
        Rerank documents using the setwise heap sort algorithm.
        
        This implementation is based on the paper's description of setwise heap sort,
        where multiple documents are compared at once during the heapify operation.

        Args:
            query: The user query
            hits: List of retrieved documents
            k: Number of documents to return

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing setwise heapsort reranking", 
                      hits_count=len(hits),
                      compare_size=self.compare_size,
                      k=k)
        
        if len(hits) <= 1:
            return hits
            
        # Create a copy of the hits to avoid modifying the original
        docs = hits.copy()
        n = len(docs)
        
        # Build a max heap
        # Start from the last non-leaf node and work backwards
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(query, docs, n, i)
            
        # Extract elements from the heap one by one
        result = []
        for i in range(n - 1, 0, -1):
            # Move current root to the end
            docs[0], docs[i] = docs[i], docs[0]
            result.append(docs[i])
            
            # If we have enough documents, stop early
            if len(result) >= k:
                break
                
            # Call heapify on the reduced heap
            self._heapify(query, docs, i, 0)
            
        # Add the last element if needed
        if len(result) < k and len(result) < n:
            result.append(docs[0])
            
        # Create new instances with updated scores based on rank
        scored_result = []
        for i, hit in enumerate(result):
            new_score = len(result) - i
            scored_result.append(update_tuple(hit, score=new_score))
            
        return scored_result

    def _heapify(self, query: str, docs: List[SearchHit], n: int, i: int) -> None:
        """
        Heapify a subtree rooted at index i.
        
        This is a modified heapify operation that compares multiple documents at once.

        Args:
            query: The user query
            docs: List of documents
            n: Size of the heap
            i: Root index of the subtree to heapify
        """
        # Initialize largest as root
        largest = i
        
        # Collect child nodes to compare
        children = []
        for j in range(1, self.compare_size):
            child_idx = 2 * i + j
            if child_idx < n:
                children.append(child_idx)
                
        if not children:
            # No children to compare, return
            return
            
        # Compare the root with its children
        compare_docs = [docs[i]] + [docs[j] for j in children]
        compare_indices = [i] + children
        
        # Find the most relevant document
        selected_idx = self._compare_documents(query, compare_docs)
        largest = compare_indices[selected_idx]
        
        # If the largest is not the root, swap and continue heapifying
        if largest != i:
            docs[i], docs[largest] = docs[largest], docs[i]
            self._heapify(query, docs, n, largest)

    def _bubblesort_rerank(self, query: str, hits: List[SearchHit], k: int) -> List[SearchHit]:
        """
        Rerank documents using the setwise bubble sort algorithm.
        
        This implementation is based on the paper's description of setwise bubble sort,
        where multiple documents are compared at once during each bubble operation.

        Args:
            query: The user query
            hits: List of retrieved documents
            k: Number of documents to return

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing setwise bubblesort reranking", 
                      hits_count=len(hits),
                      compare_size=self.compare_size,
                      k=k)
        
        if len(hits) <= 1:
            return hits
            
        # Create a copy of the hits to avoid modifying the original
        docs = hits.copy()
        n = len(docs)
        
        # Perform bubble sort with early stopping
        for i in range(min(k, n)):
            # Last i elements are already in place
            for j in range(0, n - i - 1, self.compare_size - 1):
                # Compare a set of documents
                end_idx = min(j + self.compare_size, n - i)
                compare_docs = docs[j:end_idx]
                
                if len(compare_docs) <= 1:
                    continue
                    
                # Find the most relevant document
                selected_idx = self._compare_documents(query, compare_docs)
                
                # Move the most relevant document to the end of the current window
                if selected_idx < len(compare_docs) - 1:
                    # Swap the selected document with the last one in the window
                    selected_global_idx = j + selected_idx
                    last_idx = j + len(compare_docs) - 1
                    docs[selected_global_idx], docs[last_idx] = docs[last_idx], docs[selected_global_idx]
        
        # The last k elements are the most relevant ones
        result = docs[-k:]
        
        # Reverse to get descending order of relevance
        result.reverse()
        
        # Create new instances with updated scores based on rank
        scored_result = []
        for i, hit in enumerate(result):
            new_score = len(result) - i
            scored_result.append(update_tuple(hit, score=new_score))
            
        return scored_result

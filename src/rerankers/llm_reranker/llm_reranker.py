"""
Reranker service for reranking retrieved documents.

This module provides different reranking strategies based on LLMs as described in
the paper "An Investigation of Prompt Variations for Zero-shot LLM-based Rankers".
"""
from typing import List, Optional

from rerankers.reranker_interface import RerankerInterface
from services.llms.llm_interface import LLMInterface
from utils.logging_utils import get_logger
from services.indicies import SearchHit
from utils.namedtuple_utils import update_tuple


class LLMReranker(RerankerInterface):
    """
    LLM-based reranker that uses different strategies (pointwise, pairwise, listwise, setwise)
    to rerank documents.
    """

    log = get_logger("llm_reranker")

    def __init__(
        self,
        llm_client: LLMInterface,
        strategy: str = "setwise",
        role_playing: bool = True,
        tone_words: Optional[str] = "Please",
        evidence_first: bool = True,
        max_tokens_per_doc: int = 200
    ):
        """
        Initialize the LLM reranker.

        Args:
            llm_client: LLM client to use for reranking
            strategy: Reranking strategy (pointwise, pairwise, listwise, setwise)
            role_playing: Whether to use role playing in prompts
            tone_words: Tone words to use in prompts (None for no tone words)
            evidence_first: Whether to put evidence before instructions in prompts
            max_tokens_per_doc: Maximum number of tokens per document
        """
        self.llm_client = llm_client
        self.strategy = strategy.lower()
        self.role_playing = role_playing
        self.tone_words = tone_words
        self.evidence_first = evidence_first
        self.max_tokens_per_doc = max_tokens_per_doc

        # Validate strategy
        valid_strategies = ["pointwise", "pairwise", "listwise", "setwise"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid reranking strategy: {strategy}. Must be one of {valid_strategies}")

        self.log.info("LLM reranker initialized",
                      strategy=self.strategy,
                      role_playing=self.role_playing,
                      tone_words=self.tone_words,
                      evidence_first=self.evidence_first)

    def rerank(self, query: str, hits: List[SearchHit], k: int = None) -> List[SearchHit]:
        """
        Rerank the retrieved documents using the specified strategy.

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
                      strategy=self.strategy,
                      hits_count=len(hits),
                      k=k)

        # Truncate documents to max_tokens_per_doc
        truncated_hits = self._truncate_documents(hits)

        # Apply the appropriate reranking strategy
        if self.strategy == "pointwise":
            reranked_hits = self._pointwise_rerank(query, truncated_hits)
        elif self.strategy == "pairwise":
            reranked_hits = self._pairwise_rerank(query, truncated_hits)
        elif self.strategy == "listwise":
            reranked_hits = self._listwise_rerank(query, truncated_hits)
        elif self.strategy == "setwise":
            reranked_hits = self._setwise_rerank(query, truncated_hits, k)
        else:
            # This should never happen due to validation in __init__
            self.log.error("Unknown reranking strategy",
                           strategy=self.strategy)
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
        Build a prompt for the LLM based on the reranking strategy.

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
            if self.strategy == "pointwise":
                role_playing_text = "You are an expert document relevance judge. "
            elif self.strategy == "pairwise":
                role_playing_text = "You are an expert document comparison system. "
            elif self.strategy == "listwise":
                role_playing_text = "You are RankGPT, an expert document ranking system. "
            elif self.strategy == "setwise":
                role_playing_text = "You are an expert document selection system. "

        # Tone words component
        tone_text = ""
        if self.tone_words:
            tone_text = f"{self.tone_words} "

        # Task instruction and output type based on strategy
        if self.strategy == "pointwise":
            task_instruction = f"Determine if the passage answers the query: {query}"
            output_type = f"{tone_text}Answer with YES or NO only."
        elif self.strategy == "pairwise":
            task_instruction = f"Compare the two passages and determine which one is more relevant to the query: {query}"
            output_type = f"{tone_text}Answer with A or B only."
        elif self.strategy == "listwise":
            task_instruction = f"Rank the passages based on their relevance to the query: {query}"
            output_type = f"{tone_text}List the passage numbers in order of relevance, most relevant first. Format: [1] > [2] > [3] ..."
        elif self.strategy == "setwise":
            task_instruction = f"Select the most relevant passage for the query: {query}"
            output_type = f"{tone_text}Return only the number of the most relevant passage."
        else:
            # This should never happen due to validation in __init__
            task_instruction = f"Analyze the passages for the query: {query}"
            output_type = f"{tone_text}Provide your analysis."

        # Format documents with IDs
        if doc_ids is None:
            doc_ids = [str(i+1) for i in range(len(docs))]

        formatted_docs = []
        for i, (doc_id, doc) in enumerate(zip(doc_ids, docs)):
            formatted_docs.append(f"[{i+1}] {doc}")

        evidence_text = "\n\n".join(formatted_docs)

        # Assemble the prompt based on evidence_first flag
        if self.evidence_first:
            prompt = f"{role_playing_text}{evidence_text}\n\n{task_instruction}\n\n{output_type}"
        else:
            prompt = f"{role_playing_text}{task_instruction}\n\n{evidence_text}\n\n{output_type}"

        return prompt

    def _pointwise_rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """
        Rerank documents using pointwise strategy.
        Each document is evaluated individually for relevance.

        Args:
            query: The user query
            hits: List of retrieved documents

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing pointwise reranking", hits_count=len(hits))

        # Score each document individually
        scored_hits = []
        for hit in hits:
            prompt = self._build_prompt(query, [hit.metadata.text])
            response, _ = self.llm_client.complete_chat_once(prompt, None)

            # Parse response - looking for YES/NO
            response = response.strip().upper()
            # Score: 1.0 for YES, 0.0 for NO, 0.5 for anything else
            score = 1.0 if response == "YES" else (
                0.0 if response == "NO" else 0.5)

            scored_hits.append((hit, score))
            self.log.debug("Pointwise score", doc_id=hit.id,
                           score=score, response=response)

        # Sort by score (descending)
        scored_hits.sort(key=lambda x: x[1], reverse=True)

        # Return reranked hits
        return [hit for hit, _ in scored_hits]

    def _pairwise_rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """
        Rerank documents using pairwise strategy.
        Documents are compared in pairs to determine relative relevance.

        Args:
            query: The user query
            hits: List of retrieved documents

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing pairwise reranking", hits_count=len(hits))

        if len(hits) <= 1:
            return hits

        # Initialize scores
        scores = {hit.id: 0 for hit in hits}

        # Compare each pair of documents
        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                doc_a = hits[i]
                doc_b = hits[j]

                prompt = self._build_prompt(
                    query, [doc_a.metadata.text, doc_b.metadata.text], ["A", "B"])
                response, _ = self.llm_client.complete_chat_once(prompt, None)

                # Parse response - looking for A or B
                response = response.strip().upper()
                if response == "A":
                    scores[doc_a.id] += 1
                elif response == "B":
                    scores[doc_b.id] += 1
                else:
                    # If response is unclear, give half point to both
                    scores[doc_a.id] += 0.5
                    scores[doc_b.id] += 0.5

                self.log.debug("Pairwise comparison",
                               doc_a=doc_a.id,
                               doc_b=doc_b.id,
                               winner=response)

        # Sort hits by score
        reranked_hits = sorted(
            hits, key=lambda hit: scores[hit.id], reverse=True)

        # Create new instances with updated scores
        scored_hits = []
        for hit in reranked_hits:
            scored_hits.append(update_tuple(hit, score=scores[hit.id]))

        return scored_hits

    def _listwise_rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """
        Rerank documents using listwise strategy.
        All documents are ranked at once.

        Args:
            query: The user query
            hits: List of retrieved documents

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing listwise reranking", hits_count=len(hits))

        if len(hits) <= 1:
            return hits

        # Extract document texts
        docs = [hit.metadata.text for hit in hits]

        # Build prompt for listwise ranking
        prompt = self._build_prompt(query, docs)

        # Get ranking from LLM
        response, _ = self.llm_client.complete_chat_once(prompt, None)

        # Parse response - expected format: [1] > [2] > [3] ...
        # Extract numbers from response
        import re
        numbers = re.findall(r'\[(\d+)\]', response)

        # Convert to integers, filter out invalid indices
        try:
            indices = [int(num) - 1 for num in numbers if 0 <
                       int(num) <= len(hits)]
        except ValueError:
            self.log.warning(
                "Failed to parse listwise ranking response", response=response)
            return hits

        # Handle missing or duplicate indices
        seen_indices = set()
        ordered_indices = []
        for idx in indices:
            if idx not in seen_indices and 0 <= idx < len(hits):
                ordered_indices.append(idx)
                seen_indices.add(idx)

        # Add any missing indices at the end
        for i in range(len(hits)):
            if i not in seen_indices:
                ordered_indices.append(i)

        # Rerank hits based on the order
        reranked_hits = [hits[i] for i in ordered_indices]

        # Create new instances with updated scores based on rank
        scored_hits = []
        for i, hit in enumerate(reranked_hits):
            new_score = len(reranked_hits) - i
            scored_hits.append(update_tuple(hit, score=new_score))

        return scored_hits

    def _setwise_rerank(self, query: str, hits: List[SearchHit], k: int = None) -> List[SearchHit]:
        """
        Rerank documents using setwise strategy.
        Iteratively select the most relevant document from the set.

        Args:
            query: The user query
            hits: List of retrieved documents
            k: Number of documents to return (default: all)

        Returns:
            Reranked list of SearchHit objects
        """
        self.log.debug("Performing setwise reranking", hits_count=len(hits))

        if len(hits) <= 1:
            return hits

        remaining_hits = hits.copy()
        reranked_hits = []

        # Set default value for k if not provided
        if k is None:
            k = len(hits)

        # Iteratively select the most relevant document
        while remaining_hits:
            if len(remaining_hits) == 1:
                # Last document, just add it
                reranked_hits.append(remaining_hits[0])
                break
            
            # Early return if we've collected enough documents
            if len(reranked_hits) >= k:
                self.log.debug("Early return from setwise reranking", 
                               collected=len(reranked_hits), 
                               requested=k)
                break

            # Extract document texts
            docs = [hit.metadata.text for hit in remaining_hits]

            # Build prompt for setwise selection
            prompt = self._build_prompt(query, docs)

            # Get selection from LLM
            response, _ = self.llm_client.complete_chat_once(prompt, None)

            # Parse response - looking for a number
            import re
            match = re.search(r'\d+', response)
            if match:
                try:
                    selected_idx = int(match.group()) - 1
                    if 0 <= selected_idx < len(remaining_hits):
                        selected_hit = remaining_hits.pop(selected_idx)
                        reranked_hits.append(selected_hit)
                        self.log.debug("Selected document",
                                       doc_id=selected_hit.id,
                                       rank=len(reranked_hits))
                    else:
                        # Invalid index, select the first document
                        self.log.warning("Invalid index in setwise response",
                                         index=selected_idx,
                                         response=response)
                        selected_hit = remaining_hits.pop(0)
                        reranked_hits.append(selected_hit)
                except ValueError:
                    # Failed to parse index, select the first document
                    self.log.warning(
                        "Failed to parse setwise response", response=response)
                    selected_hit = remaining_hits.pop(0)
                    reranked_hits.append(selected_hit)
            else:
                # No number found, select the first document
                self.log.warning(
                    "No number found in setwise response", response=response)
                selected_hit = remaining_hits.pop(0)
                reranked_hits.append(selected_hit)

        # Create new instances with updated scores based on rank
        scored_hits = []
        for i, hit in enumerate(reranked_hits):
            new_score = len(reranked_hits) - i
            scored_hits.append(update_tuple(hit, score=new_score))

        return scored_hits

    def _setwise_rerank_rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """
        This is an prefix-stable speed optimization of the setwise reranking method.
        We can potentially get LLM inference engine to reuse 90% of the kv cache.

        Kun:
        
        - This method modifies the _setwise_rerank method to use a prefix-based approach where each prompt starts with the same prefix.
        - Instead of removing documents from the set, we'll keep all documents but mark the selected ones as already chosen.
        - Each new prompt will include all the documents (both selected and remaining), but in the end list out the selected documents in order.
        - The prompt will ask the LLM to select the next most relevant document from the remaining ones.
        """
        
        # TODO: implement this if we can get _setwise_rerank to work well
        return hits

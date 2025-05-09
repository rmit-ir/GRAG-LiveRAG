from typing import List

from services.indicies import QueryService, SearchHit
from utils.logging_utils import get_logger

logger = get_logger("doc_listing_utils")


def truncate_doc_listings(listings: List[List[SearchHit]], context_word_limit: int) -> List[SearchHit]:
    docs: List[SearchHit] = []
    docs_set = set()
    total_words = 0
    max_list_len = max([len(l) for l in listings])

    for i in range(max_list_len):
        for listing in listings:
            # for each listing, we take the i-th document
            if i < len(listing):
                doc = listing[i]
                doc_words = len(doc.metadata.text.split())
                # check if we've already had enough words
                if total_words + doc_words >= context_word_limit:
                    logger.debug("Word threshold reached",
                                 total_words=total_words,
                                 threshold=context_word_limit,
                                 docs_included=len(docs),
                                 docs_total=sum(len(l) for l in listings),
                                 listings_total=len(listings))
                    # early stop if we have enough words
                    return docs
                if doc.id not in docs_set:
                    total_words += doc_words
                    docs_set.add(doc.id)
                    docs.append(doc)
    # return all documents if we haven't reached the limit
    return docs


def truncate_docs(docs: List[SearchHit], words_threshold: int = 20000) -> List[SearchHit]:
    """
    Truncate a list of SearchHit documents based on a word count threshold.

    Args:
        docs: List of SearchHit objects
        words_threshold: Maximum number of words to include (default: 20000)

    Returns:
        Truncated list of SearchHit objects
    """
    if not docs:
        return []

    truncated_docs = []
    total_words = 0

    for doc in docs:
        # Count words in the document text
        doc_text = doc.metadata.text if hasattr(doc.metadata, "text") else ""
        word_count = len(doc_text.split())

        # Check if adding this document would exceed the threshold
        if total_words + word_count > words_threshold:
            logger.debug("Word threshold reached",
                         total_words=total_words,
                         threshold=words_threshold,
                         docs_included=len(truncated_docs),
                         docs_total=len(docs))
            break

        # Add document and update word count
        truncated_docs.append(doc)
        total_words += word_count

    logger.debug("Documents truncated",
                 original_count=len(docs),
                 truncated_count=len(truncated_docs),
                 total_words=total_words)
    return truncated_docs


if __name__ == "__main__":
    query_service = QueryService()
    docs1 = query_service.query_keywords("what is the capital of France")
    docs2 = query_service.query_keywords("what is the capital of China")
    final_docs = truncate_doc_listings([docs1, docs2], 800)

    for doc in final_docs:
        print("-" * 80)
        print(
            f"Document ID: {doc.id}, Text length: {len(doc.metadata.text.split())}, Text: \n{doc.metadata.text}")

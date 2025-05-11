from typing import List

from services.indicies import SearchHit
from services.live_rag_metadata import LiveRAGMetadata
from services.llms.mini_tgi_client import MiniTGIClient
from utils.doc_listing_utils import truncate_docs
from utils.logging_utils import get_logger
from utils.namedtuple_utils import update_tuple

log = get_logger("logits_reranker")

PROMPT_IDENTIFY_SOURCE = """<|system|>
You are a helpful assistant that determines if a question is related to a given document. Answer only with 'Yes' or 'No'.
<|user|>
Document: {doc_text}

Question: {question}

Is this question generated from this document (only answer 'Yes' or 'No')?
<|assistant|>
"""

PROMPT_CAN_ANSWER = """<|system|>
You are a helpful assistant that determines if a document contains information that helps answer a given question. Answer only with 'Yes' or 'No'.
<|user|>
Document: {doc_text}

Question: {question}

Does this document contain information that helps answer this question (only answer 'Yes' or 'No')?
<|assistant|>
"""


class LogitsReranker:
    def __init__(self, logits_llm: MiniTGIClient, prompt_version='can_answer'):
        self.logits_llm = logits_llm
        self.prompt_version = prompt_version

        self.tokens = ['Yes', 'No']
        self.goal_token = 'Yes'

        if prompt_version == 'identify_source':
            self.prompt_tpl = PROMPT_IDENTIFY_SOURCE
        elif prompt_version == 'can_answer':
            self.prompt_tpl = PROMPT_CAN_ANSWER
        else:
            log.error(
                "Invalid prompt version. Use 'identify_source' or 'can_answer'.")
            self.prompt_tpl = PROMPT_IDENTIFY_SOURCE

    def rerank(self, docs: List[SearchHit], question: str, yes_threshold: float = 0.5, k_docs: int = None, words_limit: int = None) -> List[SearchHit]:
        log.debug("Documents before reranking", documents=docs)
        id_doc_dict = {doc.id: doc for doc in docs}
        id_yes_prob = []
        tokens = self.tokens
        goal_token = self.goal_token

        for hit in docs:
            doc_text = hit.metadata.text.replace("\n", " ")
            prompt = self.prompt_tpl.format(
                doc_text=doc_text, question=question)
            logits = self.logits_llm.get_token_logits(prompt, tokens=tokens)
            log.debug("Logits for document", doc_id=hit.id, logits=logits)
            yes_raw_prob = logits['raw_probabilities'].get(goal_token, 0.0)
            id_yes_prob.append((hit.id, yes_raw_prob))
        log.debug("Logits for documents", logits=id_yes_prob)

        sorted_docs = sorted(id_yes_prob, key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, yes_raw_prob in sorted_docs:
            if doc_id in id_doc_dict:
                if yes_threshold is not None and yes_raw_prob < yes_threshold:
                    # If the yes probability is below the threshold, we stop adding documents
                    break
                doc = id_doc_dict[doc_id]
                doc = update_tuple(doc, score=yes_raw_prob)
                reranked_docs.append(doc)
        log.debug("Documents after reranking", documents=reranked_docs)

        final_docs = reranked_docs
        if words_limit:
            final_docs = truncate_docs(reranked_docs, words_limit)
        elif k_docs:
            final_docs = reranked_docs[:k_docs]
        log.info("Reranked docs",
                 docs_input=len(docs), docs_output=len(final_docs),
                 highest_score=final_docs[0].score if final_docs else None,
                 lowest_score=final_docs[-1].score if final_docs else None,)
        return final_docs


if __name__ == "__main__":
    from services.llms.mini_tgi_client import MiniTGIClient
    from services.indicies import SearchHit

    def to_hit(txt: str, doc_id: str, score: float) -> SearchHit:
        return SearchHit(id=doc_id, metadata=LiveRAGMetadata(text=txt, chunk_order=1.0, doc_id=doc_id, is_first_chunk=True, is_last_chunk=True, total_doc_chunks=5), score=score, retrieval_model='test')

    # Example usage
    logits_llm = MiniTGIClient()
    reranker = LogitsReranker(logits_llm=logits_llm)
    docs = [
        to_hit("Patty likes dogs", "1", 0.9),
        to_hit("The weather is nice today", "2", 0.9),
        to_hit("Paul likes cats", "3", 0.9),
        to_hit("Patty likes cats", "4", 0.9),
        to_hit("Paul hates dogs", "5", 0.9),
    ]
    question = "What does Patty like?"
    k = 10
    reranked_docs = reranker.rerank(docs=docs, question=question, k_docs=k)
    print("Reranked documents:")
    for doc in reranked_docs:
        print(
            f"Doc ID: {doc.id}, Score: {doc.score}, Text: {doc.metadata.text}")

# Context Recall Evaluator

This evaluator calculates the recall of retrieved documents compared to gold context documents. It measures how many of the retrieved chunks are also included in the gold context document IDs.

## Metrics

The evaluator calculates the following metrics:

- **Context Recall**: The proportion of gold context documents that were successfully retrieved, important
  - Formula: `|retrieved_docs ∩ gold_docs| / |gold_docs|`
  
- **Context Precision**: The proportion of retrieved documents that are in the gold context, not important for LiveRAG
  - Formula: `|retrieved_docs ∩ gold_docs| / |retrieved_docs|`
  
- **Context F1**: The harmonic mean of precision and recall, not important for LiveRAG
  - Formula: `2 * (precision * recall) / (precision + recall)`

- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10, important for fundamental RAG systems
  - Measures ranking quality, with higher scores indicating relevant documents are ranked higher

- **Retrieved Docs Count**: The number of unique documents retrieved
- **Gold Docs Count**: The number of documents in the gold context
- **Correct Docs Count**: The number of documents that were correctly retrieved

## Document ID Handling

The evaluator handles the different formats of document IDs between retrieved chunks and gold context:

- Retrieved chunks format: `doc-<urn:uuid:27c682dc-c97b-4c3e-b41c-999c65264a6d>::chunk-0`
- Gold context format: `['<urn:uuid:417fbc69-80f9-4b53-8b5c-1c9cdd2bb0e9>']`

The evaluator extracts the UUID part from the retrieved chunks to make them comparable with the gold context document IDs.

## Usage

evaluate.py interface:

```bash
uv run scripts/evaluate.py --evaluator evaluators.context_recall.evaluator.ContextRecall --results data/rag_results/dmds_fJ20pJnq9zcO1_BasicRAGSystem.tsv --reference data/generated_qa_pairs/dmds_fJ20pJnq9zcO1.n100.tsv;
```

Python interface:

```python
from evaluators.context_recall import ContextRecall
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair

# Initialize the evaluator
evaluator = ContextRecall()

# Evaluate RAG results against reference QA pairs
evaluation_result = evaluator.evaluate(rag_results, references)

# Access the metrics
print(f"Average Context Recall: {evaluation_result.metrics['avg_context_recall']}")
print(f"Average Context Precision: {evaluation_result.metrics['avg_context_precision']}")
print(f"Average Context F1: {evaluation_result.metrics['avg_context_f1']}")
print(f"NDCG@10: {evaluation_result.metrics['ndcg_10']}")
```

## Interpretation

- Higher recall indicates that the retrieval system is finding most of the relevant documents
- Higher precision indicates that the retrieval system is not including many irrelevant documents, not needed for LiveRAG (gold document is only 1 or 2)
- Higher F1 indicates a good balance between recall and precision, not needed for LiveRAG (gold document is only 1 or 2)
- Higher NDCG@10 indicates better ranking quality of retrieved documents. This is important for fundamental RAG systems like bm25/embedding based. But for the final system this is not important, all 10 will be used for answer generation and judgement.

For LiveRAG, the most important metric is recall.

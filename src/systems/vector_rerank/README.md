# Vector Rerank System

A RAG system implementation that uses vector-based (embedding) search to retrieve relevant documents for a given question, followed by LLM-based reranking to improve retrieval quality. The system leverages both semantic similarity and advanced LLM-based reranking strategies to find the most relevant context for answering questions.

## Reranking Implementation

The reranking component is based on the research paper "An Investigation of Prompt Variations for Zero-shot LLM-based Rankers" (Shuoqi Sun et al., 2025). The system implements four different reranking strategies:

1. **Pointwise**: Evaluates each document individually for relevance to the query
2. **Pairwise**: Compares pairs of documents to determine relative relevance
3. **Listwise**: Ranks all documents at once based on their relevance to the query
4. **Setwise**: Iteratively selects the most relevant document from the set

The reranker also implements various prompt engineering techniques identified in the paper as important factors for reranking effectiveness:

- **Role Playing**: Optional role definition to help the LLM "impersonate" an expert in document relevance
- **Tone Words**: Optional words that express a positive, negative, or neutral connotation
- **Evidence Ordering**: Control over whether evidence (documents) is presented before or after instructions
- **Evidence Position**: Control over whether evidence is placed at the beginning or end of the prompt

## How to Run

```bash
uv run scripts/run.py --system systems.vector_rerank.vector_rerank.VectorRerank --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.vector_rerank.vector_rerank.VectorRerank --help
```

## Configuration Options

The VectorRerank system accepts the following configuration parameters:

- `llm_client`: LLM client to use (default: "ai71_client")
- `reranker_strategy`: Reranking strategy to use - "pointwise", "pairwise", "listwise", or "setwise" (default: "setwise")
- `reranker_k`: Number of documents to return after reranking (default: 5)
- `role_playing`: Whether to use role playing in reranker prompts (default: True)
- `tone_words`: Tone words to use in reranker prompts (default: "Please")
- `evidence_first`: Whether to put evidence before instructions in reranker prompts (default: True)

Example with custom configuration:

```bash
uv run scripts/run.py --system systems.vector_rerank.vector_rerank.VectorRerank --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv --reranker_strategy listwise --reranker_k 3 --role_playing False
```

## Performance Considerations

According to the research paper, different reranking strategies may perform better depending on the dataset and LLM backbone:

- **Setwise** and **Pairwise** methods generally deliver the best results across different datasets and LLM backbones
- **Pointwise** can be competitive with specific prompt configurations
- **Listwise** shows mixed results depending on the dataset

The paper also found that prompt variations can significantly impact reranking effectiveness, which is why this implementation allows configuring various prompt components.

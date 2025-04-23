# Vector RAG System

A RAG system implementation that uses vector-based (embedding) search to retrieve relevant documents for a given question. The system leverages semantic similarity rather than keyword matching to find the most relevant context for answering questions.

## How to Run

```bash
uv run scripts/run.py --system systems.vector_rag.vector_rag.VectorRAG --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.vector_rag.vector_rag.VectorRAG --help
```

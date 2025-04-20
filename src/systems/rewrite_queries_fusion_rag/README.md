# Rewrite Queries Fusion RAG System

An advanced RAG implementation that rewrites the original question to generate multiple search queries and uses fusion search to improve retrieval quality. The system generates a few alternative queries (configurable), uses fusion search (combining embedding and keyword search) for each query, and applies reciprocal rank fusion to combine results from different queries.

## How to Run

```bash
uv run scripts/run.py --system systems.rewrite_queries_fusion_rag.rag.FusionRAGSystem --input data/generated_qa_pairs/datamorgana_dataset_20250414_181830.n2.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.rewrite_queries_fusion_rag.rag.FusionRAGSystem --help
```

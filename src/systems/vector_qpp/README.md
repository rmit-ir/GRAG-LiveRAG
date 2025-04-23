# Vector QPP System

A RAG system implementation that uses vector search with Query Performance Prediction (QPP) to assess the quality of search results. The system retrieves documents using embedding-based search and calculates QPP metrics (entropy, standard deviation, mean) to evaluate the confidence in the retrieved results.

## How to Run

```bash
uv run scripts/run.py --system systems.vector_qpp.vector_qpp.VectorQPP --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.vector_qpp.vector_qpp.VectorQPP --help
```

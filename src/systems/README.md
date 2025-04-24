# LiveRAG Systems

This directory contains RAG (Retrieval-Augmented Generation) system implementations.

## Overview

RAG systems retrieve relevant documents for a given question and generate answers using LLMs. The workflow:

1. Process a question using a RAG system
2. Retrieve relevant documents from the index
3. Generate an answer using an LLM with the retrieved context
4. Return a structured result with the answer and metadata

## Creating a New System

To create a new RAG system:

1. Create a directory under `src/systems/` for your system
2. Implement the [`RAGSystemInterface`](./rag_system_interface.py) abstract base class
3. See [`basic_rag/basic_rag_system.py`](./basic_rag/basic_rag_system.py) for a complete example

## Running a System

For a complete overview of the end-to-end workflow, see [The LiveRAG Workflow](../../README.md#the-liverag-workflow) in the main README.

```bash
uv run scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem \
  --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv \
  --num-threads 5
```

This generates results in `data/rag_results/dmds_JK09SKjyanxs1_BasicRAGSystem.tsv`

For system-specific parameters:

```bash
uv run scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --help
```

Notice that the system's constructor arguments are automatically made available as command line arguments.

## System Interface

All RAG systems must implement the [`RAGSystemInterface`](./rag_system_interface.py) abstract base class:

```python
def process_question(self, question: str, qid: Optional[str] = None) -> RAGResult:
    """
    Process a question and generate an answer using RAG.
    
    Args:
        question: The user's question
        qid: Optional query ID, will be populated in RAGResult
        
    Returns:
        A RAGResult containing the answer and metadata
    """
    pass
```

## Best Practices

1. **Use the logging utility**: `from utils.logging_utils import get_logger`
2. **Track performance metrics** in the `RAGResult` (e.g., `total_time_ms`)
3. **Be creative!**

## Interesting Systems

- **[BasicRAGSystem](./basic_rag/basic_rag_system.py)**: Simple implementation using keyword search and LLM generation
  - Path: `systems.basic_rag.basic_rag_system.BasicRAGSystem`

- **[FusionRAGSystem](./rewrite_queries_fusion_rag/rag.py)**: Slightly complicated (not necessarily better) implementation with query rewriting and fusion search
  - Path: `systems.rewrite_queries_fusion_rag.rag.FusionRAGSystem`

- **[QPPFusionSystem](./qpp_fusion/qpp_fusion_rag.py)**: Advanced implementation that uses Query Performance Prediction (QPP) to select the most effective queries from multiple generated queries and applies fusion search
  - Path: `systems.qpp_fusion.qpp_fusion_rag.QPPFusionSystem`

- **[VectorQPP](./vector_qpp/vector_qpp.py)**: Vector search implementation with Query Performance Prediction to assess the quality of search results
  - Path: `systems.vector_qpp.vector_qpp.VectorQPP`

## Development Logs

| Date | Name | Avg Relevance Score | Avg Faithfulness Score |
|------|------|--------------------:|----------------------:|
| 2025-04-22 | rewrite_queries_fusion_rag | 1.57 | 0.57 |
| 2025-04-22 | rewrite_queries_fusion_rag | 1.50 | 0.66 |
| 2025-04-24 | basic_rag | 1.51 | 0.43 |
| 2025-04-24 | qpp_fusion max_documents=10, max_query_documents=10 | 1.57 | 0.64 |
| 2025-04-24 | qpp_fusion max_documents=5, max_query_documents=200 | 1.34 | 0.63 |
| 2025-04-24 | qpp_fusion max_documents=10, max_query_documents=200 | 1.36 | 0.56 |

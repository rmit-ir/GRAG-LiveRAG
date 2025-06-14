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
3. See [`grag/grag.py`](./grag/grag.py) for a complete example

## Running a System

You can run a system using the class name:

```bash
uv run scripts/run.py --system GRAG \
  --live \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl \
  --num-threads 5
```

This generates results in `data/rag_results/` folder.

For system-specific parameters:

```bash
uv run scripts/run.py --system GRAG --help
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

## Available Systems

- **[GRAG](./grag/grag.py)**: Advanced RAG implementation with query expansion, fusion search, and logits reranking
  - Path: `systems.grag.grag.GRAG`

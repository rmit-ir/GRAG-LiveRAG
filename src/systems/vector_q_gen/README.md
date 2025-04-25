# Vector Q-Gen RAG System

This RAG system decomposes queries into components, generates queries for each component, and performs fusion to retrieve relevant documents.

## Overview

The Vector Q-Gen RAG system works by:

1. Decomposing a complex question into multiple components
2. Generating optimized queries for each component
3. Retrieving documents for each query using embedding search
4. Performing fusion within each component to select the most relevant documents
5. Taking the top documents from each component (with a total document limit)
6. Generating an answer based on the selected documents

## Usage

```bash
uv run scripts/run.py --system systems.vector_q_gen.vector_q_gen.VectorQGen --help
```

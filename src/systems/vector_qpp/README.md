# Basic RAG System

A simple implementation of Retrieval-Augmented Generation (RAG) that uses keyword search to retrieve relevant documents and an LLM to generate answers. The system retrieves up to 10 documents for input question and uses the AI71 client with Falcon 3 10B Instruct model.

## How to Run

```bash
uv run scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --input data/generated_qa_pairs/datamorgana_dataset_20250414_181830.n2.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --help
```

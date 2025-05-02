# QPPPreSystem: Query Performance Prediction for RAG
## Overview
QPPPreSystem is a Retrieval-Augmented Generation (RAG) system that uses query performance prediction for pre-retrieval optimization. It focuses on sparse (keyword-based) retrieval to improve answer quality by selecting the most effective query variation.

## Key Features
- Query Performance Prediction (QPP): Evaluates query effectiveness using entropy, standard deviation, and mean-based metrics
- Query Generation: Creates optimized keyword search queries from the original question
- Adaptive Query Selection: Selects the most effective query based on QPP confidence scores

## How to Run

```bash
uv run scripts/run.py --system QPPPreSystem --input data/generated_qa_pairs/datamorgana_dataset_20250414_181830.n2.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system QPPPreSystem --help
```
# LiveRAG Evaluators

This directory contains evaluators for assessing RAG (Retrieval-Augmented Generation) system performance.

## Overview

Evaluators compare RAG system outputs against reference answers and calculate performance metrics. The evaluation workflow:

1. Run a RAG system on questions using [`scripts/run.py`](../../scripts/run.py)
2. Evaluate the results against reference answers using [`scripts/evaluate.py`](../../scripts/evaluate.py)
3. Analyze the evaluation results

## Running Evaluations

### Step 1: Run a RAG system

```bash
uv run scripts/run.py --system GRAG \
  --live \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl \
  --num-threads 5
```

This generates results in `data/rag_results/` folder.

### Step 2: Evaluate the results

You can evaluate results using either the full path or just the class name:

```bash
uv run scripts/evaluate.py \
  --evaluator ContextRecall \
  --results data/rag_results/your_results.tsv \
  --reference data/generated_qa_pairs/your_reference.tsv
```

Both commands generate evaluation results in `data/evaluation_results/` folder:

- `your_results.eval[timestamp].ContextRecall.aggregated.tsv`
- `your_results.eval[timestamp].ContextRecall.rows.tsv`

For evaluator-specific parameters:

```bash
uv run scripts/evaluate.py --evaluator ContextRecall --help
```

The `evaluate.py` script automatically extracts parameters from your evaluator's `__init__` method and makes them available as command-line arguments. For example, if your evaluator has parameters like `normalize=True` in its constructor, you can pass `--normalize` or `--no-normalize` directly on the command line.

## Creating a New Evaluator

To create a new evaluator:

1. Create a directory under `src/evaluators/` for your evaluator
2. Implement the [`EvaluatorInterface`](./evaluator_interface.py) abstract base class
3. See [`context_recall/evaluator.py`](./context_recall/evaluator.py) for a complete example

## Evaluator Interface

All evaluators must implement the [`EvaluatorInterface`](./evaluator_interface.py) abstract base class:

```python
def evaluate(self, rag_results: List[RAGResult], references: List[QAPair]) -> EvaluationResult:
    """
    Evaluate a list of RAG results against a list of reference QA pairs.
    
    Args:
        rag_results: List of RAG results to evaluate
        references: List of reference QA pairs to compare against
        
    Returns:
        An EvaluationResult containing evaluation metrics and optionally row-level results
    """
    pass
```

## Best Practices

1. **Focus on a single metric or related set of metrics**
2. **Handle edge cases** (empty answers, missing references)
3. **Use the logging utility**: `from utils.logging_utils import get_logger`
4. **Refer to existing evaluators** for implementation patterns
5. **Test often** write a `__main__` block to run your evaluator independently

## Available Evaluators

- **[ContextRecall](./context_recall/evaluator.py)**: Calculates recall, precision, F1, and NDCG metrics for retrieved context documents
  - Path: `evaluators.context_recall.evaluator.ContextRecall`

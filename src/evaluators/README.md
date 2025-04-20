# LiveRAG Evaluators

This directory contains evaluators for assessing RAG (Retrieval-Augmented Generation) system performance.

## Overview

Evaluators compare RAG system outputs against reference answers generated from DataMorgana and calculate performance metrics. The evaluation workflow:

1. Run a RAG system on questions using [`scripts/run.py`](../scripts/run.py)
2. Evaluate the results against reference answers using [`scripts/evaluate.py`](../scripts/evaluate.py)
3. Analyze the evaluation results

## Creating a New Evaluator

To create a new evaluator:

1. Create a directory under `src/evaluators/` for your evaluator
2. Implement the [`EvaluatorInterface`](./evaluator_interface.py) abstract base class
3. See [`basic_evaluator/edit_distance_evaluator.py`](./basic_evaluator/edit_distance_evaluator.py) for a complete example

## Running Evaluations

For a complete overview of the end-to-end workflow, see [The LiveRAG Workflow](../../README.md#the-liverag-workflow) in the main README.

### Step 1: Run a RAG system

```bash
uv run scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem \
  --input data/generated_qa_pairs/dmds_4p3PUk5HORIw.n5.tsv \
  --num-threads 5
```

This generates results in `data/rag_results/dmds_4p3PUk5HORIw_BasicRAGSystem.tsv`

### Step 2: Evaluate the results

```bash
uv run scripts/evaluate.py \
  --evaluator evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator \
  --results data/rag_results/dmds_4p3PUk5HORIw_BasicRAGSystem.tsv \
  --reference data/generated_qa_pairs/dmds_4p3PUk5HORIw.n5.tsv
```

This generates:

- `data/evaluation_results/dmds_4p3PUk5HORIw_BasicRAGSystem.EditDistanceEvaluator.evaluation.aggregated.tsv`
- `data/evaluation_results/dmds_4p3PUk5HORIw_BasicRAGSystem.EditDistanceEvaluator.evaluation.rows.tsv`

For evaluator-specific parameters:

```bash
uv run scripts/evaluate.py --evaluator evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator --help
```

As you might've noticed from running `--help`, the `evaluate.py` script automatically extracts parameters from your evaluator's `__init__` method and makes them available as command-line arguments. For example, if your evaluator has parameters like `normalize=True` in its constructor, you can pass `--normalize` or `--no-normalize` directly on the command line. This means you don't need to modify the evaluation script when creating new evaluators with custom parameters.

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

- **[EditDistanceEvaluator](./basic_evaluator/edit_distance_evaluator.py)**: Calculates the Levenshtein distance between generated and reference answers
  - Path: `evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator`

## Extending the System

Consider implementing evaluators for:

- Semantic similarity using embeddings
- Using Claude Sonnet 3.5 just like LiveRAG hosts will do
- ROUGE or BLEU scores for text generation quality
- Factual consistency metrics
- Domain-specific evaluation metrics
- More evaluators mentioned in [Linear Issue #RMI-25](https://linear.app/rmit-liverag-2025/issue/RMI-25/develop-an-llm-based-evaluator-for-rag-system-assessment)

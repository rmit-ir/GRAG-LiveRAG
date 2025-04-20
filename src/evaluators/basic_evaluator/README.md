# Basic Evaluator

A simple implementation of RAG evaluation that uses edit distance (Levenshtein distance) to compare generated answers with reference answers. The evaluator calculates how many single-character edits (insertions, deletions, or substitutions) are needed to transform one string into another.

## EditDistanceEvaluator

The `EditDistanceEvaluator` measures the similarity between generated answers and reference answers using the Levenshtein distance algorithm. A lower edit distance indicates higher similarity between the strings.

Key features:

- Calculates raw edit distance between strings
- Optional normalization by the length of the longer string (results in a value between 0 and 1)
- Provides both individual and aggregated metrics (average, min, max, median)

## How to Run

```bash
uv run scripts/evaluate.py \
  --evaluator evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator \
  --results data/rag_results/your_results_file.tsv \
  --reference data/generated_qa_pairs/your_reference_file.tsv
```

## Output

The evaluator generates two output files:

1. Aggregated metrics: `data/evaluation_results/{dataset}_{system}.EditDistanceEvaluator.evaluation.aggregated.tsv`
2. Row-level results: `data/evaluation_results/{dataset}_{system}.EditDistanceEvaluator.evaluation.rows.tsv`

### Example Output

#### Aggregated Metrics (aggregated.tsv)

```tsv
evaluator_name	sample_count	system_name	timestamp	is_aggregated	avg_edit_distance	min_edit_distance	max_edit_distance	median_edit_distance	avg_normalized_distance	min_normalized_distance	max_normalized_distance	median_normalized_distance	processing_time_ms
EditDistanceEvaluator	5	BasicRAGSystem	2025-04-20T23:37:01.056758	True	707.8	419	939	792	0.7284345302950931	0.6925619834710743	0.7483552631578947	0.7434679334916865	0.7562637329101562
```

#### Row-level Results (rows.tsv)

```tsv
qid	evaluator_name	is_aggregated	edit_distance	normalized_distance
1	EditDistanceEvaluator	False	792	0.7141568981064021
2	EditDistanceEvaluator	False	934	0.7436305732484076
3	EditDistanceEvaluator	False	939	0.7434679334916865
4	EditDistanceEvaluator	False	419	0.6925619834710743
5	EditDistanceEvaluator	False	455	0.7483552631578947
```

### Metrics Provided

- `edit_distance`: Raw edit distance between strings
- `normalized_distance`: Edit distance normalized by the length of the longer string (if normalization is enabled)
- `avg_edit_distance`: Average edit distance across all evaluated pairs
- `min_edit_distance`: Minimum edit distance found
- `max_edit_distance`: Maximum edit distance found
- `median_edit_distance`: Median edit distance
- Similar metrics for normalized distances when normalization is enabled

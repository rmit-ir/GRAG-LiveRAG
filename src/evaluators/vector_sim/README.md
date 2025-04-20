# Vector Similarity Evaluator

A semantic evaluation approach for RAG systems that uses cosine similarity between embeddings to compare generated answers with reference answers. The evaluator measures the semantic similarity between texts rather than exact character matches.

## VectorSimEvaluator

The `VectorSimEvaluator` measures the semantic similarity between generated answers and reference answers by calculating the cosine similarity of their embeddings. A higher cosine similarity indicates higher semantic similarity between the texts.

## How to Run

```bash
uv run scripts/evaluate.py \
  --evaluator evaluators.vector_sim.evaluator.VectorSimEvaluator \
  --results data/rag_results/your_results_file.tsv \
  --reference data/generated_qa_pairs/your_reference_file.tsv
```

## Output

The evaluator generates two output files:

1. Aggregated metrics: `data/evaluation_results/{dataset}_{system}.VectorSimEvaluator.evaluation.aggregated.tsv`
2. Row-level results: `data/evaluation_results/{dataset}_{system}.VectorSimEvaluator.evaluation.rows.tsv`

### Example Output

#### Aggregated Metrics (aggregated.tsv)

```tsv
evaluator_name	sample_count	system_name	timestamp	is_aggregated	avg_cosine_similarity	min_cosine_similarity	max_cosine_similarity	median_cosine_similarity	processing_time_ms
VectorSimEvaluator	5	BasicRAGSystem	2025-04-21T01:18:41.029768	True	0.9045914361877934	0.8255572628656083	0.9462781220966373	0.9128268172558995	3282.4790477752686
```

#### Row-level Results (rows.tsv)

```tsv
qid	evaluator_name	is_aggregated	cosine_similarity
1	VectorSimEvaluator	False	0.9462781220966373
2	VectorSimEvaluator	False	0.9128268172558995
3	VectorSimEvaluator	False	0.9351586571243274
4	VectorSimEvaluator	False	0.9031363215964942
5	VectorSimEvaluator	False	0.8255572628656083
```

### Metrics Provided

- `cosine_similarity`: Cosine similarity between the embeddings of generated and reference answers
- `avg_cosine_similarity`: Average cosine similarity across all evaluated pairs
- `min_cosine_similarity`: Minimum cosine similarity found
- `max_cosine_similarity`: Maximum cosine similarity found
- `median_cosine_similarity`: Median cosine similarity

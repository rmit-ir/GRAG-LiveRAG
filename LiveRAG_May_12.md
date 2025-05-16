# Live Day Execution Summary

LiveRAG repo commit: [9152c560c9ef3e792a94c828c2f84a5923344f01](https://github.com/rmit-ir/LiveRAG/tree/9152c560c9ef3e792a94c828c2f84a5923344f01)

## Runs

### Config 1

Base config, naive answer generation prompt, no query expansion, no HyDE.

```bash
uv run scripts/run.py --system AnovaRAGLite \
  --live \
  --num-threads 20 \
  --llm_client ec2_llm \
  --query_expansion_mode none \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 10000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl
```

### Config 2

Same as config 1 but using full query decomposition and query variants generation, longer context allowance.

```bash
uv run scripts/run.py --system AnovaRAGLite \
  --live \
  --num-threads 20 \
  --llm_client ec2_llm \
  --query_expansion_mode decomposition \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 15000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/questions.jsonl
```

### Config 3

Using HyDE to generate an extra query for retrieval.

```bash
uv run scripts/run.py --system AnovaRAGLite \
  --live \
  --num-threads 20 \
  --llm_client ec2_llm \
  --query_expansion_mode none \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --enable_hyde \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 10000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl
```

### Config 4

[unsubmitted] similar to config 3 but using query decomposition and variants generation. Answers longer than 300 words will be condensed.

```bash
uv run scripts/run.py --system AnovaRAGLite \
  --live \
  --num-threads 20 \
  --llm_client ec2_llm \
  --query_expansion_mode variants \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --enable_hyde \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 10000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl
```

## Reproducing Results

Clone and checkout the live day commit

```bash
git checkout Challenge-Day
```

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .
```

Launch the logits server(preferably a GPU instance for faster speed)

```bash
uv run scripts/aws/apps/mini_tgi/llm_server.py --port 8977
```

Configure AI71 credentials

```bash
cp .env.example .env
# edit AI71_API_KEY= value
```

Run a config using commands from **Runs** chapter, but without `--llm_client` parameter (it defaults to ai71).
For example, to run config 3 (which is the final selected run):

```bash
uv run scripts/run.py --system AnovaRAGLite \
  --live \
  --num-threads 20 \
  --query_expansion_mode none \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --enable_hyde \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 10000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl
```

After it finishes, you can find the results in `data/rag_results/` folder.

Note

1. If you hit AI71 rate limits, you can reduce `--num-threads`.
2. By default run.py will connect to logits server at <http://localhost:8977>, if you launch it elsewhere, you need to port forward it to localhost.

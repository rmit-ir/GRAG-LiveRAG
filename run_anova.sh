#!/bin/bash

# Configuration
k_queries=10
sanitize_query=true
first_step_ranker="bm25+embedding_model"
num_first_retrieved_documents=4

# File paths
input="data/generated_qa_pairs/dmds_2_05012333.tsv"
output_dir="data/anova_result/${k_queries}_${sanitize_query}_${first_step_ranker}_${num_first_retrieved_documents}.tsv"

# Common arguments
common_args="--llm_client ai71 --system AnovaRAG --input $input --output-dir $output_dir --num-threads 20 --k_queries=$k_queries --first_step_ranker $first_step_ranker --num_first_retrieved_documents $num_first_retrieved_documents"

# Run command with appropriate flags
if [ "$sanitize_query" = true ]; then
    uv run scripts/run.py $common_args --sanitize_query
else
    uv run scripts/run.py $common_args
fi

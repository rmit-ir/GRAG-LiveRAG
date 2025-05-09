#!/bin/bash

# Configuration
original_question_inlcuded=false
k_queries=5
sanitize_query=true
qpp='no_qpp'
first_step_ranker="keywords+embedding_model"
num_first_retrieved_documents=3
fusion_method="concatenation"
reranker="no_reranker"
num_reranked_documents=10 # (optional) Number of documents returned from rerank

# File paths
input="data/generated_qa_pairs/dmds_2_05012333.tsv"
# output_dir is set dynamically based on the parameters
output_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${sanitize_query}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}"

# Common arguments
common_args="--llm_client ai71 --system AnovaRAG --input $input --output-dir $output_dir --num-threads 20 --original_question_inlcuded=$original_question_inlcuded --k_queries=$k_queries --sanitize_query=$sanitize_query --qpp=$qpp --num_first_retrieved_documents $num_first_retrieved_documents --first_step_ranker $first_step_ranker --fusion_method $fusion_method --reranker $reranker"

# Run command
uv run scripts/run.py $common_args
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

# File prefix, used for aligning run and eval
reference_file="dmds_2_05012333"

# File paths
input="data/generated_qa_pairs/${reference_file}.tsv"

# Loop through prompt levels
for query_gen_prompt_level in "naive" "medium" ; do
    for rag_prompt_level in "naive" "medium" ; do
        # output_dir is set dynamically based on the parameters
        output_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${sanitize_query}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"

        # Common arguments
        common_args="--llm_client ai71 --system AnovaRAG --input $input --output-dir $output_dir --num-threads 20 --original_question_inlcuded=$original_question_inlcuded --k_queries=$k_queries --sanitize_query=$sanitize_query --qpp=$qpp --num_first_retrieved_documents $num_first_retrieved_documents --first_step_ranker $first_step_ranker --fusion_method $fusion_method --reranker $reranker --rag_prompt_level $rag_prompt_level --query_gen_prompt_level $query_gen_prompt_level"

        # Run command
        echo "Running with query_gen_prompt_level=$query_gen_prompt_level and rag_prompt_level=$rag_prompt_level"
        uv run scripts/run.py $common_args
    done
done
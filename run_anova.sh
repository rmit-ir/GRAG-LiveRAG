#!/bin/bash

# Configuration
original_question_inlcuded=false
k_queries=5
qpp='no'
first_step_ranker="both"
num_first_retrieved_documents=5
fusion_method="concat"
reranker="no"
num_reranked_documents=5 # (optional) Number of documents returned from rerank

# File prefix, used for aligning run and eval
reference_file="dmds_2_05012333"

# File paths
input="data/generated_qa_pairs/${reference_file}.tsv"

# Loop through prompt levels
for query_gen_prompt_level in "naive" "medium" ; do
    for rag_prompt_level in "naive" "medium" ; do
        # output_dir is set dynamically based on the parameters
        output_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"

        # Common arguments
        common_args="--llm_client ai71 --system AnovaRAG --input $input --output-dir $output_dir --num-threads 20 --original_question_inlcuded=$original_question_inlcuded --k_queries=$k_queries --qpp=$qpp --num_first_retrieved_documents $num_first_retrieved_documents --first_step_ranker $first_step_ranker --fusion_method $fusion_method --reranker $reranker --rag_prompt_level $rag_prompt_level --query_gen_prompt_level $query_gen_prompt_level"

        # Run command
        echo "Running with query_gen_prompt_level=$query_gen_prompt_level and rag_prompt_level=$rag_prompt_level"
        uv run scripts/run.py $common_args
    done
done
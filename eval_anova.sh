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
num_reranked_documents=10

# File prefix, used for aligning run and eval
reference_file="dmds_2_05012333"

# Loop through prompt levels
for query_gen_prompt_level in "naive" "medium" "advanced"; do
    for rag_prompt_level in "naive" "medium" "advanced"; do
        # Construct the base directory path
        base_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${sanitize_query}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"
        
        # Find the most recent result file
        results_path=$(ls -t "${base_dir}"/${reference_file}.run*.AnovaRAG.tsv 2>/dev/null | head -n1)
        
        if [ -z "$results_path" ]; then
            echo "No result file found in ${base_dir}, skipping..."
            continue
        fi
        
        output_dir="data/evaluation_results/${original_question_inlcuded}_${k_queries}_${sanitize_query}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"
        
        echo "Evaluating with query_gen_prompt_level=$query_gen_prompt_level and rag_prompt_level=$rag_prompt_level"
        echo "Using result file: $results_path"
        
        uv run scripts/evaluate.py \
            --evaluator LLMEvaluator \
            --results "$results_path" \
            --reference data/generated_qa_pairs/${reference_file}.tsv \
            --output-dir "$output_dir"
    done
done

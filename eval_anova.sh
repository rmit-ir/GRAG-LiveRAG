#!/bin/bash

# Configuration
original_question_inlcuded=false # false, true
k_queries=5 # 3, 5, 8
qpp='no' # no
first_step_ranker="both" # both, keywords, embedding
num_first_retrieved_documents=5 #5, 10, 15
fusion_method="concat" # concatenation, weighted_sum
reranker="no" # no, logits
num_reranked_documents=5 # 5, 10, 15

# File prefix, used for aligning run and eval
reference_file="dmds_2_05012333"

# Loop through prompt levels
for query_gen_prompt_level in "naive" "medium" "advanced"; do
    for rag_prompt_level in "naive" "medium" "advanced"; do
        # Construct the base directory path
        base_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"
        
        # Find the most recent result file
        results_path=$(ls -t "${base_dir}"/${reference_file}.run*.AnovaRAG.tsv 2>/dev/null | head -n1)
        
        if [ -z "$results_path" ]; then
            echo "No result file found in ${base_dir}, skipping..."
            continue
        fi
        
        output_dir="data/evaluation_results/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${rag_prompt_level}"
        
        echo "Evaluating with query_gen_prompt_level=$query_gen_prompt_level and rag_prompt_level=$rag_prompt_level"
        echo "Using result file: $results_path"
        
        uv run scripts/evaluate.py \
            --evaluator LLMEvaluator \
            --results "$results_path" \
            --reference data/generated_qa_pairs/${reference_file}.tsv \
            --output-dir "$output_dir"
    done
done

#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")/../../../"

# File prefix, used for aligning run and eval
reference_file="dmds_500_hard_sampled_15"

# # Configuration lists
# QUERY_GEN_PROMPT_LEVELS=("naive" "medium")
# RAG_PROMPT_LEVELS=("naive" "medium" "advanced")
# ORIGINAL_QUESTION_INCLUDED_VALUES=("false" "true")
# K_QUERIES_VALUES=(4 5 8)
# QPP_VALUES=("no")
# FIRST_STEP_RANKER_VALUES=("both" "keywords" "embedding")
# NUM_FIRST_RETRIEVED_DOCUMENTS_VALUES=(5 8)
# FUSION_METHOD_VALUES=("concat")
# RERANKER_VALUES=("no" "logits")
# NUM_RERANKED_DOCUMENTS_VALUES_NO=(0)
# NUM_RERANKED_DOCUMENTS_VALUES_LOGITS=(10 15 20)

# Configuration lists
QUERY_GEN_PROMPT_LEVELS=("naive" "medium")
RAG_PROMPT_LEVELS=("naive" "medium")
ORIGINAL_QUESTION_INCLUDED_VALUES=("false" "true")
K_QUERIES_VALUES=(5 8)
QPP_VALUES=("no")
FIRST_STEP_RANKER_VALUES=("both" "embedding")
NUM_FIRST_RETRIEVED_DOCUMENTS_VALUES=(5 8)
FUSION_METHOD_VALUES=("concat")
RERANKER_VALUES=("no" "logits")
NUM_RERANKED_DOCUMENTS_VALUES_NO=(0)
NUM_RERANKED_DOCUMENTS_VALUES_LOGITS=(10 15)

# Add trap handler for Ctrl+C
stop_loops=false
trap 'stop_loops=true; echo -e "\nStopping after current iteration..."' SIGINT

# Loop through all parameter combinations
for query_gen_prompt_level in "${QUERY_GEN_PROMPT_LEVELS[@]}"; do
    if [ "$stop_loops" = true ]; then break; fi
    for rag_prompt_level in "${RAG_PROMPT_LEVELS[@]}"; do
        if [ "$stop_loops" = true ]; then break; fi
        for original_question_inlcuded in "${ORIGINAL_QUESTION_INCLUDED_VALUES[@]}"; do
            if [ "$stop_loops" = true ]; then break; fi
            for k_queries in "${K_QUERIES_VALUES[@]}"; do
                if [ "$stop_loops" = true ]; then break; fi
                for qpp in "${QPP_VALUES[@]}"; do
                    if [ "$stop_loops" = true ]; then break; fi
                    for first_step_ranker in "${FIRST_STEP_RANKER_VALUES[@]}"; do
                        if [ "$stop_loops" = true ]; then break; fi
                        for num_first_retrieved_documents in "${NUM_FIRST_RETRIEVED_DOCUMENTS_VALUES[@]}"; do
                            if [ "$stop_loops" = true ]; then break; fi
                            for fusion_method in "${FUSION_METHOD_VALUES[@]}"; do
                                if [ "$stop_loops" = true ]; then break; fi
                                for reranker in "${RERANKER_VALUES[@]}"; do
                                    if [ "$stop_loops" = true ]; then break; fi
                                    
                                    # Set num_reranked_documents based on reranker
                                    if [ "$reranker" = "no" ]; then
                                        num_reranked_documents_values=("${NUM_RERANKED_DOCUMENTS_VALUES_NO[@]}")
                                    else
                                        num_reranked_documents_values=("${NUM_RERANKED_DOCUMENTS_VALUES_LOGITS[@]}")
                                    fi
                                    
                                    for num_reranked_documents in "${num_reranked_documents_values[@]}"; do
                                        if [ "$stop_loops" = true ]; then break; fi
                                        base_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${num_reranked_documents}_${rag_prompt_level}"
                                        output_dir="data/evaluation_results/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${num_reranked_documents}_${rag_prompt_level}"
                                        
                                        # Skip if output directory exists and contains aggregate files
                                        if [ -d "$output_dir" ] && [ -n "$(ls -A $output_dir/*.aggregate.* 2>/dev/null)" ]; then
                                            echo "Skipping existing evaluation: $output_dir"
                                            continue
                                        fi
                                        
                                        results_path=$(ls -t "${base_dir}"/${reference_file}.run*.AnovaRAG.tsv 2>/dev/null | head -n1)
                                        if [ -z "$results_path" ]; then
                                            echo "No result file found in ${base_dir}, skipping..."
                                            continue
                                        fi
                                        
                                        echo "Evaluating with original_question_inlcuded=$original_question_inlcuded, k_queries=$k_queries, qpp=$qpp, first_step_ranker=$first_step_ranker, num_first_retrieved_documents=$num_first_retrieved_documents, fusion_method=$fusion_method, reranker=$reranker, num_reranked_documents=$num_reranked_documents, query_gen_prompt_level=$query_gen_prompt_level, rag_prompt_level=$rag_prompt_level"
                                        echo "Using result file: $results_path"
                                        uv run scripts/evaluate.py \
                                            --evaluator LLMEvaluator \
                                            --num_threads 20 \
                                            --results "$results_path" \
                                            --reference data/generated_qa_pairs/${reference_file}.tsv \
                                            --output-dir "$output_dir"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Reset trap
trap - SIGINT

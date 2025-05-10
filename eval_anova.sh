#!/bin/bash

# File prefix, used for aligning run and eval
reference_file="dmds_500_hard_sampled_15"

# Add trap handler for Ctrl+C
stop_loops=false
trap 'stop_loops=true; echo -e "\nStopping after current iteration..."' SIGINT

# Loop through all parameter combinations
for original_question_inlcuded in false true; do
    if [ "$stop_loops" = true ]; then break; fi
    for k_queries in 3 5 8; do
        if [ "$stop_loops" = true ]; then break; fi
        for qpp in no; do
            if [ "$stop_loops" = true ]; then break; fi
            for first_step_ranker in both keywords embedding; do
                if [ "$stop_loops" = true ]; then break; fi
                for num_first_retrieved_documents in 5 10 15; do
                    if [ "$stop_loops" = true ]; then break; fi
                    for fusion_method in concat; do
                        if [ "$stop_loops" = true ]; then break; fi
                        for reranker in no logits; do
                            if [ "$stop_loops" = true ]; then break; fi
                            for num_reranked_documents in 5 10 15; do
                                if [ "$stop_loops" = true ]; then break; fi
                                # for query_gen_prompt_level in naive medium advanced; do
                                for query_gen_prompt_level in naive medium; do
                                    if [ "$stop_loops" = true ]; then break; fi
                                    for rag_prompt_level in naive medium advanced; do
                                        if [ "$stop_loops" = true ]; then break; fi
                                        base_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${num_reranked_documents}_${rag_prompt_level}"
                                        results_path=$(ls -t "${base_dir}"/${reference_file}.run*.AnovaRAG.tsv 2>/dev/null | head -n1)
                                        if [ -z "$results_path" ]; then
                                            echo "No result file found in ${base_dir}, skipping..."
                                            continue
                                        fi
                                        output_dir="data/evaluation_results/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${num_reranked_documents}_${rag_prompt_level}"
                                        echo "Evaluating with original_question_inlcuded=$original_question_inlcuded, k_queries=$k_queries, qpp=$qpp, first_step_ranker=$first_step_ranker, num_first_retrieved_documents=$num_first_retrieved_documents, fusion_method=$fusion_method, reranker=$reranker, num_reranked_documents=$num_reranked_documents, query_gen_prompt_level=$query_gen_prompt_level, rag_prompt_level=$rag_prompt_level"
                                        echo "Using result file: $results_path"
                                        uv run scripts/evaluate.py \
                                            --evaluator LLMEvaluator \
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

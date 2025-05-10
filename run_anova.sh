#!/bin/bash

llm_client="ec2_llm" # ai71 or ec2_llm
# File prefix, used for aligning run and eval
reference_file="dmds_500_hard_sampled_15"
# File paths
input="data/generated_qa_pairs/${reference_file}.tsv"

# Add trap handler for Ctrl+C
stop_loops=false
trap 'stop_loops=true; echo -e "\nStopping after current iteration..."' SIGINT

# Loop through all parameter combinations
for query_gen_prompt_level in naive medium; do
    if [ "$stop_loops" = true ]; then break; fi
    for rag_prompt_level in naive medium advanced; do
        if [ "$stop_loops" = true ]; then break; fi
        for original_question_inlcuded in false true; do
            if [ "$stop_loops" = true ]; then break; fi
            for k_queries in 4 5 8; do
                if [ "$stop_loops" = true ]; then break; fi
                for qpp in no; do
                    if [ "$stop_loops" = true ]; then break; fi
                    for first_step_ranker in both keywords embedding; do
                        if [ "$stop_loops" = true ]; then break; fi
                        for num_first_retrieved_documents in 5 8; do
                            if [ "$stop_loops" = true ]; then break; fi
                            for fusion_method in concat; do
                                if [ "$stop_loops" = true ]; then break; fi
                                for reranker in no logits; do
                                    if [ "$stop_loops" = true ]; then break; fi
                                    
                                    # Set num_reranked_documents based on reranker
                                    if [ "$reranker" = "no" ]; then
                                        num_reranked_documents_values=(0)
                                    else
                                        num_reranked_documents_values=(10 15 20)
                                    fi
                                    
                                    for num_reranked_documents in "${num_reranked_documents_values[@]}"; do
                                        if [ "$stop_loops" = true ]; then break; fi
                                        
                                        output_dir="data/anova_result/${original_question_inlcuded}_${k_queries}_${query_gen_prompt_level}_${qpp}_${num_first_retrieved_documents}_${first_step_ranker}_${fusion_method}_${reranker}_${num_reranked_documents}_${rag_prompt_level}"
                                        
                                        # Skip if output directory exists and contains run files
                                        if [ -d "$output_dir" ] && [ -n "$(ls -A $output_dir/*.run 2>/dev/null)" ]; then
                                            echo "Skipping existing configuration: $output_dir"
                                            continue
                                        fi
                                        
                                        common_args="--llm_client $llm_client --system AnovaRAG --input $input --output-dir $output_dir --num-threads 20 --original_question_inlcuded=$original_question_inlcuded --k_queries=$k_queries --qpp=$qpp --num_first_retrieved_documents $num_first_retrieved_documents --first_step_ranker $first_step_ranker --fusion_method $fusion_method --reranker $reranker --num_reranked_documents $num_reranked_documents --rag_prompt_level $rag_prompt_level --query_gen_prompt_level $query_gen_prompt_level"
                                        echo "Running with original_question_inlcuded=$original_question_inlcuded, k_queries=$k_queries, qpp=$qpp, first_step_ranker=$first_step_ranker, num_first_retrieved_documents=$num_first_retrieved_documents, fusion_method=$fusion_method, reranker=$reranker, num_reranked_documents=$num_reranked_documents, query_gen_prompt_level=$query_gen_prompt_level, rag_prompt_level=$rag_prompt_level"
                                        uv run scripts/run.py $common_args
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
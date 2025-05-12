#!/bin/bash

# Change to the project root directory
cd "$(dirname "$0")/../../../"

# File prefix, used for aligning run and eval
reference_file="dmds_85random_15hard_questions"

# Configuration lists
QUERY_GEN_PROMPT_LEVELS=("medium" "advanced")
RAG_PROMPT_LEVELS=("naive" "advanced")
QUERY_EXPANSION_MODES=("none" "variants" "decomposition")
N_QUERIES_VALUES=(8)
QPP_VALUES=("no")
FIRST_STEP_RANKER_VALUES=("both_concat" "both_fusion")
INITIAL_RETRIEVAL_K_DOCS_VALUES=(50 100)
RERANKER_VALUES=("logits")
CONTEXT_WORDS_LIMIT_VALUES=(10000 15000)

# Add trap handler for Ctrl+C
stop_loops=false
trap 'stop_loops=true; echo -e "\nStopping after current iteration..."' SIGINT

# Loop through all parameter combinations
for query_gen_prompt_level in "${QUERY_GEN_PROMPT_LEVELS[@]}"; do
    if [ "$stop_loops" = true ]; then break; fi
    for rag_prompt_level in "${RAG_PROMPT_LEVELS[@]}"; do
        if [ "$stop_loops" = true ]; then break; fi
        for query_expansion_mode in "${QUERY_EXPANSION_MODES[@]}"; do
            if [ "$stop_loops" = true ]; then break; fi
            for n_queries in "${N_QUERIES_VALUES[@]}"; do
                if [ "$stop_loops" = true ]; then break; fi
                for qpp in "${QPP_VALUES[@]}"; do
                    if [ "$stop_loops" = true ]; then break; fi
                    for first_step_ranker in "${FIRST_STEP_RANKER_VALUES[@]}"; do
                        if [ "$stop_loops" = true ]; then break; fi
                        for initial_retrieval_k_docs in "${INITIAL_RETRIEVAL_K_DOCS_VALUES[@]}"; do
                            if [ "$stop_loops" = true ]; then break; fi
                            for reranker in "${RERANKER_VALUES[@]}"; do
                                if [ "$stop_loops" = true ]; then break; fi
                                for context_words_limit in "${CONTEXT_WORDS_LIMIT_VALUES[@]}"; do
                                    if [ "$stop_loops" = true ]; then break; fi
                                    
                                    base_dir="data/anova_result/${query_expansion_mode}_${n_queries}_${query_gen_prompt_level}_${qpp}_${initial_retrieval_k_docs}_${first_step_ranker}_${reranker}_${context_words_limit}_${rag_prompt_level}"
                                    output_dir="data/evaluation_results/${query_expansion_mode}_${n_queries}_${query_gen_prompt_level}_${qpp}_${initial_retrieval_k_docs}_${first_step_ranker}_${reranker}_${context_words_limit}_${rag_prompt_level}"
                                    
                                    # Skip if output directory exists and contains aggregate files
                                    if [ -d "$output_dir" ] && [ -n "$(ls -A $output_dir/*.aggregate.* 2>/dev/null)" ]; then
                                        echo "Skipping existing evaluation: $output_dir"
                                        continue
                                    fi
                                    
                                    results_path=$(ls -t "${base_dir}"/${reference_file}.run*.AnovaRAGLite.tsv 2>/dev/null | head -n1)
                                    if [ -z "$results_path" ]; then
                                        echo "No result file found in ${base_dir}, skipping..."
                                        continue
                                    fi
                                    
                                    echo "Evaluating with query_expansion_mode=$query_expansion_mode, n_queries=$n_queries, qpp=$qpp, first_step_ranker=$first_step_ranker, initial_retrieval_k_docs=$initial_retrieval_k_docs, reranker=$reranker, context_words_limit=$context_words_limit, query_gen_prompt_level=$query_gen_prompt_level, rag_prompt_level=$rag_prompt_level"
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

# Reset trap
trap - SIGINT

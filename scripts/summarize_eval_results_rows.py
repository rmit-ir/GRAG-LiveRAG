#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from pathlib import Path
import glob

def get_project_root():
    """Get the project root directory."""
    # Assuming the script is in scripts/ directory
    return Path(__file__).parent.parent

def resolve_path(path):
    """Resolve a path to absolute path, handling both relative and absolute paths."""
    if os.path.isabs(path):
        return path
    return os.path.join(get_project_root(), path)

def get_directory_factors(dir_path):
    """Extract all factors from directory path."""
    # Get the directory name and split by underscore
    dir_name = os.path.basename(dir_path)
    parts = dir_name.split('_')
    
    # Initialize factors list
    factors = []
    i = 0
    
    while i < len(parts):
        # Check for both_concat or both_fusion pattern
        if parts[i] == 'both' and i + 1 < len(parts) and parts[i + 1] in ['concat', 'fusion']:
            # Combine both and concat/fusion into one factor
            factors.append(f"both_{parts[i + 1]}")
            i += 2
            # The next part should be the reranker value
            if i < len(parts):
                factors.append(parts[i])
                i += 1
        else:
            factors.append(parts[i])
            i += 1
    
    return factors

def get_factor_names():
    """Return the list of factor names in the correct order."""
    return [
        'query_expansion_mode',
        'n_queries',
        'query_gen_prompt_level',
        'qpp',
        'initial_retrieval_k_docs',
        'first_step_ranker',
        'reranker',
        'context_words_limit',
        'rag_prompt_level'
    ]

def process_eval_results(eval_results_path):
    """Process all evaluation results and combine them into a single DataFrame."""
    # Resolve the path to absolute
    eval_results_path = resolve_path(eval_results_path)
    print(f"Processing evaluation results from: {eval_results_path}")
    
    # Find all .rows. files in subdirectories
    pattern = os.path.join(eval_results_path, "**", "*.rows.*")
    row_files = glob.glob(pattern, recursive=True)
    
    if not row_files:
        print(f"No .rows. files found in {eval_results_path}")
        return None
    
    print(f"Found {len(row_files)} .rows. files to process")
    
    # List to store all DataFrames
    dfs = []
    factor_names = get_factor_names()
    
    for file_path in row_files:
        try:
            # Read the TSV file
            df = pd.read_csv(file_path, sep='\t')
            
            # Get all factors from the directory path
            dir_path = os.path.dirname(file_path)
            factors = get_directory_factors(dir_path)
            
            # Add each factor as a separate column with proper names
            for i, factor in enumerate(factors):
                if i < len(factor_names):
                    df[factor_names[i]] = factor
                else:
                    print(f"Warning: Extra factor found in directory name: {factor}")
            
            # Select only the required columns
            columns = ['qid'] + factor_names[:len(factors)] + ['relevance_score', 'faithfulness_score']
            df = df[columns]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not dfs:
        print("No valid data found in any files")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Get all factor columns that exist in the DataFrame
    factor_columns = [col for col in factor_names if col in combined_df.columns]
    
    # Sort by qid and all factors
    combined_df = combined_df.sort_values(['qid'] + factor_columns)
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description='Summarize evaluation results from multiple directories into a single CSV file.')
    parser.add_argument('eval_results_path', type=str, 
                      help='Path to the evaluation results directory (can be relative to project root, e.g., data/evaluation_results, or absolute path)')
    parser.add_argument('--output-name', type=str, default='eval_variants_summary.csv',
                      help='Name of the output CSV file (default: eval_variants_summary.csv)')
    
    args = parser.parse_args()
    
    # Process the evaluation results
    combined_df = process_eval_results(args.eval_results_path)
    
    if combined_df is not None:
        # Create output path
        output_path = os.path.join(resolve_path(args.eval_results_path), args.output_name)
        
        # Save to CSV with sorted data
        combined_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        print(f"Total records: {len(combined_df)}")
        print(f"Unique qids: {combined_df['qid'].nunique()}")
        
        # Print information about factors
        factor_columns = [col for col in get_factor_names() if col in combined_df.columns]
        print("\nFactors found in directory names:")
        for factor_col in factor_columns:
            print(f"{factor_col}: {combined_df[factor_col].nunique()} unique values")
            print(f"Unique values: {sorted(combined_df[factor_col].unique())}")
        
        # Print example of the grouping
        print("\nExample of grouped data (first qids):")
        example_df = combined_df[combined_df['qid'].isin(combined_df['qid'].unique()[:1])]
        print(example_df.to_string(index=False))
        
        # Print summary statistics by factors
        print("\nSummary statistics by factors:")
        factor_stats = combined_df.groupby(factor_columns)[['relevance_score', 'faithfulness_score']].agg(['mean', 'std', 'count'])
        print(factor_stats)
    else:
        print("No data was processed. Please check the input path and file structure.")

if __name__ == "__main__":
    main() 
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

def get_query_variants_label(dir_path):
    """Extract the query variants label from directory path."""
    # Get the directory name and split by underscore
    dir_name = os.path.basename(dir_path)
    return dir_name.split('_')[0]

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
    
    for file_path in row_files:
        try:
            # Read the TSV file
            df = pd.read_csv(file_path, sep='\t')
            
            # Get the query variants label from the directory path
            dir_path = os.path.dirname(file_path)
            query_variants_label = get_query_variants_label(dir_path)
            
            # Add the query variants label column
            df['query_variants_label'] = query_variants_label
            
            # Select only the required columns
            df = df[['qid', 'query_variants_label', 'relevance_score', 'faithfulness_score']]
            
            dfs.append(df)
            # print(f"Successfully processed: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not dfs:
        print("No valid data found in any files")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by qid and query_variants_label
    combined_df = combined_df.sort_values(['qid', 'query_variants_label'])
    
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
        print(f"Unique query variants: {combined_df['query_variants_label'].nunique()}")
        
        # Print example of the grouping
        print("\nExample of grouped data (first qids):")
        example_df = combined_df[combined_df['qid'].isin(combined_df['qid'].unique()[:1])]
        print(example_df.to_string(index=False))
    else:
        print("No data was processed. Please check the input path and file structure.")

if __name__ == "__main__":
    main() 
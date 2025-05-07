import argparse
import pandas as pd
import json
import os
from datetime import datetime
from scipy import stats
import re

from pathlib import Path

# Get the project root directory (parent of the scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def load_file(file_path):
    """Load a TSV or CSV file based on its extension or content."""
    # Resolve the file path relative to the project root if it's not absolute
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.endswith('.tsv'):
        return pd.read_csv(file_path, sep='\t')
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        print(f"Warning: File {file_path} does not have a recognized extension (.tsv or .csv).")

def find_column(df, pattern):
    """Find a column name that matches a pattern (case insensitive)."""
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    return None

def perform_t_test(df1, df2):
    """Perform t-tests on relevance and faithfulness scores."""
    results = {}
    
    # Find relevant columns
    qid_col1 = find_column(df1, 'qid')
    qid_col2 = find_column(df2, 'qid')
    
    if not qid_col1 or not qid_col2:
        raise ValueError("qid column not found in one or both files")
    
    # Find score columns
    rel_col1 = find_column(df1, 'relevance_score')
    rel_col2 = find_column(df2, 'relevance_score')
    
    if not rel_col1 or not rel_col2:
        raise ValueError("relevance score column not found in one or both files")
    
    faith_col1 = find_column(df1, 'faithfulness_score')
    faith_col2 = find_column(df2, 'faithfulness_score')
    
    if not faith_col1 or not faith_col2:
        raise ValueError("faithfulness score column not found in one or both files")
    
    # Prepare dataframes for merging by renaming columns
    df1_with_key = df1.rename(columns={qid_col1: 'qid_key', rel_col1: 'rel1', faith_col1: 'faith1'})
    df2_with_key = df2.rename(columns={qid_col2: 'qid_key', rel_col2: 'rel2', faith_col2: 'faith2'})
    
    # Select only the columns we need
    df1_with_key = df1_with_key[['qid_key', 'rel1', 'faith1']]
    df2_with_key = df2_with_key[['qid_key', 'rel2', 'faith2']]
    
    # Merge on qid to ensure we compare the same records
    merged = pd.merge(df1_with_key, df2_with_key, on='qid_key')
    
    if merged.empty:
        raise ValueError("No common qid values found between the files")
    
    # Perform t-test for relevance scores
    t_stat_rel, p_value_rel = stats.ttest_rel(merged['rel1'], merged['rel2'])
    
    results['relevance_test_score'] = float(t_stat_rel)
    results['relevance_is_significant'] = bool(p_value_rel < 0.05)
    
    # Perform t-test for faithfulness scores
    t_stat_faith, p_value_faith = stats.ttest_rel(merged['faith1'], merged['faith2'])
    
    results['faithfulness_test_score'] = float(t_stat_faith)
    results['faithfulness_is_significant'] = bool(p_value_faith < 0.05)
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Perform t-test on two files.')
    parser.add_argument('file1', help='Path to the first file')
    parser.add_argument('file2', help='Path to the second file')
    args = parser.parse_args()
    
    try:
        # Load the files
        df1 = load_file(args.file1)
        df2 = load_file(args.file2)
        
        # Perform t-tests
        results = perform_t_test(df1, df2)
        
        # Add file paths and test time to the results
        output = {
            'test_file_1': args.file1,
            'test_file_2': args.file2,
            **results,
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print the results to the console
        print(json.dumps(output, indent=4))
        
        # # Create the output directory if it doesn't exist
        # os.makedirs('data/t_test', exist_ok=True)
        
        # # Generate the output file name from the input file names
        # file1_name = os.path.basename(args.file1).split('.')[0]
        # file2_name = os.path.basename(args.file2).split('.')[0]
        # output_file = f"data/t_test/{file1_name}_{file2_name}.json"
        
        # # Write the results to a JSON file
        # with open(output_file, 'w') as f:
        #     json.dump(output, f, indent=4)
        
        # print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
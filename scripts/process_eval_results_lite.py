import os
import pandas as pd
import glob

def parse_folder_name(folder_name):
    """Parse the folder name to extract parameter values for AnovaRAGLite."""
    parts = folder_name.split('_')
    return {
        'query_expansion_mode': parts[0],
        'n_queries': parts[1],
        'query_gen_prompt_level': parts[2],
        'qpp': parts[3],
        'initial_retrieval_k_docs': parts[4],
        'first_step_ranker': parts[5],
        'reranker': parts[6],
        'context_words_limit': parts[7],
        'rag_prompt_level': parts[8]
    }

def get_scores_from_aggregated_file(folder_path):
    """Read scores from the aggregated.tsv file."""
    # Find the aggregated.tsv file
    aggregated_files = glob.glob(os.path.join(folder_path, "*.aggregated.tsv"))
    if not aggregated_files:
        return None, None
    
    # Read the most recent aggregated file
    latest_file = max(aggregated_files, key=os.path.getctime)
    df = pd.read_csv(latest_file, sep='\t')
    
    return df['avg_relevance_score'].iloc[0], df['avg_faithfulness_score'].iloc[0]

def main():
    # Get the project root directory (assuming script is in project_folder/scripts)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Base directory for evaluation results
    base_dir = os.path.join(project_root, "data", "evaluation_results")
    
    # List to store all results
    results = []
    
    # Iterate through all subdirectories
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
        
        try:
            # Parse folder name to get parameters
            params = parse_folder_name(folder_name)
            
            # Get scores from aggregated file
            relevance_score, faithfulness_score = get_scores_from_aggregated_file(folder_path)
            
            if relevance_score is not None and faithfulness_score is not None:
                # Add scores to parameters
                params['relevance_score'] = relevance_score
                params['faithfulness_score'] = faithfulness_score
                
                # Append to results
                results.append(params)
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            continue
    
    if not results:
        print("No valid results found to process.")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    
    # Sort the results by relevance score in descending order
    df = df.sort_values('relevance_score', ascending=False)
    
    # Save to CSV
    output_file = os.path.join(project_root, "data", "anova_result", "evaluation_results_summary_lite.csv")
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total configurations evaluated: {len(df)}")
    print("\nTop 5 configurations by relevance score:")
    print(df[['query_expansion_mode', 'n_queries', 'query_gen_prompt_level', 'first_step_ranker', 
              'reranker', 'context_words_limit', 'rag_prompt_level', 'relevance_score', 
              'faithfulness_score']].head().to_string())

if __name__ == "__main__":
    main() 
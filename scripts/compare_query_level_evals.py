import pandas as pd
import os

def read_factor_analysis_scores(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter rows based on specific conditions
    filtered_df = df[
        (df['query_expansion_mode'] == 'none') &
        (df['n_queries'] == 8) &
        (df['query_gen_prompt_level'] == 'medium') &
        (df['qpp'] == 'no') &
        (df['initial_retrieval_k_docs'] == 50) &
        (df['first_step_ranker'] == 'both_fusion') &
        (df['reranker'] == 'logits') &
        (df['context_words_limit'] == 15000) &
        (df['rag_prompt_level'] == 'naive')
    ]
    
    # Select only required columns
    return filtered_df[['qid', 'relevance_score', 'faithfulness_score']]

def read_hyde_scores(file_path):
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Select only required columns
    return df[['qid', 'relevance_score', 'faithfulness_score']]

def compare_scores(df1, df2):
    # Get unique query IDs from both dataframes
    factor_qids = set(df1['qid'])
    hyde_qids = set(df2['qid'])
    
    # Count overlapping query IDs
    overlapping_qids = len(factor_qids.intersection(hyde_qids))
    
    # Merge the dataframes on qid
    merged_df = pd.merge(df1, df2, on='qid', suffixes=('_factor', '_hyde'))
    
    # Count where factor analysis scores are higher
    factor_higher_relevance = sum(merged_df['relevance_score_factor'] > merged_df['relevance_score_hyde'])
    factor_higher_faithfulness = sum(merged_df['faithfulness_score_factor'] > merged_df['faithfulness_score_hyde'])
    
    # Count where hyde scores are higher
    hyde_higher_relevance = sum(merged_df['relevance_score_hyde'] > merged_df['relevance_score_factor'])
    hyde_higher_faithfulness = sum(merged_df['faithfulness_score_hyde'] > merged_df['faithfulness_score_factor'])
    
    # Count where scores are equal
    equal_relevance = sum(merged_df['relevance_score_factor'] == merged_df['relevance_score_hyde'])
    equal_faithfulness = sum(merged_df['faithfulness_score_factor'] == merged_df['faithfulness_score_hyde'])
    
    return {
        'overlapping_qids': overlapping_qids,
        'factor_analysis': {
            'higher_relevance': factor_higher_relevance,
            'higher_faithfulness': factor_higher_faithfulness
        },
        'hyde': {
            'higher_relevance': hyde_higher_relevance,
            'higher_faithfulness': hyde_higher_faithfulness
        },
        'equal_scores': {
            'relevance': equal_relevance,
            'faithfulness': equal_faithfulness
        }
    }

def main():
    # Define file paths
    factor_analysis_file = '../data/evaluation_results/factor_analysis_query_level_scores.csv'
    hyde_file = '../data/evaluation_results/factor_analysis_query_level_scores.hyde.tsv'
    
    # Read and process the files
    factor_df = read_factor_analysis_scores(factor_analysis_file)
    hyde_df = read_hyde_scores(hyde_file)
    
    # Compare scores
    comparison_results = compare_scores(factor_df, hyde_df)
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'file_name': 'factor_analysis_query_level_scores.csv',
            'higher_relevance_count': comparison_results['factor_analysis']['higher_relevance'],
            'higher_faithfulness_count': comparison_results['factor_analysis']['higher_faithfulness']
        },
        {
            'file_name': 'factor_analysis_query_level_scores.hyde.tsv',
            'higher_relevance_count': comparison_results['hyde']['higher_relevance'],
            'higher_faithfulness_count': comparison_results['hyde']['higher_faithfulness']
        },
        {
            'file_name': 'equal_scores',
            'higher_relevance_count': comparison_results['equal_scores']['relevance'],
            'higher_faithfulness_count': comparison_results['equal_scores']['faithfulness']
        },
        {
            'file_name': 'overlapping_query_ids',
            'higher_relevance_count': comparison_results['overlapping_qids'],
            'higher_faithfulness_count': 0  # Not applicable for this row
        }
    ])
    
    # Save results to CSV
    output_file = '../data/evaluation_results/score_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 
import pandas as pd
import numpy as np
import argparse

def analyze_factor_options(df, relevance_weight=0.6, faithfulness_weight=0.4):
    """
    Analyze the performance of different options within each factor for AnovaRAGLite.
    
    Args:
        df (pd.DataFrame): Input DataFrame with factor columns and score columns
        relevance_weight (float): Weight for relevance score
        faithfulness_weight (float): Weight for faithfulness score
    
    Returns:
        dict: Dictionary containing performance analysis for each factor
    """
    # Calculate weighted score
    df['weighted_score'] = (df['relevance_score'] * relevance_weight + 
                          df['faithfulness_score'] * faithfulness_weight)
    
    # Define AnovaRAGLite specific factor columns
    factor_columns = [
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
    
    # Verify all expected factor columns are present
    missing_factors = [col for col in factor_columns if col not in df.columns]
    if missing_factors:
        print(f"Warning: Missing expected factor columns: {missing_factors}")
        factor_columns = [col for col in factor_columns if col in df.columns]
    
    results = {}
    for factor in factor_columns:
        # Group by factor and calculate statistics
        grouped = df.groupby(factor).agg({
            'weighted_score': ['mean', 'std', 'count', 'min', 'max'],
            'relevance_score': ['mean', 'std'],
            'faithfulness_score': ['mean', 'std']
        }).round(3)
        
        # Calculate percentage of total
        total = grouped[('weighted_score', 'count')].sum()
        grouped[('weighted_score', 'percentage')] = (grouped[('weighted_score', 'count')] / total * 100).round(1)
        
        # Sort by mean weighted score
        grouped = grouped.sort_values(('weighted_score', 'mean'), ascending=False)
        
        results[factor] = grouped
    
    return results

def print_results(results):
    """Print the analysis results in a readable format."""
    for factor, stats in results.items():
        print(f"\nFactor: {factor}")
        print("=" * 100)
        print(f"{'Option':<20} {'Mean Score':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8} {'Count':<8} {'%':<8} {'Mean Rel':<12} {'Mean Faith':<12}")
        print("-" * 100)
        
        for option, row in stats.iterrows():
            print(f"{str(option):<20} "
                  f"{row[('weighted_score', 'mean')]:<12.3f} "
                  f"{row[('weighted_score', 'std')]:<12.3f} "
                  f"{row[('weighted_score', 'min')]:<8.3f} "
                  f"{row[('weighted_score', 'max')]:<8.3f} "
                  f"{row[('weighted_score', 'count')]:<8} "
                  f"{row[('weighted_score', 'percentage')]:<8.1f} "
                  f"{row[('relevance_score', 'mean')]:<12.3f} "
                  f"{row[('faithfulness_score', 'mean')]:<12.3f}")

def analyze_best_combinations(df, top_n=5):
    """Analyze the best performing combinations of factors."""
    print("\n--- Best Performing Combinations ---")
    print("=" * 100)
    
    # Sort by weighted score
    df_sorted = df.sort_values('weighted_score', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        print(f"\nCombination {i}:")
        print("-" * 50)
        for col in df.columns:
            if col not in ['weighted_score', 'relevance_score', 'faithfulness_score']:
                print(f"{col}: {row[col]}")
        print(f"Weighted Score: {row['weighted_score']:.3f}")
        print(f"Relevance Score: {row['relevance_score']:.3f}")
        print(f"Faithfulness Score: {row['faithfulness_score']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze AnovaRAGLite factor options performance')
    parser.add_argument('--csv_file', type=str, required=True,
                      help='Path to the CSV data file containing AnovaRAGLite results')
    parser.add_argument('--relevance_weight', type=float, default=0.6,
                      help='Weight for relevance score (default: 0.6)')
    parser.add_argument('--faithfulness_weight', type=float, default=0.4,
                      help='Weight for faithfulness score (default: 0.4)')
    parser.add_argument('--top_n', type=int, default=5,
                      help='Number of top combinations to show (default: 5)')
    
    args = parser.parse_args()
    
    try:
        # Read data from file
        df = pd.read_csv(args.csv_file)
        
        # Validate weights
        if not (0 <= args.relevance_weight <= 1 and 0 <= args.faithfulness_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(args.relevance_weight + args.faithfulness_weight - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1")
        
        # Analyze and print results
        print("--- Analyzing AnovaRAGLite Factor Options ---")
        results = analyze_factor_options(df, args.relevance_weight, args.faithfulness_weight)
        print_results(results)
        
        # Analyze best combinations
        analyze_best_combinations(df, args.top_n)
        
    except FileNotFoundError:
        print(f"Error: Could not find data file '{args.csv_file}'")
    except pd.errors.EmptyDataError:
        print(f"Error: The data file '{args.csv_file}' is empty")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
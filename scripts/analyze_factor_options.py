import pandas as pd
import numpy as np
import argparse

def analyze_factor_options(df, relevance_weight=0.5, faithfulness_weight=0.5):
    """
    Analyze the performance of different options within each factor.
    
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
    
    # Get factor columns (all columns except score columns)
    factor_cols = [col for col in df.columns if col not in ['relevance_score', 'faithfulness_score', 'weighted_score']]
    
    results = {}
    
    for factor in factor_cols:
        # Group by factor and calculate statistics
        grouped = df.groupby(factor)['weighted_score'].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Add percentage of total
        total_count = grouped['count'].sum()
        grouped['percentage'] = (grouped['count'] / total_count * 100).round(1)
        
        results[factor] = grouped
    
    return results

def print_results(results):
    """Print the analysis results in a readable format."""
    for factor, stats in results.items():
        print(f"\n=== {factor.upper()} ===")
        print("Options ranked by mean weighted score (descending):")
        print("=" * 80)
        print(f"{'Option':<20} {'Mean Score':<12} {'Std Dev':<12} {'Count':<8} {'% of Total':<10}")
        print("-" * 80)
        
        for option, row in stats.iterrows():
            print(f"{str(option):<20} {row['mean']:.3f}      {row['std']:.3f}      {row['count']:<8} {row['percentage']:.1f}%")
        print("=" * 80)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze factor options performance')
    parser.add_argument('--data_file', type=str, required=True,
                      help='Path to the CSV file containing the data')
    parser.add_argument('--relevance_weight', type=float, default=0.5,
                      help='Weight for relevance score (default: 0.5)')
    parser.add_argument('--faithfulness_weight', type=float, default=0.5,
                      help='Weight for faithfulness score (default: 0.5)')
    
    args = parser.parse_args()
    
    # Read the data from file
    try:
        df = pd.read_csv(args.data_file)
    except Exception as e:
        print(f"Error reading data file: {e}")
        exit(1)
    
    # Analyze factor options
    results = analyze_factor_options(df, 
                                   relevance_weight=args.relevance_weight,
                                   faithfulness_weight=args.faithfulness_weight)
    
    # Print results
    print_results(results) 
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from io import StringIO
import argparse
import os

def analyze_factor_importance(file_path_or_data, relevance_col='relevance_score', faithfulness_col='faithfulness_score', 
                              relevance_weight=0.6, faithfulness_weight=0.4, delimiter=','):
    """
    Analyzes the importance of categorical factors on a weighted score using ANOVA for AnovaRAGLite results.

    Args:
        file_path_or_data (str or pd.DataFrame): Path to the input data file (e.g., CSV) or a pandas DataFrame.
        relevance_col (str): Name of the column for the relevance score.
        faithfulness_col (str): Name of the column for the faithfulness score.
        relevance_weight (float): Weight for the relevance score.
        faithfulness_weight (float): Weight for the faithfulness score.
        delimiter (str): Delimiter used in the input file if file_path_or_data is a path.

    Returns:
        list: A list of factor names sorted by importance (descending F-statistic).
        pd.DataFrame: A DataFrame containing factors, their F-statistics, and p-values.
    """
    try:
        # 1. Load Data
        if isinstance(file_path_or_data, str):
            try:
                df = pd.read_csv(file_path_or_data, delimiter=delimiter)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path_or_data, delimiter=delimiter, encoding='latin1')
                except Exception as e:
                    print(f"Error reading file with multiple encodings: {e}")
                    return [], pd.DataFrame()
        elif isinstance(file_path_or_data, pd.DataFrame):
            df = file_path_or_data.copy()
        else:
            print("Error: file_path_or_data must be a file path (string) or a pandas DataFrame.")
            return [], pd.DataFrame()

        if df.empty:
            print("Error: The DataFrame is empty.")
            return [], pd.DataFrame()

        # 2. Identify score columns and factor columns
        score_cols = [relevance_col, faithfulness_col]
        if not all(col in df.columns for col in score_cols):
            print(f"Error: Score columns '{relevance_col}' and/or '{faithfulness_col}' not found in the data.")
            print(f"Available columns: {df.columns.tolist()}")
            return [], pd.DataFrame()

        # Convert score columns to numeric
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Define factor columns specific to AnovaRAGLite
        factor_cols = [
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
        missing_factors = [col for col in factor_cols if col not in df.columns]
        if missing_factors:
            print(f"Warning: Missing expected factor columns: {missing_factors}")
            factor_cols = [col for col in factor_cols if col in df.columns]

        print(f"Analyzing Factor Columns: {factor_cols}")
        print(f"Score Columns: {score_cols}")

        # 3. Calculate Weighted Score
        df['weighted_score'] = (df[relevance_col] * relevance_weight) + \
                               (df[faithfulness_col] * faithfulness_weight)

        # 4. Data Cleaning
        df_cleaned = df.dropna(subset=['weighted_score'])

        if df_cleaned.empty:
            print("Error: No data remaining after handling missing values in scores.")
            return [], pd.DataFrame()

        factor_results = []

        # 5. Perform ANOVA for each factor
        for factor in factor_cols:
            temp_df_factor_anova = df_cleaned[[factor, 'weighted_score']].dropna()

            if temp_df_factor_anova[factor].nunique() < 2:
                print(f"Skipping factor '{factor}': Not enough unique values for ANOVA.")
                factor_results.append({'Factor': factor, 'F-statistic': 0, 'p-value': 1.0, 'Error': 'Not enough unique values'})
                continue
            
            if len(temp_df_factor_anova) < 2:
                print(f"Skipping factor '{factor}': Not enough data points for ANOVA.")
                factor_results.append({'Factor': factor, 'F-statistic': 0, 'p-value': 1.0, 'Error': 'Not enough data points'})
                continue

            try:
                formula = f'weighted_score ~ C({factor})'
                model = ols(formula, data=temp_df_factor_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                f_statistic = anova_table['F'][f'C({factor})']
                p_value = anova_table['PR(>F)'][f'C({factor})']
                
                factor_results.append({'Factor': factor, 'F-statistic': f_statistic, 'p-value': p_value, 'Error': None})
            except Exception as e:
                print(f"Could not perform ANOVA for factor '{factor}': {e}")
                factor_results.append({'Factor': factor, 'F-statistic': float('-inf'), 'p-value': float('inf'), 'Error': str(e)})
        
        if not factor_results:
            print("No factors could be analyzed.")
            return [], pd.DataFrame()

        # 6. Create and sort results DataFrame
        results_df = pd.DataFrame(factor_results)
        results_df = results_df.sort_values(by='F-statistic', ascending=False)
        
        # 7. Get sorted factor names
        sorted_factors = results_df['Factor'].tolist()
        
        return sorted_factors, results_df

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], pd.DataFrame()

def analyze_best_configurations(df, top_n=5):
    """Analyze the best performing configurations."""
    # Sort by weighted score
    df_sorted = df.sort_values('weighted_score', ascending=False)
    
    print("\n--- Top Configurations by Weighted Score ---")
    for i, (_, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
        print(f"\nConfiguration {i}:")
        for col in df.columns:
            if col not in ['weighted_score', 'relevance_score', 'faithfulness_score']:
                print(f"{col}: {row[col]}")
        print(f"Weighted Score: {row['weighted_score']:.3f}")
        print(f"Relevance Score: {row['relevance_score']:.3f}")
        print(f"Faithfulness Score: {row['faithfulness_score']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze AnovaRAGLite results using ANOVA')
    parser.add_argument('--csv_file', type=str, required=True,
                      help='Path to the CSV file containing AnovaRAGLite evaluation results')
    parser.add_argument('--relevance_weight', type=float, default=0.6,
                      help='Weight for relevance score (default: 0.6)')
    parser.add_argument('--faithfulness_weight', type=float, default=0.4,
                      help='Weight for faithfulness score (default: 0.4)')
    parser.add_argument('--top_n', type=int, default=5,
                      help='Number of top configurations to show (default: 5)')
    
    args = parser.parse_args()
    
    print("--- Analyzing AnovaRAGLite Results ---")
    
    ordered_factors, factor_details_df = analyze_factor_importance(args.csv_file, 
                                                                  relevance_weight=args.relevance_weight, 
                                                                  faithfulness_weight=args.faithfulness_weight)

    if ordered_factors:
        print("\n--- Factor Importance Order (Most to Least Important) ---")
        for i, factor in enumerate(ordered_factors):
            print(f"{i+1}. {factor}")
        
        print("\n--- Detailed Factor Analysis Results ---")
        print(factor_details_df.to_string())
        
        # Load the data again for best configuration analysis
        df = pd.read_csv(args.csv_file)
        df['weighted_score'] = (df['relevance_score'] * args.relevance_weight) + \
                              (df['faithfulness_score'] * args.faithfulness_weight)
        analyze_best_configurations(df, args.top_n)
    else:
        print("\nAnalysis could not be completed. Please check error messages above.")

if __name__ == '__main__':
    main() 
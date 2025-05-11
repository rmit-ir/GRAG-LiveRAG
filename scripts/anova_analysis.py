import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from io import StringIO

def analyze_factor_importance(file_path_or_data, relevance_col='relevance_score', faithfulness_col='faithfulness_score', 
                              relevance_weight=0.5, faithfulness_weight=0.5, delimiter=','):
    """
    Analyzes the importance of categorical factors on a weighted score using ANOVA.

    Args:
        file_path_or_data (str or pd.DataFrame): Path to the input data file (e.g., CSV) or a pandas DataFrame.
        relevance_col (str): Name of the column for the relevance score.
        faithfulness_col (str): Name of the column for the faithfulness score.
        relevance_weight (float): Weight for the relevance score.
        faithfulness_weight (float): Weight for the faithfulness score.
        delimiter (str): Delimiter used in the input file if file_path_or_data is a path.

    Returns:
        list: A list of factor names sorted by importance (descending F-statistic).
              Returns an empty list and prints error messages if issues occur.
        pd.DataFrame: A DataFrame containing factors, their F-statistics, and p-values.
                      Returns an empty DataFrame if issues occur.
    """
    try:
        # 1. Load Data
        if isinstance(file_path_or_data, str):
            # Try reading with common encodings if default utf-8 fails
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

        # Attempt to convert score columns to numeric, coercing errors to NaN
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Define factor columns: all columns except the score columns
        # This assumes score columns are distinctly named and not among factors.
        # A more robust way if there are other non-factor ID columns is to explicitly list factor columns.
        # For this script, we assume all other columns are factors.
        factor_cols = [col for col in df.columns if col not in score_cols]
        
        if not factor_cols:
            print("Error: No factor columns found. Factors are assumed to be all columns other than the specified score columns.")
            return [], pd.DataFrame()

        print(f"Identified Factor Columns: {factor_cols}")
        print(f"Identified Score Columns: {score_cols}")

        # 3. Calculate Weighted Score
        df['weighted_score'] = (df[relevance_col] * relevance_weight) + \
                               (df[faithfulness_col] * faithfulness_weight)

        # 4. Data Cleaning: Drop rows where weighted_score is NaN 
        # (this happens if relevance_score or faithfulness_score was NaN or non-numeric)
        df_cleaned = df.dropna(subset=['weighted_score'])

        if df_cleaned.empty:
            print("Error: No data remaining after handling missing values in scores. Cannot perform ANOVA.")
            return [], pd.DataFrame()

        factor_results = []

        # 5. Perform ANOVA for each factor
        for factor in factor_cols:
            # Ensure the factor column is treated as categorical and has more than one level
            # in the cleaned data subset relevant for this factor.
            
            # Create a temporary DataFrame for the current factor's ANOVA
            # This ensures that we only use rows where the factor itself is not NaN
            temp_df_factor_anova = df_cleaned[[factor, 'weighted_score']].dropna()

            if temp_df_factor_anova[factor].nunique() < 2:
                print(f"Skipping factor '{factor}': Not enough unique values (less than 2) for ANOVA after cleaning.")
                factor_results.append({'Factor': factor, 'F-statistic': 0, 'p-value': 1.0, 'Error': 'Not enough unique values'})
                continue
            
            if len(temp_df_factor_anova) < 2: # Need at least 2 data points
                 print(f"Skipping factor '{factor}': Not enough data points (less than 2) for ANOVA after cleaning.")
                 factor_results.append({'Factor': factor, 'F-statistic': 0, 'p-value': 1.0, 'Error': 'Not enough data points'})
                 continue


            try:
                # Using C() to ensure the factor is treated as categorical
                formula = f'weighted_score ~ C({factor})'
                model = ols(formula, data=temp_df_factor_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # The F-statistic and p-value are for the factor itself
                f_statistic = anova_table['F'][f'C({factor})']
                p_value = anova_table['PR(>F)'][f'C({factor})']
                
                factor_results.append({'Factor': factor, 'F-statistic': f_statistic, 'p-value': p_value, 'Error': None})
            except Exception as e:
                print(f"Could not perform ANOVA for factor '{factor}': {e}")
                factor_results.append({'Factor': factor, 'F-statistic': float('-inf'), 'p-value': float('inf'), 'Error': str(e)})
        
        if not factor_results:
            print("No factors could be analyzed.")
            return [], pd.DataFrame()

        # 6. Create a DataFrame from results and sort
        results_df = pd.DataFrame(factor_results)
        results_df = results_df.sort_values(by='F-statistic', ascending=False)
        
        # 7. Get the sorted list of factor names
        sorted_factors = results_df['Factor'].tolist()
        
        return sorted_factors, results_df

    except FileNotFoundError:
        print(f"Error: The file '{file_path_or_data}' was not found.")
        return [], pd.DataFrame()
    except pd.errors.EmptyDataError:
        print("Error: The provided file is empty.")
        return [], pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], pd.DataFrame()

if __name__ == '__main__':

    
    print("--- Running Analysis with Data from CSV file ---")
    # Specify your score column names if they are different
    # relevance_col_name = 'relevance_score'
    # faithfulness_col_name = 'faithfulness_score'
    
    # Specify weights if different from 0.5/0.5
    # rel_weight = 0.6
    # faith_weight = 0.4
    
    # Specify the path to the CSV file
    csv_file_path = '/Users/sunshuoqi/Downloads/harder-questions-not-complete/anova_result/evaluation_results_summary.csv'   
    
    ordered_factors, factor_details_df = analyze_factor_importance(csv_file_path, 
                                                                  relevance_col='relevance_score', 
                                                                  faithfulness_col='faithfulness_score',
                                                                  relevance_weight=0.5, 
                                                                  faithfulness_weight=0.5,
                                                                  delimiter=',')

    if ordered_factors:
        print("\n--- Factor Importance Order (Most to Least Important) ---")
        for i, factor in enumerate(ordered_factors):
            print(f"{i+1}. {factor}")
        
        print("\n--- Detailed Factor Analysis Results ---")
        print(factor_details_df.to_string())
    else:
        print("\nAnalysis could not be completed. Please check error messages above.")


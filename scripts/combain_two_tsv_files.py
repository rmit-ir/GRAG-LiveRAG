#!/usr/bin/env python3
"""
Script to combine two TSV files with identical structure.
The second file will be appended to the first file (excluding its header).
"""

import os
import pandas as pd
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def combine_tsv_files(file1_path: str, file2_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Combine two TSV files by appending the second file (excluding its header) to the first.
    
    Args:
        file1_path: Path to the first TSV file
        file2_path: Path to the second TSV file
        output_path: Path to save the combined TSV file (optional)
    
    Returns:
        Combined pandas DataFrame
    """

    # Check if files exist
    for file_path in [file1_path, file2_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the TSV files
    logger.info(f"Reading first file: {file1_path}")
    df1 = pd.read_csv(file1_path, sep='\t', encoding='utf-8')
    logger.info(f"Reading second file: {file2_path}")
    df2 = pd.read_csv(file2_path, sep='\t', encoding='utf-8')
    
    # Check if the column structures match
    if list(df1.columns) != list(df2.columns):
        raise ValueError("The column structures of the two files do not match")
    
    id_column = "qid"
    logger.info(f"ID column: {id_column}")
    
    # Get the maximum ID from the first DataFrame
    max_id_df1 = df1[id_column].max()
    logger.info(f"Maximum ID in first file: {max_id_df1}")
    
    # Check if ID column is numeric
    if pd.api.types.is_numeric_dtype(df1[id_column]) and pd.api.types.is_numeric_dtype(df2[id_column]):
        # Update the IDs in the second DataFrame to continue from max_id_df1
        logger.info(f"Updating IDs in second file to continue from {max_id_df1}")
        
        # Create a copy of df2 to avoid modifying the original
        df2 = df2.copy()
        
        # Get the minimum ID from the second DataFrame to calculate the offset
        min_id_df2 = df2[id_column].min()
        
        # Calculate the offset to add to each ID in df2
        id_offset = max_id_df1 - min_id_df2 + 1
        
        # Update the IDs in df2
        df2[id_column] = df2[id_column] + id_offset
        
        logger.info(f"Updated IDs in second file with offset: {id_offset}")
    else:
        logger.warning(f"ID column '{id_column}' is not numeric. Proceeding without ID adjustment.")
        
    # Combine the DataFrames
    logger.info("Combining files")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
        # Set default output path if not provided
    if output_path is None:
        output_path = f"dmds_combained_questions_{len(combined_df)}.tsv"
        
    # Save the combined DataFrame
    combined_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    logger.info(f"Saved combined dataset as TSV to {output_path}")
    logger.info(f"Total rows: {len(combined_df)}")
    
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two TSV files with identical structure")
    parser.add_argument("file1", help="Path to the first TSV file")
    parser.add_argument("file2", help="Path to the second TSV file")
    
    args = parser.parse_args()
    
    try:
        combine_tsv_files(args.file1, args.file2)
        logger.info("Files combined successfully")
    except Exception as e:
        logger.error(f"Error combining files: {e}")
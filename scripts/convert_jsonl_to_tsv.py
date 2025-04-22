#!/usr/bin/env python
"""
Script to convert all JSONL files in the data/generated_qa_pairs directory to TSV format.

This script:
1. Finds all JSONL files in the data/generated_qa_pairs directory
2. Loads each JSONL file
3. Converts the data to a DataFrame using the qa_pairs_to_enhanced_dataframe function
4. Saves the DataFrame as a TSV file with a new filename format starting with "dmds_"

Usage:
    uv run scripts/convert_jsonl_to_tsv.py
"""

import os
import sys
import glob
import pandas as pd

# Add scripts folder to the Python path to allow importing from scripts
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from services.ds_data_morgana import QAPair
from utils.path_utils import get_data_dir
from utils.logging_utils import get_logger
from utils.query_utils import generate_short_id
from create_datamorgana_dataset import qa_pairs_to_enhanced_dataframe, save_dataframe

# Initialize logger
logger = get_logger("convert_jsonl_to_tsv")


def convert_jsonl_to_tsv():
    """
    Convert all JSONL files in the data/generated_qa_pairs directory to TSV format.
    """
    # Get the path to the data/generated_qa_pairs directory
    data_dir = get_data_dir()
    qa_pairs_dir = os.path.join(data_dir, "generated_qa_pairs")
    
    # Find all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(qa_pairs_dir, "results_id_*.jsonl"))
    
    if not jsonl_files:
        logger.info("No JSONL files found in the directory", directory=qa_pairs_dir)
        return
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to convert", directory=qa_pairs_dir)
    
    # Process each JSONL file
    for jsonl_file in jsonl_files:
        try:
            # Extract the file name without path and extension
            file_name = os.path.basename(jsonl_file)
            logger.info(f"Processing file", file=file_name)
            
            # Load the JSONL file
            df = pd.read_json(jsonl_file, lines=True)
            logger.info(f"Loaded {len(df)} records from file", file=file_name)
            
            # Convert records to QAPair objects
            qa_pairs = []
            for record in df.to_dict('records'):
                qa_pairs.append(QAPair.from_dict(record))
            
            # Convert QAPair objects to enhanced DataFrame
            enhanced_df = qa_pairs_to_enhanced_dataframe(qa_pairs)
            
            # Generate a new file name for the TSV file
            short_id = generate_short_id()
            record_count = len(enhanced_df)
            new_file_name = f"dmds_{short_id}.n{record_count}.tsv"
            output_path = os.path.join(qa_pairs_dir, new_file_name)
            
            # Save the DataFrame as a TSV file
            save_dataframe(enhanced_df, output_path, 'tsv')
            
            logger.info(f"Converted file to TSV", 
                       source=file_name, 
                       destination=new_file_name, 
                       records=record_count)
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}", error=str(e))


def main():
    """Main function to run the conversion."""
    logger.info("Starting conversion of JSONL files to TSV")
    
    try:
        convert_jsonl_to_tsv()
        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error("Conversion failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

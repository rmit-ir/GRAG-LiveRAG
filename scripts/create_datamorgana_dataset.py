#!/usr/bin/env python
"""
Script to create a DataMorgana query answer dataset and represent it in a tabular format.

This script can be run with uv to generate a dataset of question-answer pairs using DataMorgana,
and then convert it to a pandas DataFrame for analysis and storage.

Usage:
    uv run scripts/create_datamorgana_dataset.py [options]

Options:
    --n_questions=<n>       Number of questions to generate [default: 2]
    --output=<path>         Path to save the output file [default: auto-generated in data/generated_qa_pairs/]
    --format=<format>       Output format (tsv, excel, parquet, jsonl) [default: tsv]
    --document_ids=<ids>    Comma-separated list of document IDs to use (optional)
    --wait_time=<seconds>   Time to wait between polling attempts in seconds [default: 2]
    --config=<path>         Path to config file [default: scripts/config/datamorgana_config.json]

The script uses a JSON configuration file to define question and user categories for DataMorgana.
This file is located in the scripts/config/ directory by default.
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# Import project modules using the correct import format
from services.ds_data_morgana import DataMorgana, QAPair
from utils.path_utils import get_data_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a DataMorgana query answer dataset",
        epilog="""
Examples:
    # Basic usage with default parameters (generates 2 questions in TSV format)
    uv run scripts/create_datamorgana_dataset.py
    
    # Generate 50 questions
    uv run scripts/create_datamorgana_dataset.py --n_questions=50
    
    # Generate questions and save as Excel file
    uv run scripts/create_datamorgana_dataset.py --format=excel
    
    # Generate questions and save to a specific path
    uv run scripts/create_datamorgana_dataset.py --output=data/my_dataset.tsv
    
    # Generate questions using specific document IDs
    uv run scripts/create_datamorgana_dataset.py --document_ids=doc1,doc2,doc3
    
    # Generate questions with a custom configuration file
    uv run scripts/create_datamorgana_dataset.py --config=path/to/custom_config.json
    
    # Generate 20 questions in JSONL format with longer wait time between polling
    uv run scripts/create_datamorgana_dataset.py --n_questions=20 --format=jsonl --wait_time=5
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n_questions", type=int, default=2, help="Number of questions to generate")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output file")
    parser.add_argument("--format", type=str, default="tsv", choices=["tsv", "excel", "parquet", "jsonl"], 
                        help="Output format")
    parser.add_argument("--document_ids", type=str, default=None, 
                        help="Comma-separated list of document IDs to use")
    parser.add_argument("--wait_time", type=int, default=2, 
                        help="Time to wait between polling attempts in seconds")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    return parser.parse_args()


def get_script_dir():
    """Get the directory of the current script."""
    return os.path.dirname(os.path.abspath(__file__))


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(get_script_dir(), 'config', 'datamorgana_config.json')
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_question_categories(config_data):
    return config_data.get('question_categories', [])


def get_user_categories(config_data):
    return config_data.get('user_categories', [])


def qa_pairs_to_enhanced_dataframe(qa_pairs: List[QAPair]) -> pd.DataFrame:
    """
    Convert a list of QAPair objects to a pandas DataFrame with enhanced columns.
    
    This function creates a DataFrame with columns for the question, answer, and context,
    as well as additional columns for each question and user category.
    
    Args:
        qa_pairs: List of QAPair objects
        
    Returns:
        pd.DataFrame: DataFrame with enhanced columns
    """
    # Basic data extraction
    data = []
    for qa in qa_pairs:
        # Start with basic fields
        row = {
            'question': qa.question,
            'answer': qa.answer,
            'context': qa.context,
            'document_ids': qa.document_ids
        }
        
        # Extract question categories into separate columns
        for qc in qa.question_categories:
            cat_name = f"question_{qc['categorization_name']}"
            row[cat_name] = qc['category_name']
        
        # Extract user categories into separate columns
        for uc in qa.user_categories:
            cat_name = f"user_{uc['categorization_name']}"
            row[cat_name] = uc['category_name']
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add metadata columns
    df['generation_timestamp'] = datetime.now().isoformat()
    df['question_length'] = df['question'].str.len()
    df['answer_length'] = df['answer'].str.len()
    df['context_length'] = df['context'].apply(lambda x: sum(len(c) for c in x) if isinstance(x, list) else 0)
    
    return df


def save_dataframe(df: pd.DataFrame, output_path: str, format: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save in the specified format
    if format == 'tsv':
        df.to_csv(output_path, index=False, sep='\t')
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'jsonl':
        df.to_json(output_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Dataset saved to {output_path}")


def main():
    """Main function to create a DataMorgana dataset."""
    # Load environment variables from .env file
    load_dotenv()
    
    args = parse_arguments()
    
    # Initialize DataMorgana client
    print("Initializing DataMorgana client...")
    dm = DataMorgana()
    
    # Parse document IDs if provided
    document_ids = None
    if args.document_ids:
        document_ids = args.document_ids.split(',')
        print(f"Using document IDs: {document_ids}")
    
    # Load config and get categories
    config_data = load_config(args.config)
    question_categorizations = get_question_categories(config_data)
    user_categorizations = get_user_categories(config_data)
    
    # Generate bulk QA pairs
    print(f"Generating {args.n_questions} QA pairs...")
    generation = dm.generate_qa_pair_bulk(
        n_questions=args.n_questions,
        question_categorizations=question_categorizations,
        user_categorizations=user_categorizations,
        document_ids=document_ids
    )
    
    # Get the generation ID
    generation_id = generation["request_id"]
    print(f"Generation ID: {generation_id}")
    
    # Wait for and retrieve generation results
    print("Waiting for generation results...")
    qa_pairs = dm.wait_generation_results(generation_id, sleep_sec=args.wait_time)
    print(f"Retrieved {len(qa_pairs)} QA pairs")
    
    # Convert to enhanced DataFrame
    print("Converting to DataFrame...")
    df = qa_pairs_to_enhanced_dataframe(qa_pairs)
    
    # Display DataFrame info
    print("\nDataFrame Information:")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Save DataFrame
    if args.output:
        output_path = args.output
    else:
        # Generate default output path with number of records
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_count = len(df)
        filename = f"datamorgana_dataset_{timestamp}.n{record_count}.{args.format}"
        output_path = os.path.join(get_data_dir(), "generated_qa_pairs", filename)
    
    save_dataframe(df, output_path, args.format)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total QA pairs: {len(df)}")
    
    # Question categories distribution
    print("\nQuestion Categories Distribution:")
    for col in [c for c in df.columns if c.startswith('question_')]:
        if col != 'question_length':
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    # User categories distribution
    print("\nUser Categories Distribution:")
    for col in [c for c in df.columns if c.startswith('user_')]:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    return df


if __name__ == "__main__":
    df = main()

#!/usr/bin/env python
"""
Script to create a DataMorgana query answer dataset and represent it in a tabular format.

uv run scripts/create_datamorgana_dataset.py [options]

The script uses a JSON configuration file to define question and user categories for DataMorgana.

This file is located in the scripts/config/ directory by default, you can also 
specify a custom path with --config
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
from utils.logging_utils import get_logger
from utils.query_utils import generate_short_id

# Initialize logger
logger = get_logger("datamorgana_dataset")


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
    uv run scripts/create_datamorgana_dataset.py --n_questions=20 --format=jsonl --wait_time=10
    
    # Process results from an existing generation ID
    uv run scripts/create_datamorgana_dataset.py --generation-id=your-generation-id
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n_questions", type=int, default=2,
                        help="Number of questions to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output file")
    parser.add_argument("--format", type=str, default="tsv", choices=["tsv", "excel", "parquet", "jsonl"],
                        help="Output format")
    parser.add_argument("--document_ids", type=str, default=None,
                        help="Comma-separated list of document IDs to use")
    parser.add_argument("--wait_time", type=int, default=5,
                        help="Time to wait between polling attempts in seconds")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--generation-id", type=str, default=None,
                        help="Generation ID to wait for results instead of creating new generation")
    return parser.parse_args()


def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(
            script_dir, 'config', 'datamorgana_config.json')

    with open(config_path, 'r') as f:
        return json.load(f)


def qa_pairs_to_enhanced_dataframe(qa_pairs: List[QAPair]) -> pd.DataFrame:
    """
    Convert a list of QAPair objects to a pandas DataFrame with enhanced columns.

    This function creates a DataFrame with columns for the question, answer, and context,

    enhanced?: additional columns for each question and user category.

    Args:
        qa_pairs: List of QAPair objects

    Returns:
        pd.DataFrame: DataFrame with enhanced columns
    """
    # Basic data extraction
    data = []
    for idx, qa in enumerate(qa_pairs, start=1):
        # Start with basic fields
        row = {
            # Use existing qid or generate one
            'qid': qa.qid if qa.qid is not None else str(idx),
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
    df['question_length'] = df['question'].apply(lambda x: len(x.split()))
    df['answer_length'] = df['answer'].apply(lambda x: len(x.split()))
    df['context_length'] = df['context'].apply(lambda x: sum(
        len(c.split()) for c in x) if isinstance(x, list) else 0)

    return df


def save_dataframe(df: pd.DataFrame, output_path: str, format: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save in the specified format
    if format == 'tsv':
        df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
        logger.info("Saved dataset as TSV", path=output_path, rows=len(df))
    elif format == 'excel':
        df.to_excel(output_path, index=False)
        logger.info("Saved dataset as Excel", path=output_path, rows=len(df))
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
        logger.info("Saved dataset as Parquet", path=output_path, rows=len(df))
    elif format == 'jsonl':
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        logger.info("Saved dataset as JSONL", path=output_path, rows=len(df))
    else:
        logger.error("Unsupported format", format=format)
        raise ValueError(f"Unsupported format: {format}")


def process_generation_results(dm, generation_id, wait_time, output=None, format="tsv"):
    """
    Process generation results from a given generation ID.
    
    Args:
        dm: DataMorgana client
        generation_id: ID of the generation to process
        wait_time: Time to wait between polling attempts in seconds
        output: Path to save the output file (optional)
        format: Output format (tsv, excel, parquet, jsonl)
        
    Returns:
        pd.DataFrame: DataFrame with the processed results
    """
    # Wait for and retrieve generation results
    logger.info("Waiting for generation results", generation_id=generation_id, wait_time=wait_time)
    qa_pairs = dm.wait_generation_results(generation_id, sleep_sec=wait_time)
    logger.info("Retrieved QA pairs", count=len(qa_pairs))
    
    # Convert to enhanced DataFrame
    logger.debug("Converting to DataFrame")
    df = qa_pairs_to_enhanced_dataframe(qa_pairs)
    
    # Log DataFrame info
    logger.info("DataFrame created",
                shape=df.shape,
                columns=list(df.columns),
                avg_question_length=df['question_length'].mean(),
                avg_answer_length=df['answer_length'].mean())
    
    # Save DataFrame
    if output:
        output_path = output
    else:
        # Generate default output path with number of records and datetime
        record_count = len(df)
        current_datetime = datetime.now().strftime("%m%d%H%M")  # Format: MMDDHHMM
        filename = f"dmds_{record_count}_{current_datetime}.{format}"
        output_path = os.path.join(get_data_dir(), "generated_qa_pairs", filename)
        logger.debug("Generated output path", path=output_path)
    
    save_dataframe(df, output_path, format)
    
    # Log summary statistics
    logger.info("Dataset summary", total_qa_pairs=len(df))
    
    # Question categories distribution
    question_categories = {}
    for col in [c for c in df.columns if c.startswith('question_') and c != 'question_length']:
        question_categories[col] = df[col].value_counts().to_dict()
    logger.info("Question categories distribution", categories=question_categories)
    
    # User categories distribution
    user_categories = {}
    for col in [c for c in df.columns if c.startswith('user_')]:
        user_categories[col] = df[col].value_counts().to_dict()
    logger.info("User categories distribution", categories=user_categories)
    
    logger.info("DataMorgana dataset creation completed successfully", output_path=output_path)
    return df


def main():
    """Main function to create a DataMorgana dataset."""
    # Load environment variables from .env file
    load_dotenv()

    logger.info("Starting DataMorgana dataset creation")
    args = parse_arguments()
    logger.debug("Arguments parsed", n_questions=args.n_questions, format=args.format,
                 wait_time=args.wait_time, config_path=args.config)

    # Initialize DataMorgana client
    logger.info("Initializing DataMorgana client")
    dm = DataMorgana()

    # Check if generation ID is provided
    if args.generation_id:
        logger.info("Using provided generation ID", generation_id=args.generation_id)
        # Skip generation step and process results directly
        return process_generation_results(
            dm=dm,
            generation_id=args.generation_id,
            wait_time=args.wait_time,
            output=args.output,
            format=args.format
        )

    # Parse document IDs if provided
    document_ids = None
    if args.document_ids:
        document_ids = args.document_ids.split(',')
        logger.info("Using specific document IDs", document_ids=document_ids)

    # Load config and get categories
    try:
        config_data = load_config(args.config)
        question_categorizations = config_data.get('question_categories', [])
        user_categorizations = config_data.get('user_categories', [])
        logger.debug("Config loaded",
                     question_categories_count=len(question_categorizations),
                     user_categories_count=len(user_categorizations))
    except Exception as e:
        logger.error("Failed to load config",
                     error=str(e), config_path=args.config)
        raise

    # Generate bulk QA pairs
    logger.info("Generating QA pairs", count=args.n_questions)
    try:
        generation = dm.generate_qa_pair_bulk(
            n_questions=args.n_questions,
            question_categorizations=question_categorizations,
            user_categorizations=user_categorizations,
            document_ids=document_ids
        )

        # Get the generation ID
        generation_id = generation["request_id"]
        logger.info("Generation request submitted",
                    generation_id=generation_id)

        # Process the generation results
        return process_generation_results(
            dm=dm,
            generation_id=generation_id,
            wait_time=args.wait_time,
            output=args.output,
            format=args.format
        )
    except Exception as e:
        logger.error("Failed to generate QA pairs", error=str(e))
        raise


if __name__ == "__main__":
    try:
        df = main()
        logger.info("Script executed successfully")
    except Exception as e:
        logger.error("Script execution failed", error=str(e), exc_info=True)
        raise e

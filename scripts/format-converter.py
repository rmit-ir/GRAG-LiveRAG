#!/usr/bin/env python
"""
Script to convert between different file formats (tsv, csv, jsonl, xlsx).

This script:
1. Takes a file (tsv, csv, jsonl, or xlsx) as input
2. Detects the format from the file name
3. Converts it to a specified target format
4. Outputs the file to the same location with the format suffix changed
5. Displays the input and output file locations to the user

Usage:
    uv run python scripts/format-converter.py --input data/file.tsv --format csv
    uv run python scripts/format-converter.py --input data/file.xlsx --format jsonl
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Import project modules
from utils.logging_utils import get_logger

# Initialize logger
logger = get_logger("format-converter")


def detect_format(file_path):
    """
    Detect the format of a file based on its extension.

    Args:
        file_path: Path to the input file

    Returns:
        The detected format as a string ('tsv', 'csv', 'jsonl', or 'xlsx')

    Raises:
        ValueError: If the file format is not supported
    """
    extension = Path(file_path).suffix.lower()

    if extension == '.tsv':
        return 'tsv'
    elif extension == '.csv':
        return 'csv'
    elif extension in ['.jsonl', '.json']:
        return 'jsonl'
    elif extension == '.xlsx':
        return 'xlsx'
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. Supported formats are .tsv, .csv, .jsonl, and .xlsx")


def load_file(file_path, input_format):
    """
    Load a file into a pandas DataFrame based on its format.

    Args:
        file_path: Path to the input file
        input_format: Format of the input file ('tsv', 'csv', 'jsonl', or 'xlsx')

    Returns:
        A pandas DataFrame containing the file data
    """
    logger.info(f"Loading {input_format} file", file=file_path)

    try:
        if input_format == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif input_format == 'csv':
            df = pd.read_csv(file_path)
        elif input_format == 'jsonl':
            df = pd.read_json(file_path, lines=True)
        elif input_format == 'xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        logger.info(f"Successfully loaded file", file=file_path,
                    rows=len(df), columns=len(df.columns))
        return df

    except Exception as e:
        logger.error(f"Error loading file", file=file_path, error=str(e))
        raise


def save_file(df, output_path, output_format):
    """
    Save a pandas DataFrame to a file in the specified format.

    Args:
        df: The pandas DataFrame to save
        output_path: Path to the output file
        output_format: Format to save the file in ('tsv', 'csv', 'jsonl', or 'xlsx')
    """
    logger.info(f"Saving to {output_format} format", file=output_path)

    try:
        if output_format == 'tsv':
            df.to_csv(output_path, sep='\t', index=False)
        elif output_format == 'csv':
            df.to_csv(output_path, index=False)
        elif output_format == 'jsonl':
            df.to_json(output_path, orient='records', lines=True)
        elif output_format == 'xlsx':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        logger.info(f"Successfully saved file", file=output_path)

    except Exception as e:
        logger.error(f"Error saving file", file=output_path, error=str(e))
        raise


def convert_file(input_path, output_format=None):
    """
    Convert a file from one format to another.

    Args:
        input_path: Path to the input file
        output_format: Format to convert to ('tsv', 'csv', 'jsonl', or 'xlsx')
                      If None, will be prompted from user

    Returns:
        Path to the output file
    """
    # Detect input format
    input_format = detect_format(input_path)

    # If output format is not specified, prompt user
    if output_format is None:
        valid_formats = ['tsv', 'csv', 'jsonl', 'xlsx']
        print(f"Current format: {input_format}")
        print(f"Available output formats: {', '.join(valid_formats)}")
        output_format = input("Enter output format: ").lower().strip()

        if output_format not in valid_formats:
            raise ValueError(
                f"Unsupported output format: {output_format}. Supported formats are {', '.join(valid_formats)}")

    # Skip if input and output formats are the same
    if input_format == output_format:
        logger.info(
            "Input and output formats are the same, no conversion needed")
        return input_path

    # Generate output path
    input_path_obj = Path(input_path)
    output_path = str(input_path_obj.with_suffix(f".{output_format}"))

    # Load the input file
    df = load_file(input_path, input_format)

    # Save to the output format
    save_file(df, output_path, output_format)

    return output_path


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert between different file formats (tsv, csv, jsonl, xlsx)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input file')
    parser.add_argument('--format', type=str, required=False,
                        choices=['tsv', 'csv', 'jsonl', 'xlsx'],
                        help='Format to convert to (tsv, csv, jsonl, or xlsx)')

    args = parser.parse_args()

    try:
        logger.info("Starting file format conversion")

        # Convert the file
        output_path = convert_file(args.input, args.format)

        # Display results to user
        print("\nConversion completed successfully!")
        print(f"Input file: {args.input}")
        print(f"Output file: {output_path}")

        logger.info("Conversion completed successfully")

    except Exception as e:
        logger.error("Conversion failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

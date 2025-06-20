#!/usr/bin/env python
"""
Script to evaluate RAG system results against reference answers.

This script:
1. Takes a RAG system result file (TSV or JSON)
2. Takes a reference dataset with QA pairs
3. Runs the specified evaluators on each result
4. Saves the evaluation results to a file
"""
from datetime import datetime
import os
import sys
import json
import time
import argparse
import importlib
import inspect
import ast
import pandas as pd
from typing import List, Dict, Any, Type
from dotenv import load_dotenv

from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir
from systems.rag_result import RAGResult
from services.ds_data_morgana import QAPair, CategoryDict
from evaluators.evaluator_interface import EvaluatorInterface
from evaluators.evaluation_result import EvaluationResult
from utils.time_utils import to_sec


logger = get_logger("evaluate")
# Load environment variables from .env file
load_dotenv()


def extract_evaluator_parameters(evaluator_class: Type[EvaluatorInterface]) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter information from an evaluator class's __init__ method.

    This function inspects the __init__ method signature and docstring to gather
    information about each parameter, including type, default value, and description.

    Args:
        evaluator_class: The evaluator class to inspect

    Returns:
        Dictionary mapping parameter names to their type, default value, and description
    """
    # Get the signature of the __init__ method
    signature = inspect.signature(evaluator_class.__init__)

    # Get the docstring of the __init__ method
    docstring = evaluator_class.__init__.__doc__ or ""

    # Parse parameter descriptions from docstring
    param_descriptions = {}
    current_param = None
    for line in docstring.split('\n'):
        line = line.strip()
        if line.startswith('Args:'):
            continue

        # Check for parameter documentation in the format "param_name: Description"
        if ': ' in line and not line.startswith(' '):
            parts = line.split(': ', 1)
            if len(parts) == 2:
                current_param = parts[0].strip()
                param_descriptions[current_param] = parts[1].strip()
        # Continue description for the current parameter
        elif current_param and line:
            param_descriptions[current_param] += ' ' + line

    # Extract parameter information
    params = {}
    for name, param in signature.parameters.items():
        # Skip 'self' parameter
        if name == 'self':
            continue

        # Get parameter type annotation
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else None

        # Get default value
        default = param.default if param.default != inspect.Parameter.empty else None

        # Get description from docstring or use parameter name if no description is available
        description = param_descriptions.get(name, f"Parameter '{name}'")

        params[name] = {
            'type': param_type,
            'default': default,
            'description': description
        }

    logger.debug("Extracted evaluator parameters",
                 evaluator_class=evaluator_class.__name__,
                 param_count=len(params))

    return params


def get_evaluator_params_from_args(evaluator_class: Type[EvaluatorInterface], args) -> Dict[str, Any]:
    """
    Extract evaluator parameters from command-line arguments.

    Args:
        evaluator_class: The evaluator class to extract parameters for
        args: The parsed command-line arguments

    Returns:
        Dictionary of parameter name to value mappings
    """
    evaluator_params = {}
    for name in extract_evaluator_parameters(evaluator_class).keys():
        # Check if the parameter exists in args
        if hasattr(args, name):
            value = getattr(args, name)
            # Only include non-None values to allow defaults to be used
            if value is not None:
                evaluator_params[name] = value

    logger.debug("Extracted evaluator parameters from args",
                 evaluator_class=evaluator_class.__name__,
                 param_count=len(evaluator_params))

    return evaluator_params


def find_evaluator_class_paths(class_name: str) -> List[str]:
    """
    Search recursively for Python files that contain a class with the given name
    and return the full module paths.

    Args:
        class_name: Name of the class to find

    Returns:
        List of module paths in the format 'evaluators.xxx.xxx.ClassName'
    """
    import os
    import re

    matches = []
    evaluators_dir = "src/evaluators"

    # Regular expression to find class definitions
    class_pattern = re.compile(
        rf"class\s+{class_name}\s*\(\s*.*EvaluatorInterface")

    # Walk through all directories under evaluators
    for root, _, files in os.walk(evaluators_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)

                # Read the file content and search for the class definition
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if class_pattern.search(content):
                            # Convert file path to module path
                            rel_path = os.path.relpath(file_path, "src")
                            module_path = rel_path.replace(
                                os.path.sep, '.').replace('.py', '')
                            # Create full module path with class name
                            full_path = f"{module_path}.{class_name}"
                            matches.append(full_path)
                except Exception:
                    # Skip files that can't be read
                    continue

    return matches


def load_evaluator_class(evaluator_class_path: str) -> Type[EvaluatorInterface]:
    """
    Dynamically import and return the specified evaluator class.

    Args:
        evaluator_class_path: Full path to the evaluator class 
                             (e.g., 'evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator')
                             or just the class name (e.g., 'EditDistanceEvaluator')

    Returns:
        The evaluator class

    Raises:
        ImportError: If the evaluator cannot be imported
        ValueError: If multiple matching classes are found for a class name
    """
    # Check if the path contains dots (indicating a module path)
    if '.' in evaluator_class_path:
        try:
            # Split the path into module path and class name
            module_path, class_name = evaluator_class_path.rsplit('.', 1)

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            evaluator_class = getattr(module, class_name)

            logger.info("Successfully loaded evaluator class",
                        module_path=module_path,
                        class_name=class_name)

            return evaluator_class
        except (ImportError, AttributeError) as e:
            logger.error("Failed to import evaluator class",
                         evaluator_class_path=evaluator_class_path,
                         error=str(e))
            raise ImportError(
                f"Could not import evaluator class '{evaluator_class_path}': {e}")
    else:
        # If only a class name is provided, search for it in the evaluators directory
        class_name = evaluator_class_path
        matching_paths = find_evaluator_class_paths(class_name)

        # Check the results of the search
        if not matching_paths:
            logger.error("No matching evaluator class found",
                         class_name=class_name)
            raise ImportError(
                f"Could not find any evaluator class named '{class_name}'")
        elif len(matching_paths) == 1:
            # Get the full module path and recursively call load_evaluator_class with it
            full_path = matching_paths[0]
            logger.info("Found matching evaluator class", full_path=full_path)

            # Recursively call load_evaluator_class with the full path
            return load_evaluator_class(full_path)
        else:
            # Multiple matches found, raise an error with details
            logger.error("Multiple matching evaluator classes found",
                         class_name=class_name,
                         matches=matching_paths)
            raise ValueError(
                f"Multiple evaluator classes named '{class_name}' found: {', '.join(matching_paths)}")


def load_rag_results(results_file: str) -> List[RAGResult]:
    """
    Load RAG results from a file (TSV or JSON).

    Args:
        results_file: Path to the results file

    Returns:
        List of RAGResult objects
    """
    results = []

    try:
        if results_file.endswith('.tsv'):
            # Load from TSV
            df = pd.read_csv(results_file, delimiter='\t')
            for _, row in df.iterrows():
                # Convert row to dict and handle JSON fields
                row_dict = row.to_dict()
                for field in ['context', 'doc_ids', 'generated_queries', 'rewritten_docs']:
                    if row_dict.get(field) and isinstance(row_dict[field], str):
                        try:
                            row_dict[field] = json.loads(row_dict[field])
                        except json.JSONDecodeError:
                            # If not valid JSON, keep as is
                            pass

                # Create RAGResult object
                results.append(RAGResult.from_dict(row_dict))

        elif results_file.endswith('.json') or results_file.endswith('.jsonl'):
            # Load from JSON or JSONL
            if results_file.endswith('.jsonl'):
                # JSONL format (one JSON object per line)
                with open(results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            results.append(
                                RAGResult.from_dict(json.loads(line)))
            else:
                # JSON format (single array or object)
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            results.append(RAGResult.from_dict(item))
                    else:
                        results.append(RAGResult.from_dict(data))

        else:
            raise ValueError(f"Unsupported file format: {results_file}")

        logger.info("Successfully loaded RAG results",
                    file=results_file,
                    result_count=len(results))

        return results

    except Exception as e:
        logger.error("Failed to load RAG results",
                     file=results_file,
                     error=str(e))
        raise


def load_reference_qa_pairs(reference_file: str) -> List[QAPair]:
    """
    Load reference QA pairs from a file.

    Args:
        reference_file: Path to the reference file

    Returns:
        List of QAPair objects
    """
    qa_pairs = []

    try:
        if reference_file.endswith('.tsv'):
            # Load from TSV
            df = pd.read_csv(reference_file, delimiter='\t')
            for _, row in df.iterrows():
                # Extract fields
                qid = row.get('qid', str(len(qa_pairs) + 1))
                question = row.get('question', '')
                answer = row.get('answer', '')

                # Handle context and document IDs
                context = row.get('context', [])
                if isinstance(context, str):
                    try:
                        context = json.loads(context)
                    except json.JSONDecodeError:
                        context = [context]

                doc_ids = row.get('document_ids', "[]")
                if isinstance(doc_ids, str):
                    try:
                        # First try json.loads
                        doc_ids = json.loads(doc_ids)
                    except json.JSONDecodeError:
                        # If that fails, try ast.literal_eval which can handle Python literals
                        try:
                            doc_ids = ast.literal_eval(doc_ids)
                        except (SyntaxError, ValueError):
                            # If both fail, treat as a single document ID
                            doc_ids = [doc_ids]

                # Extract question categories from column names
                question_categories = []
                user_categories = []

                for col_name, value in row.items():
                    if col_name.startswith('question_') and col_name != 'question_length' and not pd.isna(value):
                        categorization_name = col_name[len('question_'):]
                        question_categories.append(CategoryDict(
                            categorization_name=categorization_name,
                            category_name=value
                        ))
                    elif col_name.startswith('user_') and not pd.isna(value):
                        categorization_name = col_name[len('user_'):]
                        user_categories.append(CategoryDict(
                            categorization_name=categorization_name,
                            category_name=value
                        ))

                # Create QAPair object
                qa_pairs.append(QAPair(
                    question=question,
                    answer=answer,
                    context=context,
                    question_categories=question_categories,
                    user_categories=user_categories,
                    document_ids=doc_ids,
                    qid=qid
                ))

        elif reference_file.endswith('.json') or reference_file.endswith('.jsonl'):
            # Load from JSON or JSONL
            if reference_file.endswith('.jsonl'):
                # JSONL format (one JSON object per line)
                with open(reference_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            qa_pairs.append(QAPair.from_dict(data))
            else:
                # JSON format (single array or object)
                with open(reference_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            qa_pairs.append(QAPair.from_dict(item))
                    else:
                        qa_pairs.append(QAPair.from_dict(data))

        else:
            raise ValueError(f"Unsupported file format: {reference_file}")

        logger.info("Successfully loaded reference QA pairs",
                    file=reference_file,
                    qa_pair_count=len(qa_pairs))

        return qa_pairs

    except Exception as e:
        logger.error("Failed to load reference QA pairs",
                     file=reference_file,
                     error=str(e))
        raise


def save_evaluation_results(result: EvaluationResult, base_name: str, output_format: str = None, full_command: str = None) -> None:
    """
    Save evaluation results to separate files for aggregated results and individual rows.

    Args:
        result: EvaluationResult object
        base_name: Base name for output files (without extension)
        output_format: Format to save results in ('jsonl', 'tsv', or None to default to 'tsv')
        full_command: Full command string including all parameters and values
    """
    try:
        # Determine format if not specified
        if output_format is None:
            output_format = 'tsv'

        # Create filenames for aggregated results and rows
        ext = '.jsonl' if output_format == 'jsonl' else '.tsv'
        aggregated_file = f"{base_name}.aggregated{ext}"
        rows_file = f"{base_name}.rows{ext}"

        # Save aggregated results
        if output_format == 'jsonl':
            # Save aggregated results as JSONL (without rows)
            aggregated_dict = {
                "metrics": result.metrics,
                "evaluator_name": result.evaluator_name,
                "sample_count": result.sample_count,
                "system_name": result.system_name,
                "timestamp": result.timestamp.isoformat()
            }
            
            # Add full command if available
            if full_command:
                aggregated_dict["full_command"] = full_command

            if result.total_time_ms is not None:
                aggregated_dict["total_time_ms"] = result.total_time_ms

            if result.total_cost is not None:
                aggregated_dict["total_cost"] = result.total_cost

            if result.system_analysis is not None:
                aggregated_dict["system_analysis"] = result.system_analysis.to_dict(
                )

            with open(aggregated_file, 'w', encoding='utf-8') as f:
                f.write(json.dumps(aggregated_dict) + '\n')

            # Save rows as JSONL if available
            if result.rows:
                with open(rows_file, 'w', encoding='utf-8') as f:
                    for row in result.rows:
                        f.write(json.dumps(row.to_dict()) + '\n')

        elif output_format == 'tsv':
            # Save aggregated results as TSV
            agg_row = {
                "evaluator_name": result.evaluator_name,
                "sample_count": result.sample_count,
                "system_name": result.system_name,
                "timestamp": result.timestamp.isoformat(),
                "is_aggregated": True
            }
            
            # Add full command if available
            if full_command:
                agg_row["full_command"] = full_command

            if result.total_time_ms is not None:
                agg_row["total_time_ms"] = result.total_time_ms

            if result.total_cost is not None:
                agg_row["total_cost"] = result.total_cost

            if result.system_analysis is not None:
                agg_row["system_analysis"] = result.system_analysis.analysis

            # Flatten metrics into the row
            for metric_key, metric_value in result.metrics.items():
                agg_row[metric_key] = metric_value

            # Create DataFrame and save aggregated results to TSV
            agg_df = pd.DataFrame([agg_row])
            agg_df.to_csv(aggregated_file, sep='\t',
                          index=False, encoding='utf-8')

            # Save rows as TSV if available
            if result.rows:
                rows_data = []
                for row in result.rows:
                    row_dict = {
                        "qid": row.qid,
                        "evaluator_name": row.evaluator_name,
                        "is_aggregated": False
                    }

                    # Flatten metrics into the row
                    for metric_key, metric_value in row.metrics.items():
                        row_dict[metric_key] = metric_value

                    rows_data.append(row_dict)

                # Create DataFrame and save rows to TSV
                rows_df = pd.DataFrame(rows_data)
                rows_df.to_csv(rows_file, sep='\t',
                               index=False, encoding='utf-8')

        else:
            raise ValueError(
                f"Unsupported output format: {output_format}. Supported formats are 'jsonl' and 'tsv'.")

        logger.info("Successfully saved evaluation results",
                    aggregated_file=aggregated_file,
                    rows_file=rows_file if result.rows else None)

    except Exception as e:
        logger.error("Failed to save evaluation results",
                     base_name=base_name,
                     output_format=output_format,
                     error=str(e))
        raise


def create_parser_with_evaluator_params(evaluator_class=None):
    """
    Create an argument parser with evaluator-specific parameters if an evaluator class is provided.

    Args:
        evaluator_class: Optional evaluator class to extract parameters from

    Returns:
        An argparse.ArgumentParser instance
    """
    # Create description based on evaluator class if available
    if evaluator_class:
        description = (f'Evaluate RAG system results against reference answers using {evaluator_class.__name__}.\n\n'
                       f'Evaluator description: {evaluator_class.__doc__.strip() if evaluator_class.__doc__ else "No description available."}')
    else:
        description = 'Evaluate RAG system results against reference answers.'

    parser = argparse.ArgumentParser(description=description)

    # Add the evaluator argument
    parser.add_argument('--evaluator', type=str, default='evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator',
                        help='Evaluator class to use. Can be either a full path (e.g., evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator) '
                             'or just the class name (e.g., ContextRecall) (default: evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator)')

    # Add input and output arguments
    parser.add_argument('--results', type=str, required=True,
                        help='Path to the RAG system results file (TSV, JSON, or JSONL)')

    parser.add_argument('--reference', type=str, required=True,
                        help='Path to the reference QA pairs file (TSV, JSON, or JSONL)')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: data/evaluation_results)')

    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output filenames (default: evaluator name)')

    parser.add_argument('--output-format', type=str, choices=['jsonl', 'tsv'], default=None,
                        help='Format to save results in (jsonl or tsv, default: tsv).')

    # Add evaluator-specific parameters if an evaluator class is provided
    if evaluator_class:
        evaluator_params = extract_evaluator_parameters(evaluator_class)

        for name, param_info in evaluator_params.items():
            arg_name = f'--{name}'  # Use underscores in CLI arguments

            # Determine the type
            param_type = param_info['type']
            if param_type == bool:
                # For boolean parameters, create two flags: --param and --no-param
                parser.add_argument(arg_name, action='store_true', dest=name,
                                    help=f"{param_info['description']} (default: {param_info['default']})")
                parser.add_argument(f'--no-{name}', action='store_false', dest=name,
                                    help=f"Disable {name} (opposite of {arg_name})")
                parser.set_defaults(**{name: param_info['default']})
            else:
                # For other types, create a regular argument
                default_value = param_info['default']
                default_str = f"default: {default_value}" if default_value is not None else "default: None"

                parser.add_argument(arg_name, type=param_type, default=default_value,
                                    help=f"{param_info['description']} ({default_str})")

    return parser


def main():
    """Main entry point for the script."""
    # Start timing the entire process
    start_time = time.time()
    
    # Capture the full command
    full_command = " ".join(sys.argv)

    # First, create a basic parser to get the evaluator argument
    basic_parser = argparse.ArgumentParser(add_help=False)
    basic_parser.add_argument(
        '--evaluator', type=str, default='evaluators.basic_evaluator.edit_distance_evaluator.EditDistanceEvaluator')

    # Parse just the evaluator argument to get the evaluator class
    basic_args, _ = basic_parser.parse_known_args()

    # Load the evaluator class
    try:
        evaluator_class = load_evaluator_class(basic_args.evaluator)

        # Create a full parser with evaluator-specific parameters
        parser = create_parser_with_evaluator_params(evaluator_class)
    except Exception as e:
        # If there's an error loading the evaluator class, use the basic parser
        print(
            f"Warning: Could not load evaluator class '{basic_args.evaluator}': {e}")
        parser = create_parser_with_evaluator_params()

    # Parse all arguments
    args = parser.parse_args()

    # Set up output directory and filenames
    if args.output_dir is None:
        output_dir = os.path.join(get_data_dir(), "evaluation_results")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Set output prefix
    output_prefix = args.output_prefix or args.evaluator.split('.')[-1]

    # Define output filename based on results file name and evaluator
    results_base_name = os.path.basename(args.results)
    results_name_without_ext = os.path.splitext(results_base_name)[0]
    timestamp = datetime.now().strftime("%m%d%H%M")  # Format: MMDDHHMM
    base_name = os.path.join(
        output_dir, f"{results_name_without_ext}.eval{timestamp}.{output_prefix}")

    try:
        # Load the evaluator class
        evaluator_class = load_evaluator_class(args.evaluator)

        # Extract evaluator parameters from args
        evaluator_params = get_evaluator_params_from_args(
            evaluator_class, args)

        # Initialize the evaluator with parameters
        evaluator = evaluator_class(**evaluator_params)

        # Load RAG results
        rag_results = load_rag_results(args.results)

        # Load reference QA pairs
        reference_qa_pairs = load_reference_qa_pairs(args.reference)

        # Evaluate RAG results against reference QA pairs
        logger.info("Evaluating RAG results",
                    result_count=len(rag_results),
                    reference_count=len(reference_qa_pairs))

        evaluation_result = evaluator.evaluate(rag_results, reference_qa_pairs)

        # Save evaluation results with specified format
        save_evaluation_results(
            evaluation_result, base_name, args.output_format, full_command)

        # Determine file extensions for logging
        ext = '.jsonl' if args.output_format == 'jsonl' else '.tsv'
        aggregated_file = f"{base_name}.aggregated{ext}"
        rows_file = f"{base_name}.rows{ext}"

        logger.info("Evaluation complete",
                    evaluator=args.evaluator,
                    results_file=args.results,
                    reference_file=args.reference,
                    aggregated_file=aggregated_file,
                    metrics=evaluation_result.metrics,
                    system_name=evaluation_result.system_name,
                    evaluator_name=evaluation_result.evaluator_name,
                    sample_count=evaluation_result.sample_count,
                    total_time_ms=evaluation_result.total_time_ms,
                    total_cost=evaluation_result.total_cost,
                    rows_file=rows_file if evaluation_result.rows else None,
                    evaluated_count=evaluation_result.sample_count)

        # Calculate overall time
        overall_time_ms = (time.time() - start_time) * 1000

        print(f"\nEvaluation complete!")
        print(f"Total evaluation time: {to_sec(overall_time_ms)}s")
        print(
            f"Average time per query: {to_sec(overall_time_ms / evaluation_result.sample_count)}s")
        print(f"Results evaluated: {evaluation_result.sample_count}")
        if evaluation_result.total_time_ms is not None:
            print(
                f"Total real evaluation time: {to_sec(evaluation_result.total_time_ms)}s")
            print(
                f"Average real query eval time: {to_sec(evaluation_result.total_time_ms / evaluation_result.sample_count)}s")
        if evaluation_result.total_cost is not None:
            print(f"Total cost: ${evaluation_result.total_cost:.6f} USD")
            print(
                f"Average cost per query: ${evaluation_result.total_cost / evaluation_result.sample_count:.6f} USD")
        print(f"Output saved to:")
        print(f"  - Aggregated results: {aggregated_file}")
        if evaluation_result.rows:
            print(f"  - Row-level results: {rows_file}")

    except Exception as e:
        logger.error("Error in main execution", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

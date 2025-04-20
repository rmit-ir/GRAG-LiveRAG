#!/usr/bin/env python
"""
Script to run a specified RAG system on a dataset of questions and save the results.

This script:
1. Takes a system name and imports the corresponding class
2. Takes an input TSV file with questions
3. Runs all questions through the specified RAG system
4. Gathers answers
5. Saves results to a TSV file

When called with --help, it will dynamically show parameters from the specified system's __init__ method.
Example: uv run python scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --help
"""
import os
import sys
import csv
import time
import argparse
import importlib
import concurrent.futures
from typing import List, Dict, Any, Type, Optional, Tuple
import json

from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir
from utils.system_params import extract_system_parameters, get_system_params_from_args
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface


logger = get_logger("run")


def load_system_class(system_class_path: str) -> Type[RAGSystemInterface]:
    """
    Dynamically import and return the specified system class.
    
    Args:
        system_class_path: Full path to the system class (e.g., 'systems.basic_rag.basic_rag_system.BasicRAGSystem')
        
    Returns:
        The system class
    
    Raises:
        ImportError: If the system cannot be imported
    """
    try:
        # Split the path into module path and class name
        module_path, class_name = system_class_path.rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        system_class = getattr(module, class_name)
        
        logger.info("Successfully loaded system class", 
                   module_path=module_path, 
                   class_name=class_name)
        
        return system_class
    except (ImportError, AttributeError) as e:
        logger.error("Failed to import system class", 
                    system_class_path=system_class_path, 
                    error=str(e))
        raise ImportError(f"Could not import system class '{system_class_path}': {e}")


def load_questions_from_tsv(input_file: str) -> List[Dict[str, Any]]:
    """
    Load questions from a TSV file.
    
    Args:
        input_file: Path to the input TSV file
        
    Returns:
        List of dictionaries containing question data
    """
    questions = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                questions.append(row)
        
        logger.info("Successfully loaded questions from TSV", 
                   file=input_file, 
                   question_count=len(questions))
        
        return questions
    except Exception as e:
        logger.error("Failed to load questions from TSV", 
                    file=input_file, 
                    error=str(e))
        raise


def save_results_to_tsv(results: List[RAGResult], output_file: str) -> None:
    """
    Save RAG results to a TSV file.
    
    Args:
        results: List of RAGResult objects
        output_file: Path to the output TSV file
    """
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            # Define the fieldnames for the TSV
            fieldnames = [
                'qid', 'question', 'answer', 'context', 'doc_ids',
                'question_words_count', 'answer_words_count', 'total_time_ms',
                'timestamp', 'generated_queries', 'rewritten_docs', 'system_name'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for i, result in enumerate(results):
                # Convert RAGResult to dict and write to TSV
                result_dict = result.to_dict()
                
                # Use existing qid or generate one if not present
                if result_dict['qid'] is None:
                    result_dict['qid'] = str(i + 1)
                
                # Convert lists to string representation
                for field in ['context', 'doc_ids', 'generated_queries', 'rewritten_docs']:
                    if result_dict[field] is not None:
                        result_dict[field] = json.dumps(result_dict[field])
                
                writer.writerow(result_dict)
        
        logger.info("Successfully saved results to TSV", 
                   file=output_file, 
                   result_count=len(results))
    
    except Exception as e:
        logger.error("Failed to save results to TSV", 
                    file=output_file, 
                    error=str(e))
        raise


def create_trec_run_file(results: List[RAGResult], output_file: str, system_name: str) -> None:
    """
    Create a standard TREC Run file from RAG results.
    
    TREC Run format:
    qid Q0 docno rank score tag
    
    Args:
        results: List of RAGResult objects
        output_file: Path to the output TREC Run file
        system_name: Name of the system (used as the tag)
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                # Use existing qid or generate one if not present
                qid = result.qid if result.qid is not None else str(i + 1)
                
                # For each document in the result, write a line to the TREC Run file
                for rank, doc_id in enumerate(result.doc_ids, start=1):
                    # Use a simple scoring mechanism - documents earlier in the list get higher scores
                    score = 1.0 / rank
                    
                    # Write the line in TREC Run format
                    f.write(f"{qid} Q0 {doc_id} {rank} {score} {system_name}\n")
        
        logger.info("Successfully created TREC Run file", 
                   file=output_file, 
                   system_name=system_name)
    
    except Exception as e:
        logger.error("Failed to create TREC Run file", 
                    file=output_file, 
                    error=str(e))
        raise


def process_single_question(args: Tuple[Type[RAGSystemInterface], Dict[str, Any], Dict[str, Any], int, int]) -> Optional[RAGResult]:
    """
    Process a single question with the specified system.
    
    Args:
        args: Tuple containing:
            - system_class: The system class to instantiate
            - system_params: Parameters to pass to the system constructor
            - question_data: Dictionary containing question data
            - index: Index of the question in the original list
            - total_questions: Total number of questions
            
    Returns:
        RAGResult object or None if an error occurred
    """
    system_class, system_params, question_data, i, total_questions = args
    
    try:
        # Initialize a new system instance for each question when in parallel mode
        system = system_class(**system_params)
        
        # Extract the question text
        question = question_data['question']
        
        # Get qid from question data or generate one
        qid = question_data.get('qid', str(i + 1))
        
        # Process the question
        logger.info(f"Processing question {i+1}/{total_questions}", 
                   question=question,
                   qid=qid)
        
        result = system.process_question(question, qid=qid)
        
        logger.info(f"Completed question {i+1}/{total_questions}", 
                   question=question,
                   answer_length=result.answer_words_count,
                   processing_time_ms=result.total_time_ms)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing question {i+1}/{total_questions}", 
                    question=question_data.get('question', 'Unknown'),
                    error=str(e))
        # Return None to indicate an error
        return None


def run_system(system_class: Type[RAGSystemInterface], questions: List[Dict[str, Any]], 
               system_params: Optional[Dict[str, Any]] = None, 
               num_threads: int = 1) -> List[RAGResult]:
    """
    Run the specified system on a list of questions.
    
    Args:
        system_class: The system class to instantiate
        questions: List of dictionaries containing question data
        system_params: Optional parameters to pass to the system constructor
        num_threads: Number of threads to use for parallel processing
        
    Returns:
        List of RAGResult objects
    """
    system_params = system_params or {}
    total_questions = len(questions)
    
    # Start timing the entire process
    start_time = time.time()
    
    # Determine if we're using parallel processing based on thread count
    parallel = num_threads > 1
    
    logger.info("Starting to process questions", 
               total_questions=total_questions,
               system_class=system_class.__name__,
               parallel=parallel,
               num_threads=num_threads)
    
    results = []
    
    if num_threads <= 1:
        # Sequential processing
        system = system_class(**system_params)
        
        for i, question_data in enumerate(questions):
            try:
                # Extract the question text
                question = question_data['question']
                
                # Get qid from question data or generate one
                qid = question_data.get('qid', str(i + 1))
                
                # Process the question
                logger.info(f"Processing question {i+1}/{total_questions}", 
                           question=question,
                           qid=qid)
                
                result = system.process_question(question, qid=qid)
                results.append(result)
                
                logger.info(f"Completed question {i+1}/{total_questions}", 
                           question=question,
                           answer_length=result.answer_words_count,
                           processing_time_ms=result.total_time_ms)
            
            except Exception as e:
                logger.error(f"Error processing question {i+1}/{total_questions}", 
                            question=question_data.get('question', 'Unknown'),
                            error=str(e))
                # Continue with the next question
    else:
        # Parallel processing
        logger.info(f"Using ThreadPoolExecutor with {num_threads} threads")
        
        # Prepare arguments for each question
        question_args = [
            (system_class, system_params, question_data, i, total_questions)
            for i, question_data in enumerate(questions)
        ]
        
        # Process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks and collect futures
            futures = [executor.submit(process_single_question, args) for args in question_args]
            
            # Collect results in order
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
    
    # Sort results by qid to maintain order
    results.sort(key=lambda r: int(r.qid) if r.qid and r.qid.isdigit() else float('inf'))
    
    # Calculate total processing time
    total_time_ms = (time.time() - start_time) * 1000
    avg_time_per_question_ms = total_time_ms / len(results) if results else 0
    
    # Sum up individual question processing times
    sum_individual_times_ms = sum(result.total_time_ms for result in results)
    avg_individual_time_ms = sum_individual_times_ms / len(results) if results else 0
    
    logger.info("Finished processing all questions", 
               total_processed=len(results),
               total_questions=total_questions,
               total_time_ms=total_time_ms,
               avg_time_per_question_ms=avg_time_per_question_ms,
               sum_individual_times_ms=sum_individual_times_ms,
               avg_individual_time_ms=avg_individual_time_ms)
    
    return results



def create_parser_with_system_params(system_class=None):
    """
    Create an argument parser with system-specific parameters if a system class is provided.
    
    Args:
        system_class: Optional system class to extract parameters from
        
    Returns:
        An argparse.ArgumentParser instance
    """
    # Create description based on system class if available
    if system_class:
        description = (f'Run the {system_class.__name__} system on a dataset of questions.\n\n'
                      f'System description: {system_class.__doc__.strip() if system_class.__doc__ else "No description available."}')
    else:
        description = 'Run a RAG system on a dataset of questions.'
    
    parser = argparse.ArgumentParser(description=description)
    
    # Add the system argument
    parser.add_argument('--system', type=str, required=True,
                        help='Full path to the system class (e.g., systems.basic_rag.basic_rag_system.BasicRAGSystem)')
    
    # Add input and output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input TSV file with questions')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: data/rag_results)')
    
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output filenames (default: system name)')
    
    # Add thread count argument for parallel processing
    parser.add_argument('--num-threads', type=int, default=1,
                        help='Number of threads to use for processing (default: 1, values > 1 enable parallel processing)')
    
    # Add system-specific parameters if a system class is provided
    if system_class:
        system_params = extract_system_parameters(system_class)
        
        for name, param_info in system_params.items():
            arg_name = f'--{name}'  # Use underscores in CLI arguments
            
            # Determine the type
            param_type = param_info['type']
            if param_type == bool:
                parser.add_argument(arg_name, action='store_true',
                                    help=f"{param_info['description']} (default: {param_info['default']})")
            else:
                # Convert type annotation to actual type if needed
                if hasattr(param_type, '__origin__') and param_type.__origin__ is Optional:
                    # Handle Optional[Type]
                    param_type = param_type.__args__[0]
                
                # Use str as default type if type is None or complex
                if param_type in (None, Any) or hasattr(param_type, '__origin__'):
                    param_type = str
                
                default_value = param_info['default']
                default_str = f"default: {default_value}" if default_value is not None else "default: None"
                
                parser.add_argument(arg_name, type=param_type, default=default_value,
                                    help=f"{param_info['description']} ({default_str})")
    
    return parser


def main():
    """Main entry point for the script."""
    start_time = time.time()
    # First, create a parser with just the system argument
    parser = create_parser_with_system_params()
    
    # Check if --help is in the arguments
    if '--help' in sys.argv:
        if '--system' in sys.argv:
            # Get the system class path
            system_idx = sys.argv.index('--system')
            if system_idx + 1 < len(sys.argv):
                system_class_path = sys.argv[system_idx + 1]
                try:
                    # Load the system class
                    system_class = load_system_class(system_class_path)
                    
                    # Create a new parser with system-specific parameters
                    parser = create_parser_with_system_params(system_class)
                except Exception as e:
                    # If there's an error loading the system class, continue with the basic parser
                    print(f"Warning: Could not load system class '{system_class_path}': {e}")
        else:
            # Remind user to pass --system to see system-specific parameters
            print("\nNOTE: To view system-specific parameters, please pass the --system argument.")
            print("Example: uv run python scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --help\n")
    
    args = parser.parse_args()
    
    # Set up output directory and filenames
    if args.output_dir is None:
        output_dir = os.path.join(get_data_dir(), "rag_results")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output prefix
    ds_name = args.input.split('/')[-1].split('.')[0]
    output_prefix = args.output_prefix or args.system.split('.')[-1]
    
    # Define output filenames
    tsv_filename = f"{ds_name}_{output_prefix}.tsv"
    trec_filename = f"{ds_name}_{output_prefix}.trec"
    
    tsv_output_path = os.path.join(output_dir, tsv_filename)
    trec_output_path = os.path.join(output_dir, trec_filename)
    
    try:
        # Load the system class
        system_class = load_system_class(args.system)
        
        # Load questions from TSV
        questions = load_questions_from_tsv(args.input)
        
        # Extract system parameters from args
        system_params = get_system_params_from_args(system_class, args)
        
        # Run the system on the questions
        results = run_system(system_class, questions, system_params, 
                            num_threads=args.num_threads)
        overall_time_ms = (time.time() - start_time) * 1000
        
        # Calculate timing information
        total_time_ms = sum(result.total_time_ms for result in results)
        
        # Save results to TSV
        save_results_to_tsv(results, tsv_output_path)
        
        # Create TREC Run file
        create_trec_run_file(results, trec_output_path, args.system)
        
        logger.info("Successfully completed the run", 
                   system=args.system,
                   input_file=args.input,
                   tsv_output=tsv_output_path,
                   trec_output=trec_output_path)
        
        print(f"\nProcessing complete!")
        print(f"Total time: {overall_time_ms:.2f} ms ({overall_time_ms/1000:.2f} s)")
        print(f"Question time: {total_time_ms:.2f} ms ({total_time_ms/1000:.2f} s)")
        print(f"Questions processed: {len(results)}")
        print(f"\nResults saved to:")
        print(f"  - TSV: {tsv_output_path}")
        print(f"  - TREC Run: {trec_output_path}")
    
    except Exception as e:
        logger.error("Error in main execution", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

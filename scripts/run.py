#!/usr/bin/env python
"""
Script to run a specified RAG system on a dataset of questions and save the results.

This script:
1. Takes a system name and imports the corresponding class
2. Takes an input file with questions (TSV or JSONL in LiveRAG format)
3. Runs all questions through the specified RAG system
4. Gathers answers
5. Saves results to a TSV file or JSONL file in LiveRAG format

When called with --help, it will dynamically show parameters from the specified system's __init__ method.
Example: uv run python scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --help

For LiveRAG Challenge format:
Example: uv run python scripts/run.py --system systems.basic_rag.basic_rag_system.BasicRAGSystem --input questions.jsonl --live
"""
import os
import sys
import csv
import time
import argparse
import importlib
import concurrent.futures
from typing import List, Dict, Any, Type, Optional, Tuple, Union, TypedDict
import json
import jsonlines

from utils.logging_utils import get_logger
from utils.path_utils import get_data_dir
from utils.system_params import extract_system_parameters, get_system_params_from_args
from systems.rag_result import RAGResult
from systems.rag_system_interface import RAGSystemInterface
from systems.live_rag_io import LiveRAGQuestion, LiveRAGAnswer
from utils.time_utils import to_sec


class QuestionData(TypedDict):
    """
    Common TypedDict for question data from both TSV and LiveRAG formats.
    """
    question: str
    qid: str


logger = get_logger("run")


def find_system_class_paths(class_name: str) -> List[str]:
    """
    Search recursively for Python files that contain a class with the given name
    and return the full module paths.
    
    Args:
        class_name: Name of the class to find
        
    Returns:
        List of module paths in the format 'systems.xxx.xxx.ClassName'
    """
    import os
    import re
    
    matches = []
    systems_dir = "src/systems"
    
    # Regular expression to find class definitions
    class_pattern = re.compile(rf"class\s+{class_name}\s*\(\s*.*RAGSystemInterface")
    
    # Walk through all directories under systems
    for root, _, files in os.walk(systems_dir):
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
                            module_path = rel_path.replace(os.path.sep, '.').replace('.py', '')
                            # Create full module path with class name
                            full_path = f"{module_path}.{class_name}"
                            matches.append(full_path)
                except Exception:
                    # Skip files that can't be read
                    continue
    
    return matches


def load_system_class(system_class_path: str) -> Type[RAGSystemInterface]:
    """
    Dynamically import and return the specified system class.
    
    Args:
        system_class_path: Full path to the system class (e.g., 'systems.basic_rag.basic_rag_system.BasicRAGSystem')
                          or just the class name (e.g., 'VectorRAG')
        
    Returns:
        The system class
    
    Raises:
        ImportError: If the system cannot be imported
        ValueError: If multiple matching classes are found for a class name
    """
    # Check if the path contains dots (indicating a module path)
    if '.' in system_class_path:
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
    else:
        # If only a class name is provided, search for it in the systems directory
        class_name = system_class_path
        matching_paths = find_system_class_paths(class_name)
        
        # Check the results of the search
        if not matching_paths:
            logger.error("No matching system class found", class_name=class_name)
            raise ImportError(f"Could not find any system class named '{class_name}'")
        elif len(matching_paths) == 1:
            # Get the full module path and recursively call load_system_class with it
            full_path = matching_paths[0]
            logger.info("Found matching system class", full_path=full_path)
            
            # Recursively call load_system_class with the full path
            return load_system_class(full_path)
        else:
            # Multiple matches found, raise an error with details
            logger.error("Multiple matching system classes found", 
                        class_name=class_name, 
                        matches=matching_paths)
            raise ValueError(f"Multiple system classes named '{class_name}' found: {', '.join(matching_paths)}")


def load_questions_from_tsv(input_file: str) -> List[QuestionData]:
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
                # Convert to QuestionData format
                question_data: QuestionData = {
                    'question': row['question'],
                    'qid': row.get('qid', '')
                }
                questions.append(question_data)
        
        logger.info("Successfully loaded questions from TSV", 
                   file=input_file, 
                   question_count=len(questions))
        
        return questions
    except Exception as e:
        logger.error("Failed to load questions from TSV", 
                    file=input_file, 
                    error=str(e))
        raise


def load_live_rag_questions(input_file: str) -> List[QuestionData]:
    """
    Load questions from a JSONL file in LiveRAG Challenge format.
    
    Args:
        input_file: Path to the input JSONL file
        
    Returns:
        List of dictionaries containing question data
    """
    questions: List[QuestionData] = []
    
    try:
        with jsonlines.open(input_file, 'r') as reader:
            for obj in reader:
                # Check if required fields are present
                if 'question' not in obj or 'id' not in obj:
                    logger.warning("Skipping invalid question object, missing required fields", 
                                  obj=obj)
                    continue
                
                # Convert to QuestionData format
                question_data: QuestionData = {
                    'question': obj['question'],
                    'qid': str(obj['id']),
                }
                questions.append(question_data)
        
        logger.info("Successfully loaded questions from JSONL", 
                   file=input_file, 
                   question_count=len(questions))
        
        return questions
    except Exception as e:
        logger.error("Failed to load questions from JSONL", 
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
                'timestamp', 'generated_queries', 'rewritten_docs', 'system_name',
                'metadata'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for i, result in enumerate(results):
                # Convert RAGResult to dict and write to TSV
                result_dict = result.to_dict()
                
                # Use existing qid or generate one if not present
                if result_dict['qid'] is None:
                    result_dict['qid'] = str(i + 1)
                
                # Convert lists and dictionaries to string representation
                for field in ['context', 'doc_ids', 'generated_queries', 'rewritten_docs', 'metadata']:
                    if result_dict.get(field) is not None:
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


def save_live_rag_results(results: List[RAGResult], output_file: str) -> None:
    """
    Save RAG results to a JSONL file in LiveRAG Challenge format.
    
    Args:
        results: List of RAGResult objects
        output_file: Path to the output JSONL file
    """
    try:
        with jsonlines.open(output_file, 'w') as writer:
            for result in results:
                # Convert RAGResult to LiveRAGAnswer
                live_answer = LiveRAGAnswer.from_rag_result(result)
                # Write the answer as a dictionary
                writer.write(live_answer.to_dict())
        
        logger.info("Successfully saved results to JSONL", 
                   file=output_file, 
                   result_count=len(results))
    
    except Exception as e:
        logger.error("Failed to save results to JSONL", 
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


def process_single_question(args: Tuple[Type[RAGSystemInterface], Dict[str, Any], QuestionData, int, int]) -> Optional[RAGResult]:
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
                   processing_time_sec=to_sec(result.total_time_ms))
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing question {i+1}/{total_questions}", 
                    question=question_data.get('question', 'Unknown'),
                    error=str(e))
        # Return None to indicate an error
        return None


def run_system(system_class: Type[RAGSystemInterface], questions: List[QuestionData], 
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
                           processing_time_sec=to_sec(result.total_time_ms))
            
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
               total_time_sec=to_sec(total_time_ms),
               avg_time_per_question_sec=to_sec(avg_time_per_question_ms),
               sum_individual_times_sec=to_sec(sum_individual_times_ms),
               avg_individual_time_sec=to_sec(avg_individual_time_ms))
    
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
                        help='System class to use. Can be either a full path (e.g., systems.basic_rag.basic_rag_system.BasicRAGSystem) '
                             'or just the class name (e.g., VectorRAG)')
    
    # Add input and output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input file with questions (TSV or JSONL in LiveRAG format)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files (default: data/rag_results)')
    
    parser.add_argument('--output-prefix', type=str, default=None,
                        help='Prefix for output filenames (default: system name)')
    
    # Add LiveRAG Challenge mode flag
    parser.add_argument('--live', action='store_true',
                        help='Enable LiveRAG Challenge mode (input and output in LiveRAG format) default: False)')
    
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
    
    # First, create a basic parser to get the evaluator argument
    basic_parser = argparse.ArgumentParser(add_help=False)
    basic_parser.add_argument('--system', type=str, default='systems.basic_rag.basic_rag_system.BasicRAGSystem')
    
    # Parse just the evaluator argument to get the evaluator class
    basic_args, _ = basic_parser.parse_known_args()

    try:
        # Load the system class
        system_class = load_system_class(basic_args.system)
        parser = create_parser_with_system_params(system_class)
    except Exception as e:
        # If there's an error loading the system class, continue with the basic parser
        print(f"Warning: Could not load system class '{basic_args.system}': {e}")
        parser = create_parser_with_system_params()

    # Parse all arguments
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
    ds_name = args.input.split('/')[-1].split('.')[0]
    output_prefix = args.output_prefix or args.system.split('.')[-1]
    
    # Standard format uses TSV and TREC
    tsv_filename = f"{ds_name}_{output_prefix}.tsv"
    # trec_filename = f"{ds_name}_{output_prefix}.trec"
    
    tsv_output_path = os.path.join(output_dir, tsv_filename)
    # trec_output_path = os.path.join(output_dir, trec_filename)
    if args.live:
        # LiveRAG Challenge format uses JSONL
        live_output_filename = f"{ds_name}_{output_prefix}.jsonl"
        live_output_path = os.path.join(output_dir, live_output_filename)
    
    try:
        # Load the system class
        system_class = load_system_class(args.system)
        
        # Load questions based on format
        if args.live:
            questions = load_live_rag_questions(args.input)
        else:
            questions = load_questions_from_tsv(args.input)
        
        # Extract system parameters from args
        system_params = get_system_params_from_args(system_class, args)
        
        # Run the system on the questions
        results = run_system(system_class, questions, system_params, 
                            num_threads=args.num_threads)
        overall_time_ms = (time.time() - start_time) * 1000
        
        # Calculate timing information
        total_time_ms = sum(result.total_time_ms for result in results)
        
        # Save in standard format (TSV and TREC)
        save_results_to_tsv(results, tsv_output_path)
        # create_trec_run_file(results, trec_output_path, args.system)
        
        logger.info("Successfully completed the run", 
                    system=args.system,
                    input_file=args.input,
                    tsv_output=tsv_output_path)
        print(f"\nProcessing complete!")
        print(f"Total running time: {to_sec(overall_time_ms)}s")
        print(f"Average time per question: {to_sec(overall_time_ms/len(results))}s")
        print(f"Total real question time: {to_sec(total_time_ms)}s")
        print(f"Real time per question: {to_sec(total_time_ms/len(results))}s")
        print(f"Questions processed: {len(results)}")
        print(f"\nResults saved to:")
        print(f"  - TSV: {tsv_output_path}")
        # print(f"  - TREC Run: {trec_output_path}")
        if args.live:
            # Save in LiveRAG Challenge format (JSONL)
            save_live_rag_results(results, live_output_path)
            print(f"\nLiveRAG Results saved to:")
            print(f"  - JSONL (LiveRAG format): {live_output_path}")
    
    except Exception as e:
        logger.error("Error in main execution", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

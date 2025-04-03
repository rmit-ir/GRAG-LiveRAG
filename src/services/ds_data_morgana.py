"""
DataMorgana API for synthetic conversation generation.

DataMorgana is a tool for generating highly customizable and diverse synthetic Q&A 
benchmarks tailored to RAG applications. It enables detailed configurations of user 
and question categories and provides control over their distribution within the benchmark.
"""
import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Any, Union


class DataMorgana:
    """
    Client for the DataMorgana API, which provides synthetic conversation generation
    capabilities for question-answer pairs based on specified categories and documents.

    Ideal Usage Examples
    ```python
    # Initialize DataMorgana client
    dm = DataMorgana()  # Uses DATAMORGANA_API_KEY environment variable
    # or
    dm = DataMorgana(api_key="your-api-key")

    # Generate a single QA pair synchronously
    qa_pair = dm.generate_qa_pair_sync(
        question_categories={...},
        user_categories={...}
    )

    # Generate bulk QA pairs asynchronously
    generation = dm.generate_qa_pair_bulk(
        n_questions=10,
        question_categorizations=[...],
        user_categorizations=[...],
    )

    # Get the generation ID
    generation_id = generation["generation_id"]

    # Wait for and retrieve generation results
    results = dm.wait_generation_results(generation_id)

    # Access the generated QA pairs
    qa_pairs = results.get("qa_pairs", [])
    for qa in qa_pairs:
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer']}")
    ```
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataMorgana client with API configuration.

        Args:
            api_key: DataMorgana API key (defaults to DATAMORGANA_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get('DATAMORGANA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DATAMORGANA_API_KEY environment variable is required")

        self.base_url = "https://api.ai71.ai/v1"

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_qa_pair_sync(
        self,
        question_categories: Dict[str, Dict[str, str]],
        user_categories: Optional[Dict[str, Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a question-answer pair synchronously.

        Args:
            question_categories: Dictionary of question categories
                Example: {
                    "formulation-categorization": {
                        "name": "natural",
                        "description": "phrased in the way people typically speak"
                        # no probability, that is for bulk generation
                    }
                }
            user_categories: Dictionary of end-user categories
                Example: {
                    "expertise-categorization": {
                        "name": "expert",
                        "description": "a specialized user with deep understanding"
                        # no probability, that is for bulk generation
                    }
                }
            document_ids: List of document IDs to use for generation

        Returns:
            Generated QA pair response
        """
        url = f"{self.base_url}/generate_sync_qa_pair"

        payload = {
            "question_categories": question_categories
        }

        if user_categories:
            payload["user_categories"] = user_categories

        if document_ids:
            payload["document_ids"] = document_ids

        response = requests.post(
            url, headers=self._get_headers(), json=payload)
        response.raise_for_status()

        return response.json()

    def generate_qa_pair_bulk(
        self,
        n_questions: int,
        question_categorizations: List[Dict[str, Any]],
        user_categorizations: Optional[List[Dict[str, Any]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Submit a bulk generation request asynchronously.

        Args:
            n_questions: Number of QA pairs to generate
            question_categorizations: List of question categorization configurations
                Example: [
                    {
                        "categorization_name": "factuality",
                        "categories": [
                            {
                                "name": "factoid",
                                "description": "question seeking a specific fact",
                                "probability": 0.5
                            }
                        ]
                    }
                ]
            user_categorizations: List of user categorization configurations
            document_ids: List of document IDs to use for generation

        Returns:
            Response containing generation ID for status tracking
        """
        url = f"{self.base_url}/bulk_generation"

        payload = {
            "n_questions": n_questions,
            "question_categorizations": question_categorizations
        }

        if user_categorizations:
            payload["user_categorizations"] = user_categorizations

        if document_ids:
            payload["document_ids"] = document_ids

        response = requests.post(
            url, headers=self._get_headers(), json=payload)
        response.raise_for_status()

        return response.json()

    def fetch_generation_results(self, generation_id: str) -> Dict[str, Any]:
        """
        Fetch results of a previously submitted bulk generation request.

        Args:
            generation_id: ID of the generation request

        Returns:
            A dictionary containing:
                - 'status': Status of the generation ('completed', 'in_progress', 'failed')
                - 'file': URL to the JSONL file containing QA pairs (when completed)
                - 'credits': Number of credits used
        """
        url = f"{self.base_url}/fetch_generation_results?request_id={generation_id}"

        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()

        return response.json()

    def parse_qa_pairs(self, file_path_or_url: str) -> List[Dict[str, Any]]:
        """
        Parse QA pairs from a JSONL file, which can be either a local file path or a URL.

        Args:
            file_path_or_url: Path to local JSONL file or URL to remote JSONL file

        Returns:
            List of parsed QA pairs

        Examples:
            ```python
            # From a URL (e.g., from fetch_generation_results)
            qa_pairs = dm.parse_qa_pairs("https://example.com/results.jsonl")

            # From a local file
            qa_pairs = dm.parse_qa_pairs("/path/to/local/file.jsonl")
            ```
        """
        # Check if input is a URL or local file path
        if file_path_or_url.startswith(('http://', 'https://')):
            # Handle URL: Download and parse
            response = requests.get(file_path_or_url)
            response.raise_for_status()
            content = response.text
        else:
            # Handle local file path
            try:
                with open(file_path_or_url, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"JSONL file not found: {file_path_or_url}")
            except PermissionError:
                raise PermissionError(
                    f"Permission denied when trying to read: {file_path_or_url}")
            except Exception as e:
                raise Exception(f"Error reading JSONL file: {str(e)}")

        # Parse JSONL content (each line is a separate JSON object)
        qa_pairs = []
        for line in content.strip().split('\n'):
            if line:
                try:
                    qa_pairs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {str(e)}")
                    continue

        return qa_pairs

    def retry_generation(self, generation_id: str) -> Dict[str, Any]:
        """
        Retry a failed bulk generation request.

        Args:
            generation_id: ID of the failed generation request

        Returns:
            Response containing status of the retry request
        """
        url = f"{self.base_url}/retry"

        payload = {
            "generation_id": generation_id
        }

        response = requests.post(
            url, headers=self._get_headers(), json=payload)
        response.raise_for_status()

        return response.json()

    def wait_generation_results(self, generation_id: str, sleep_sec=2, parse_results=True) -> Dict[str, Any]:
        """
        Wait for generation results with polling.
        If the status is 'in_progress', retry every 2 seconds until completion.

        Args:
            generation_id: ID of the generation request
            sleep_sec: Time to wait between polling attempts in seconds
            parse_results: If True, automatically download and parse QA pairs when completed

        Returns:
            If parse_results is False: Status and raw results of the generation request
            If parse_results is True: Status, file URL, credits, and parsed QA pairs
        """
        import time

        while True:
            result = self.fetch_generation_results(generation_id)
            status = result.get('status', '').lower()

            if status != 'in_progress':
                print(f"Generation status: {status}")

                if status == 'completed' and parse_results and 'file' in result:
                    try:
                        qa_pairs = self.parse_qa_pairs(result['file'])
                        print(f"Retrieved {len(qa_pairs)} QA pairs")
                        result['qa_pairs'] = qa_pairs
                    except Exception as e:
                        print(f"Error parsing QA pairs: {str(e)}")

                return result

            print(
                f"Status: {status}, waiting {sleep_sec} seconds before retrying...")
            time.sleep(sleep_sec)

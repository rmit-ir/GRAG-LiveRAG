"""
DataMorgana API for synthetic conversation generation.

DataMorgana is a tool for generating highly customizable and diverse synthetic Q&A 
benchmarks tailored to RAG applications. It enables detailed configurations of user 
and question categories and provides control over their distribution within the benchmark.
"""
import os
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from typing import TypedDict
from utils.logging_utils import get_logger


class CategoryDict(TypedDict):
    categorization_name: str
    category_name: str


@dataclass
class QAPair:
    """
    Represents a question-answer pair with associated metadata.
    """
    question: str
    answer: str
    context: List[str]
    question_categories: List[CategoryDict]
    user_categories: List[CategoryDict]
    document_ids: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAPair':
        """
        Create a QAPair instance from a dictionary.
        """
        return cls(
            question=data.get('question', ''),
            answer=data.get('answer', ''),
            context=data.get('context', []),
            question_categories=data.get('question_categories', []),
            user_categories=data.get('user_categories', []),
            document_ids=data.get('document_ids', [])
        )


class DataMorgana:
    """
    Client for the DataMorgana API, which provides synthetic conversation generation
    capabilities for question-answer pairs based on specified categories and documents.

    Ideal Usage Examples
    ```python
    # Initialize DataMorgana client
    dm = DataMorgana()  # Uses AI71_API_KEY environment variable
    # or
    dm = DataMorgana(api_key="your-api-key")

    # Generate a single QA pair synchronously
    qa_pair = dm.generate_qa_pair_sync(
        question_categories={...},
        user_categories={...}
    )
    print(f"Question: {qa_pair.question}")
    print(f"Answer: {qa_pair.answer}")

    # Generate bulk QA pairs asynchronously
    generation = dm.generate_qa_pair_bulk(
        n_questions=10,
        question_categorizations=[...],
        user_categorizations=[...],
    )

    # Get the generation ID
    generation_id = generation["generation_id"]

    # Wait for and retrieve generation results
    qa_pairs = dm.wait_generation_results(generation_id)

    # Access the generated QA pairs
    for qa in qa_pairs:
        print(f"Question: {qa.question}")
        print(f"Answer: {qa.answer}")
    ```
    """
    
    log = get_logger("ds_data_morgana")

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataMorgana client with API configuration.

        Args:
            api_key: DataMorgana API key (defaults to AI71_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get('AI71_API_KEY')
        if not self.api_key:
            raise ValueError(
                "AI71_API_KEY environment variable is required")

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
    ) -> QAPair:
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
            Generated QAPair object
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

        result = response.json()
        
        # Extract the QA pair from the nested structure
        if "response" in result and "result" in result["response"] and len(result["response"]["result"]) > 0:
            qa_data = result["response"]["result"][0]
            return QAPair.from_dict(qa_data)
        else:
            raise ValueError(f"Unexpected response format: {result}")

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

    def save_generation_to_file(self, file_url: str, generation_id: str) -> str:
        """
        Save generation results from URL to a local file.
        
        Args:
            file_url: URL to the JSONL file containing QA pairs
            generation_id: ID of the generation request (used as fallback filename)
            
        Returns:
            Path where the file was saved
        """
        import os
        import urllib.parse
        from utils.path_utils import get_project_root
        
        # Extract the filename from URL
        filename = os.path.basename(urllib.parse.urlparse(file_url).path)
        
        # If filename is empty, use the generation ID
        if not filename or filename == '':
            filename = f"results_id_{generation_id}.jsonl"
        
        # Create save path using the project root
        project_root = get_project_root()
        save_dir = os.path.join(project_root, "data", "generated_qa_pairs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        # Read the data
        df = pd.read_json(file_url, lines=True)
        
        # Save raw generated qa_pairs to local file
        df.to_json(save_path, orient='records', lines=True)
        self.log.info(f"Saved results to {save_path}")
        
        return save_path

    def wait_generation_results(self, generation_id: str, sleep_sec=2) -> List[QAPair]:
        """
        Wait for generation results with polling.
        If the status is 'in_progress', retry every 2 seconds until completion.
        When completed, saves the results to project_folder/data/generated_qa_pairs/ directory.

        Args:
            generation_id: ID of the generation request
            sleep_sec: Time to wait between polling attempts in seconds
            parse_results: If True, automatically download and parse QA pairs when completed

        Returns:
            If parse_results is True: List of QAPair objects
            If parse_results is False: Status and raw results of the generation request
        """
        import time

        while True:
            result = self.fetch_generation_results(generation_id)
            status = result.get('status', '').lower()

            if status != 'in_progress':
                self.log.info(f"Generation status: {status}")

                if status == 'completed' and 'file' in result:
                    try:
                        self.log.info(f"File URL: {result['file']}")
                        self.log.info(f"Credits used: {result.get('credits', '?')}")

                        file_url = result['file']
                        
                        # Save to local file
                        save_path = self.save_generation_to_file(file_url, generation_id)
                        
                        # Read and parse the data
                        df = pd.read_json(save_path, lines=True)
                        self.log.info(f"Retrieved {len(df)} QA pairs")
                        qa_pairs = []
                        for record in df.to_dict('records'):
                            qa_pairs.append(QAPair.from_dict(record))
                        return qa_pairs
                    except Exception as e:
                        self.log.error(f"Error parsing QA pairs: {str(e)}")
                        return []
                else:
                    self.log.error(f"Generation failed: {result}")

            self.log.debug(
                f"Status: {status}, waiting {sleep_sec} seconds before retrying...")
            time.sleep(sleep_sec)
            
    @staticmethod
    def to_dataframe(qa_pairs: List[QAPair]) -> pd.DataFrame:
        """
        Convert a list of QAPair objects to a pandas DataFrame.
        """
        data = []
        for idx, qa in enumerate(qa_pairs, start=1):
            # Convert each QAPair to a dictionary
            row = {
                'qid': idx,  # Add qid starting from 1
                'question': qa.question,
                'answer': qa.answer,
                'context': qa.context,
                'question_categories': qa.question_categories,
                'user_categories': qa.user_categories,
                'document_ids': qa.document_ids
            }
            data.append(row)
        
        return pd.DataFrame(data)

"""
DataMorgana API for synthetic conversation generation.

DataMorgana is a tool for generating highly customizable and diverse synthetic Q&A 
benchmarks tailored to RAG applications. It enables detailed configurations of user 
and question categories and provides control over their distribution within the benchmark.
"""
import os
import requests
import json
from typing import Dict, List, Optional, Any, Union


class Category:
    """
    Represents a category for question or user categorization in DataMorgana.
    """
    def __init__(self, name: str, description: str, probability: Optional[float] = None):
        """
        Initialize a Category with name and description.
        
        Args:
            name: Category name
            description: Detailed description of the category
            probability: Probability weight for this category (for bulk generation)
        """
        self.name = name
        self.description = description
        self.probability = probability
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert category to dictionary format for API requests.
        
        Returns:
            Dictionary representation of the category
        """
        result = {
            "name": self.name,
            "description": self.description,
        }
        
        if self.probability is not None:
            result["probability"] = self.probability
            
        return result


class Categorization:
    """
    Represents a categorization group containing multiple categories.
    """
    def __init__(self, categorization_name: str, categories: List[Category]):
        """
        Initialize a Categorization with name and list of categories.
        
        Args:
            categorization_name: Name of the categorization
            categories: List of Category objects
        """
        self.categorization_name = categorization_name
        self.categories = categories
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert categorization to dictionary format for API requests.
        
        Returns:
            Dictionary representation of the categorization
        """
        return {
            "categorization_name": self.categorization_name,
            "categories": [category.to_dict() for category in self.categories]
        }


class DataMorgana:
    """
    Client for the DataMorgana API, which provides synthetic conversation generation
    capabilities for question-answer pairs based on specified categories and documents.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataMorgana client with API configuration.

        Args:
            api_key: DataMorgana API key (defaults to DATAMORGANA_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get('DATAMORGANA_API_KEY')
        if not self.api_key:
            raise ValueError("DATAMORGANA_API_KEY environment variable is required")
        
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
    
    def generate_sync_qa_pair(
        self, 
        question_categories: Dict[str, Dict[str, str]],
        user_categories: Optional[Dict[str, Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a question-answer pair synchronously.
        
        Args:
            question_categories: Dictionary of question categories
            user_categories: Dictionary of end-user categories
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
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def bulk_generation(
        self,
        n_questions: int,
        question_categorizations: List[Union[Categorization, Dict[str, Any]]],
        user_categorizations: Optional[List[Union[Categorization, Dict[str, Any]]]] = None,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Submit a bulk generation request asynchronously.
        
        Args:
            n_questions: Number of QA pairs to generate
            question_categorizations: List of question categorization configurations or Categorization objects
            user_categorizations: List of user categorization configurations or Categorization objects
            document_ids: List of document IDs to use for generation
            
        Returns:
            Response containing generation ID for status tracking
        """
        url = f"{self.base_url}/bulk_generation"
        
        # Convert Categorization objects to dictionaries if needed
        formatted_question_cats = []
        for cat in question_categorizations:
            if isinstance(cat, Categorization):
                formatted_question_cats.append(cat.to_dict())
            else:
                formatted_question_cats.append(cat)
        
        payload = {
            "n_questions": n_questions,
            "question_categorizations": formatted_question_cats
        }
        
        if user_categorizations:
            formatted_user_cats = []
            for cat in user_categorizations:
                if isinstance(cat, Categorization):
                    formatted_user_cats.append(cat.to_dict())
                else:
                    formatted_user_cats.append(cat)
            payload["user_categorizations"] = formatted_user_cats
            
        if document_ids:
            payload["document_ids"] = document_ids
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def fetch_generation_results(self, generation_id: str) -> Dict[str, Any]:
        """
        Fetch results of a previously submitted bulk generation request.
        
        Args:
            generation_id: ID of the generation request
            
        Returns:
            Status and results of the generation request
        """
        url = f"{self.base_url}/fetch_generation_results/request_id={generation_id}"
        
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
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        
        return response.json()
    
    #########################################################################
    # Kun: helper methods for creating question and user categories
    # can delete them if too verbose
    def create_question_category(self, name: str, description: str, probability: Optional[float] = None) -> Category:
        """
        Helper method to create a question category.
        
        Args:
            name: Category name
            description: Detailed description of the category
            probability: Optional probability weight for bulk generation
            
        Returns:
            Category object
        """
        return Category(name=name, description=description, probability=probability)
    
    def create_categorization(self, name: str, categories: List[Category]) -> Categorization:
        """
        Helper method to create a categorization group.
        
        Args:
            name: Categorization name
            categories: List of Category objects
            
        Returns:
            Categorization object
        """
        return Categorization(categorization_name=name, categories=categories)
    
    def get_default_question_categories(self) -> Dict[str, List[Category]]:
        """
        Get the default general-purpose question categories as described in the DataMorgana paper.
        
        Returns:
            Dictionary mapping categorization names to lists of Category objects
        """
        return {
            "factuality": [
                Category(
                    name="factoid", 
                    description="question seeking a specific, concise piece of information or a short fact about a particular subject, such as a name, date, or number",
                    probability=0.5
                ),
                Category(
                    name="open-ended",
                    description="question inviting detailed or exploratory responses, encouraging discussion or elaboration",
                    probability=0.5
                )
            ],
            "premise": [
                Category(
                    name="direct",
                    description="question that does not contain any premise or any information about the user",
                    probability=0.5
                ),
                Category(
                    name="with-premise",
                    description="question starting with a very short premise, where the user reveals their needs or some information about themselves",
                    probability=0.5
                )
            ],
            "phrasing": [
                Category(
                    name="concise-and-natural",
                    description="phrased in the way people typically speak, reflecting everyday language use, without formal or artificial structure. It is a concise direct question consisting of less than 10 words",
                    probability=0.25
                ),
                Category(
                    name="verbose-and-natural",
                    description="phrased in the way people typically speak, reflecting everyday language use, without formal or artificial structure. It is a relatively long question consisting of more than 9 words",
                    probability=0.25
                ),
                Category(
                    name="short-search-query",
                    description="phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of less than 7 words",
                    probability=0.25
                ),
                Category(
                    name="long-search-query",
                    description="phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure). It consists of more than 6 words",
                    probability=0.25
                )
            ],
            "linguistic_variation": [
                Category(
                    name="similar-to-document",
                    description="phrased using the same terminology and phrases appearing in the document",
                    probability=0.5
                ),
                Category(
                    name="distant-from-document",
                    description="phrased using terms completely different from the ones appearing in the document",
                    probability=0.5
                )
            ]
        }
    
    def get_general_user_categories(self) -> List[Category]:
        """
        Get the default general-purpose user categories (expertise-based).
        
        Returns:
            List of Category objects for general user expertise
        """
        return [
            Category(
                name="expert",
                description="a specialized user with deep understanding of the corpus",
                probability=0.5
            ),
            Category(
                name="novice",
                description="a regular user with no understanding of specialized terms",
                probability=0.5
            )
        ]
    
    def get_healthcare_user_categories(self) -> List[Category]:
        """
        Get healthcare-specific user categories as described in the DataMorgana paper.
        
        Returns:
            List of Category objects for healthcare users
        """
        return [
            Category(
                name="patient",
                description="a regular patient who uses the system to get basic health information, symptom checking, and guidance on preventive care",
                probability=0.25
            ),
            Category(
                name="medical-doctor",
                description="a medical doctor who needs to access some advanced information",
                probability=0.25
            ),
            Category(
                name="clinical-researcher",
                description="a clinical researcher who uses the system to access population health data, conduct initial patient surveys, track disease progression patterns, etc",
                probability=0.25
            ),
            Category(
                name="public-health-authority",
                description="a public health authority who uses the system to manage community health information dissemination, be informed on health emergencies, etc",
                probability=0.25
            )
        ]

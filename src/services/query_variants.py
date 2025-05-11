import re
from typing import List

from services.llms.general_openai_client import GeneralOpenAIClient
from utils.logging_utils import get_logger

logger = get_logger("query_variants")

def extract_queries(llm_txt: str) -> List[str]:
    """
    Extract queries from the LLM output.
    
    Args:
        llm_txt: LLM output text
        
    Returns:
        List of extracted queries
    """
    # Split the output by new lines and filter out empty lines
    lines = [line.strip() for line in llm_txt.split("\n") if line.strip()]
    
    extracted = []
    for line in lines:
        # First, remove any numbered list prefixes (e.g., "1. ", "2. ")
        clean_line = re.sub(r'^\d+\.\s*', '', line)
        
        # Find "query:" in the line (case insensitive)
        match = re.search(r'query:', clean_line, re.IGNORECASE)
        if match:
            # Extract everything after the last "query:"
            parts = re.split(r'query:', clean_line, flags=re.IGNORECASE)
            query_text = parts[-1].strip()  # Take the last part after splitting
            
            # Remove any quotes around the query
            query_text = query_text.strip('\'"')
            
            extracted.append(query_text)
    
    logger.debug(f"Extracted queries", input=llm_txt, output=extracted)
    return extracted

class QueryVariantsGenerator:
    """
    Class for generating query variants for simple questions.
    """
    
    def __init__(self, llm_client: GeneralOpenAIClient, system_prompt: str = None):
        """
        Initialize the QueryVariantsGenerator.
        
        Args:
            llm_client: LLM client to use for generating query variants
            system_prompt: System prompt to use for generating query variants (without format control)
        """
        self.llm_client = llm_client
        
        # Default system prompt if none provided
        default_prompt = f"Generate a list of search query variants based on the user's question. There are no spelling mistakes in the original question. Do not include any other text."
        
        if system_prompt is None:
            logger.warning(f"No system_prompt provided to QueryVariantsGenerator", using_system_prompt=default_prompt)
            self.default_system_prompt = default_prompt
        else:
            self.default_system_prompt = system_prompt
    
    def generate_variants(self, question: str, n_queries: int = 5, system_prompt: str = None) -> List[str]:
        """
        Generate query variants for a simple question.
        
        Args:
            question: The original question
            n_queries: Number of query variants to generate
            system_prompt: Override the default system prompt if provided
            
        Returns:
            List of query variants
        """
        # Use provided system prompt or default
        base_system_prompt = system_prompt or self.default_system_prompt
        
        # Insert n_queries if needed
        if "{n_queries}" in base_system_prompt:
            base_system_prompt = base_system_prompt.format(n_queries=n_queries)
        elif "{k_queries}" in base_system_prompt:
            # Support for legacy format using k_queries
            base_system_prompt = base_system_prompt.format(k_queries=n_queries)
        
        # Post-append format control
        system_prompt_with_format = base_system_prompt + " Format each variant as 'query: <your query>'."
        
        query_variants, _ = self.llm_client.complete_chat_once(
            system_message=system_prompt_with_format, message=f"Question: {question}")
        
        logger.debug(f"Generated query variants", input=question, output=query_variants)
        
        # Extract queries using the extraction function
        variants = extract_queries(query_variants)
        variants = variants[:n_queries]
        
        # If extraction fails, return the original question as fallback
        if not variants:
            logger.warning(f"Failed to extract query variants, using original question as fallback", question=question)
            return [question]
            
        return variants

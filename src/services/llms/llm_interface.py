"""
Interface for Language Model clients.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union


class LLMInterface(ABC):
    """
    Interface for Language Model clients.

    This abstract class defines the common interface that all LLM clients
    should implement, providing consistent methods for text completion
    and chat-based interactions.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
    ):
        """
        Initialize the LLM client.

        Args:
            model_id (str): The model identifier to use
            system_message (str): System message to use for all queries
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            **kwargs: Additional implementation-specific parameters
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str): The text prompt to complete

        Returns:
            str: The generated text content from the model
        """
        pass

    @abstractmethod
    def complete_chat(self, messages: List[Dict[str, str]]) -> Tuple[str, Any]:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, 
                each containing 'role' (system, user, or assistant) and 'content' keys

        Returns:
            Tuple[str, Any]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        pass

    @abstractmethod
    def complete_chat_once(self, message: str, system_message: Optional[str]) -> Tuple[str, Any]:
        """
        Generate a response for a chat conversation with a single call.

        Args:
            messages (str): A single prompt message, just like prompt(), the implementation will use a chat style
                conversation with the model.

        Returns:
            Tuple[str, Any]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        pass

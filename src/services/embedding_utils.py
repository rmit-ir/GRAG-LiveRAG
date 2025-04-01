"""
Embedding utility functions for text vectorization.
"""
import torch
from typing import List, Literal
from functools import cache
from transformers import AutoModel, AutoTokenizer


class EmbeddingUtils:
    """
    Utility class for generating text embeddings using transformer models.
    Handles model loading, tokenization, and embedding generation with
    various pooling strategies.
    """

    def __init__(self, embedding_model_name: str = "intfloat/e5-base-v2"):
        """
        Initialize the EmbeddingUtils with model configuration.

        Args:
            embedding_model_name: Name of the embedding model to use
        """
        self.embedding_model_name = embedding_model_name
        self._tokenizer = None
        self._model = None

    @staticmethod
    @cache
    def has_mps() -> bool:
        """Check if MPS (Metal Performance Shaders) is available."""
        return torch.backends.mps.is_available()

    @staticmethod
    @cache
    def has_cuda() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def get_tokenizer(self):
        """Get the tokenizer for the embedding model."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        return self._tokenizer

    def get_model(self):
        """Get the embedding model, loaded to the appropriate device."""
        if self._model is None:
            model = AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True)
            if self.has_mps():
                model = model.to("mps")
            elif self.has_cuda():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
            self._model = model
        return self._model

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform average pooling on the model's hidden states.
        
        Args:
            last_hidden_states: The last hidden states from the model
            attention_mask: The attention mask
            
        Returns:
            The pooled tensor
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_query(
        self,
        query: str,
        query_prefix: str = "query: ",
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True
    ) -> List[float]:
        """
        Embed a single query using the embedding model.
        
        Args:
            query: The query text to embed
            query_prefix: Prefix to add to the query
            pooling: Pooling strategy to use (cls or avg)
            normalize: Whether to normalize the embeddings
            
        Returns:
            The embedding as a list of floats
        """
        return self.batch_embed_queries([query], query_prefix, pooling, normalize)[0]

    def batch_embed_queries(
        self,
        queries: List[str],
        query_prefix: str = "query: ",
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple queries in batch.
        
        Args:
            queries: List of query texts to embed
            query_prefix: Prefix to add to each query
            pooling: Pooling strategy to use (cls or avg)
            normalize: Whether to normalize the embeddings
            
        Returns:
            List of embeddings as lists of floats
        """
        with_prefixes = [" ".join([query_prefix, query]) for query in queries]
        tokenizer = self.get_tokenizer()
        model = self.get_model()
        
        with torch.no_grad():
            encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
            encoded = encoded.to(model.device)
            model_out = model(**encoded)
            
            if pooling == "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            elif pooling == "avg":
                embeddings = self.average_pool(model_out.last_hidden_state, encoded["attention_mask"])
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
                
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
        return embeddings.tolist()

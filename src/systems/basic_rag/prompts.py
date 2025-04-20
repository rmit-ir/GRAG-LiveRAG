# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful assistant, answer the following question 
based on the provided context. If the context doesn't contain relevant 
information, say so.."""

# Answer generation prompt template
ANSWER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:
"""

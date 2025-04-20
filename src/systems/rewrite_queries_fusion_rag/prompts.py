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

# Query generation prompt template
QUERY_GENERATION_PROMPT = """Given the following question, generate alternative search queries that could help find relevant information. 
Each query should be a different way of asking for the same information or focus on different aspects of the question.
Return each query on a new line without numbering or additional text.

Original question: {question}
"""

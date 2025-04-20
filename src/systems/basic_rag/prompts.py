# System prompt for the LLM
SYSTEM_PROMPT = "You are a helpful assistant that provides accurate and concise answers based on the provided context."

# Answer generation prompt template
ANSWER_PROMPT_TEMPLATE = """
Answer the following question based on the provided context. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:
"""

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful assistant, answer the following question 
based on the provided context. If the context doesn't contain relevant 
information, say so."""

# Answer generation prompt template
ANSWER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:
"""

# Query decomposition prompt
QUERY_DECOMPOSITION_PROMPT = """You are a query decomposition expert. For the user's question below, break it down into {max_components} distinct components or aspects that need to be addressed.

For complex questions, identify the different parts that require separate information. For simple questions, you may return fewer components.

Format your response as a numbered list enclosed in components HTML tags:

<components>
1. First component
2. Second component
...
</components>
"""

# Query generation prompt for each component
COMPONENT_QUERY_GENERATION_PROMPT = """You are a search query generator. For the component of a question below, generate {max_queries} search queries optimized for embedding-based semantic search. 

The component is part of a larger question: "{original_question}"

For this specific component: "{component}"

Generate queries that:
- Capture the semantic meaning of this component
- Use different phrasings and perspectives
- Include relevant context from the original question
- Are specific enough to retrieve targeted information

Format your response as a numbered list enclosed in queries HTML tags:

<queries>
1. query
2. query
...
</queries>
"""

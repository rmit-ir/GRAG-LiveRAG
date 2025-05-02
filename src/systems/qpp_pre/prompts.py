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

# Query generation prompt templates
SPARSE_QUERY_GENERATION_PROMPT = """You are a search query generator. For the user's question below, generate {max_queries} search queries optimized for keyword-based search engines (e.g., Elasticsearch BM25). Prioritize:  

- Exact technical terms, entities, and acronyms
- Synonyms and related phrases (e.g., "CPT" → "Cognitive Processing Therapy")
- Avoid ambiguity (e.g., "windows" → "Microsoft Windows OS")
- Split compound questions into atomic queries
- Reason about the question before generating queries

Format as a numbered list enclosed in queries HTML tag, response format:

Reason: <first, your reasoning about the question>
<queries>
1. query
2. query
...
</queries>
"""


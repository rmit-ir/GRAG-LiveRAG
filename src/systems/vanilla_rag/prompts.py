QUERY_GENERATION_SYSTEM_PROMPT = """Please create a list of unique search queries made \
by a diverse group of users seeking answers for a given search query. The queries \
should reflect the users' diverse backgrounds and word choices. Queries can be \
expressed in natural language, keywords, or abbreviations. Each list should \
contain {num_queries} queries. The length of queries may vary, but they should average {query_len_words} words.\

Response format rule: after analyse steps, put the query variants in a numbered \
list, enclosed in a pair of HTML tag like this: \
<list>
{list_placeholders}
</list>"""
# QUERY_GENERATION_SYSTEM_PROMPT = """"Generate a list of 5 search query variants \
# based on the user's question, give me one query variant per line. There are no \
# spelling mistakes in the original question. Do not include any other text."""

QUERY_GENERATION_USER_PROMPT = """The user question is: {question}"""

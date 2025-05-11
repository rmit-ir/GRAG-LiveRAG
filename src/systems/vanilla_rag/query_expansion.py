import re
from typing import List, NamedTuple

from services.llms.ai71_client import AI71Client
from services.llms.general_openai_client import GeneralOpenAIClient
from systems.vanilla_rag.vanilla_rag_optimized_prompt_prompts import IS_SIMPLE_SYS_PROMPT, QUERY_DECOMPOSITION_SYS_PROMPT, QUERY_REPHRASING_SYS_PROMPT, SUB_QUERIES_SYS_PROMPT
from utils.logging_utils import get_logger

logger = get_logger("query_expansion")


class ExpandedQueries(NamedTuple):
    """
    NamedTuple containing all intermediary results from the query expansion process.
    """
    original_query: str
    components: str  # Decomposition components
    rephrased_query: str
    is_simple: bool
    sub_queries: List[str]


def is_simple_query(q: str, llm_client: GeneralOpenAIClient) -> bool:
    # Check if the question is simple or composite
    is_simple, _ = llm_client.complete_chat_once(
        system_message=IS_SIMPLE_SYS_PROMPT, message="Question: " + q)
    logger.debug(f"Is simple question, input: \n{q}\noutput: \n{is_simple}")
    return "SIMPLE" in is_simple.upper()

def _extract_sub_queries(llm_txt: str) -> List[str]:
    """
    Extract sub-queries from the LLM output.
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
    
    logger.debug(f"Extracted sub-queries", input=llm_txt, output=extracted)
    return extracted

def expand_simple_query(q: str, llm_client: GeneralOpenAIClient, n_queries: int = 5) -> List[str]:
    """
    Generate query variants for simple questions.
    
    Args:
        q: The original query
        llm_client: LLM client to use
        n_queries: Number of query variants to generate
        
    Returns:
        List of query variants
    """
    system_prompt = f"Generate a list of {n_queries} search query variants based on the user's question. Format each variant as 'query: <your query>'. There are no spelling mistakes in the original question. Do not include any other text."
    
    query_variants, _ = llm_client.complete_chat_once(
        system_message=system_prompt, message="Question: " + q)
    
    logger.debug(f"Generated simple query variants", input=q, output=query_variants)
    
    # Extract queries using the same extraction function
    return _extract_sub_queries(query_variants)

def expand_composite_query(q: str, llm_client: GeneralOpenAIClient) -> List[str]:
    """
    Generate sub-queries for composite questions.
    
    Args:
        q: The rephrased query
        llm_client: LLM client to use
        
    Returns:
        List of sub-queries
    """
    qs_str, _ = llm_client.complete_chat_once(
        system_message=SUB_QUERIES_SYS_PROMPT, message=f"Question: {q}")
    
    logger.debug(f"Generated composite sub-queries", input=q, output=qs_str)
    
    return _extract_sub_queries(qs_str)

def expand_queries(q: str, llm_client: GeneralOpenAIClient, n_queries: int = 5) -> ExpandedQueries:
    """
    Expand a query into components, rephrased query, and sub-queries.
    
    Args:
        q: The original query
        llm_client: LLM client to use
        n_queries: Number of query variants to generate for simple queries
        
    Returns:
        ExpandedQueries object containing all intermediary results
    """
    # Query decomposition
    components_str, _ = llm_client.complete_chat_once(
        system_message=QUERY_DECOMPOSITION_SYS_PROMPT, message="Question: " + q)
    logger.debug(f"Decomposed question, input: \n{q}\noutput: \n{components_str}")
    
    # Query rephrasing
    rephrased_q, _ = llm_client.complete_chat_once(
        system_message=QUERY_REPHRASING_SYS_PROMPT, message=f"Question: {q}\n\n{components_str}")
    logger.debug(f"Rephrased question, input: \n{q}\noutput: \n{rephrased_q}")
    
    # Check if the query is simple
    simple = is_simple_query(q, llm_client)
    
    if simple:
        # For simple queries, generate query variants
        sub_queries = expand_simple_query(q, llm_client, n_queries)
        if not sub_queries:  # Fallback if extraction fails
            sub_queries = [q]
    else:
        # For composite queries, generate sub-queries
        sub_queries = expand_composite_query(rephrased_q, llm_client)
        if not sub_queries:  # Fallback if extraction fails
            sub_queries = [rephrased_q]
    
    return ExpandedQueries(
        original_query=q,
        components=components_str,
        rephrased_query=rephrased_q,
        is_simple=simple,
        sub_queries=sub_queries
    )


# question: tourist outdoor activities mid january september alice springs ahmedabad compare
# query: best outdoor leisure activities in Alice Springs mid-January  
# query: best outdoor leisure activities in Ahmedabad mid-January  
# query: best outdoor leisure activities in Alice Springs September  
# query: best outdoor leisure activities in Ahmedabad September  
# query: comparison of outdoor leisure activities in Alice Springs vs Ahmedabad mid-January  
# query: comparison of outdoor leisure activities in Alice Springs vs Ahmedabad September


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    llm_client = AI71Client()
    q1 = "tourist outdoor activities mid january september alice springs ahmedabad compare"
    qs1 = expand_queries(q1, llm_client)
    q2 = "Where was the Christmas message filmed?"
    qs2 = expand_queries(q2, llm_client)
    q3 = "As someone studying crop quality parameters, I'm curious about the early test results for the 2021 hard red winter wheat harvest. What were the key measurements for moisture content, protein levels, and test weight?"
    qs3 = expand_queries(q3, llm_client)
    q4 = "exocrine pancreatic insufficiency symptoms causes patient numbers usa"
    qs4 = expand_queries(q4, llm_client)
    q5 = "what happens first pregnancy doctor visit"
    qs5 = expand_queries(q5, llm_client)
    q6 = "As a culinary historian exploring ancient pasta-making traditions, I'd like to know how did early Chinese, Middle Eastern, and Italian cultures differ in their approach to preparing and drying noodles before modern times?"
    qs6 = expand_queries(q6, llm_client)
    q7 = "How do digital tools help retirement savings and what security risks exist?"
    qs7 = expand_queries(q7, llm_client)
    q8 = "dust collector box design requirements proper airflow considerations"
    qs8 = expand_queries(q8, llm_client)
    q9 = "How much extra do stores add for credit card payments?"
    qs9 = expand_queries(q9, llm_client)
    q10 = "As a ruminant nutrition specialist, I'm researching methane emissions from cattle digestion. What is the role of protein in ruminant digestion, and how does this relate to their contribution to global warming?"
    qs10 = expand_queries(q10, llm_client)
    q11 = "I work with digital archives and I'd like to understand the relationship between public domain status and digital copies - how does copyright expiration affect works entering the public domain, and what special considerations apply to digital reproductions of these works?"
    qs11 = expand_queries(q11, llm_client)
    q12 = "How can companies protect their intellectual property?"
    qs12 = expand_queries(q12, llm_client)
    q13 = "I'm worried about both my heart health and digestion - should I stick to low-fat dairy products, and what digestive problems might dairy cause?"
    qs13 = expand_queries(q13, llm_client)
    q14 = "What solutions exist for wildlife conflicts in energy facilities?"
    qs14 = expand_queries(q14, llm_client)
    q15 = "what hardware components needed build automatic loading firing catapult ev3"
    qs15 = expand_queries(q15, llm_client)
    q16 = "I'm an architectural historian researching luxury hotels in former financial buildings - what interesting features did they preserve in The Ned hotel from its past as a clearinghouse bank?"
    qs16 = expand_queries(q16, llm_client)
    q17 = "What factors affect school closures and what business impacts need planning?"
    qs17 = expand_queries(q17, llm_client)
    q18 = "Hey, I'm fascinated by archaeological digs happening in Israel. What are they searching for at the Tel Shiloh site and have they found anything interesting so far?"
    qs18 = expand_queries(q18, llm_client)
    q19 = "inattentive adhd anxiety disorder physical symptoms"
    qs19 = expand_queries(q19, llm_client)
    q20 = "As a veterinary student focusing on feline development, I'm curious about the exact timeline of when kittens get their first teeth and how the teething process progresses - could you explain the key stages?"
    qs20 = expand_queries(q20, llm_client)
    q21 = "What kind of materials does Bio-lutions use to make their biodegradable packaging products and how do they process them?"
    qs21 = expand_queries(q21, llm_client)
    q22 = "compare warehouse automation cold storage medical supplies handling"
    qs22 = expand_queries(q22, llm_client)
    q23 = "mosquito virus symptoms effects"
    qs23 = expand_queries(q23, llm_client)
    q24 = "important art academies realism training"
    qs24 = expand_queries(q24, llm_client)
    q25 = "What role does augmented reality play in teaching abstract STEM concepts, and how are new paper technologies being integrated into educational materials?"
    qs25 = expand_queries(q25, llm_client)
    q26 = "Why do birds fly high up in the evening?"
    qs26 = expand_queries(q26, llm_client)
    q27 = "How did ancient leaders die while defending their lands?"
    qs27 = expand_queries(q27, llm_client)
    q28 = "photo print longevity storage methods comparison"
    qs28 = expand_queries(q28, llm_client)
    q29 = "how are modern vinyl floorcloths made"
    qs29 = expand_queries(q29, llm_client)
    q30 = "As an aviation engineer, what causes ice formation on aircraft and how is it monitored for safety?"
    qs30 = expand_queries(q30, llm_client)
    q31 = "What hardware components are essential to assemble a CNC machine?"
    qs31 = expand_queries(q31, llm_client)
    q32 = "What are the key variables in convertible debt and bee population health?"
    qs32 = expand_queries(q32, llm_client)
    
    print('=' * 80)
    print(f'Original question: {q1}')
    print(f'Rephrased query: {qs1.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs1.sub_queries))
    print('=' * 80)
    print(f'Original question: {q2}')
    print(f'Rephrased query: {qs2.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs2.sub_queries))
    print('=' * 80)
    print(f'Original question: {q3}')
    print(f'Rephrased query: {qs3.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs3.sub_queries))
    print('=' * 80)
    print(f'Original question: {q4}')
    print(f'Rephrased query: {qs4.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs4.sub_queries))
    print('=' * 80)
    print(f'Original question: {q5}')
    print(f'Rephrased query: {qs5.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs5.sub_queries))
    print('=' * 80)
    print(f'Original question: {q6}')
    print(f'Rephrased query: {qs6.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs6.sub_queries))
    print('=' * 80)
    print(f'Original question: {q7}')
    print(f'Rephrased query: {qs7.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs7.sub_queries))
    print('=' * 80)
    print(f'Original question: {q8}')
    print(f'Rephrased query: {qs8.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs8.sub_queries))
    print('=' * 80)
    print(f'Original question: {q9}')
    print(f'Rephrased query: {qs9.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs9.sub_queries))
    print('=' * 80)
    print(f'Original question: {q10}')
    print(f'Rephrased query: {qs10.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs10.sub_queries))
    print('=' * 80)
    print(f'Original question: {q11}')
    print(f'Rephrased query: {qs11.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs11.sub_queries))
    print('=' * 80)
    print(f'Original question: {q12}')
    print(f'Rephrased query: {qs12.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs12.sub_queries))
    print('=' * 80)
    print(f'Original question: {q13}')
    print(f'Rephrased query: {qs13.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs13.sub_queries))
    print('=' * 80)
    print(f'Original question: {q14}')
    print(f'Rephrased query: {qs14.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs14.sub_queries))
    print('=' * 80)
    print(f'Original question: {q15}')
    print(f'Rephrased query: {qs15.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs15.sub_queries))
    print('=' * 80)
    print(f'Original question: {q16}')
    print(f'Rephrased query: {qs16.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs16.sub_queries))
    print('=' * 80)
    print(f'Original question: {q17}')
    print(f'Rephrased query: {qs17.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs17.sub_queries))
    print('=' * 80)
    print(f'Original question: {q18}')
    print(f'Rephrased query: {qs18.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs18.sub_queries))
    print('=' * 80)
    print(f'Original question: {q19}')
    print(f'Rephrased query: {qs19.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs19.sub_queries))
    print('=' * 80)
    print(f'Original question: {q20}')
    print(f'Rephrased query: {qs20.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs20.sub_queries))
    print('=' * 80)
    print(f'Original question: {q21}')
    print(f'Rephrased query: {qs21.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs21.sub_queries))
    print('=' * 80)
    print(f'Original question: {q22}')
    print(f'Rephrased query: {qs22.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs22.sub_queries))
    print('=' * 80)
    print(f'Original question: {q23}')
    print(f'Rephrased query: {qs23.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs23.sub_queries))
    print('=' * 80)
    print(f'Original question: {q24}')
    print(f'Rephrased query: {qs24.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs24.sub_queries))
    print('=' * 80)
    print(f'Original question: {q25}')
    print(f'Rephrased query: {qs25.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs25.sub_queries))
    print('=' * 80)
    print(f'Original question: {q26}')
    print(f'Rephrased query: {qs26.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs26.sub_queries))
    print('=' * 80)
    print(f'Original question: {q27}')
    print(f'Rephrased query: {qs27.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs27.sub_queries))
    print('=' * 80)
    print(f'Original question: {q28}')
    print(f'Rephrased query: {qs28.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs28.sub_queries))
    print('=' * 80)
    print(f'Original question: {q29}')
    print(f'Rephrased query: {qs29.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs29.sub_queries))
    print('=' * 80)
    print(f'Original question: {q30}')
    print(f'Rephrased query: {qs30.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs30.sub_queries))
    print('=' * 80)
    print(f'Original question: {q31}')
    print(f'Rephrased query: {qs31.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs31.sub_queries))
    print('=' * 80)
    print(f'Original question: {q32}')
    print(f'Rephrased query: {qs32.rephrased_query}')
    print(f'Sub-queries:\n' + "\n".join(qs32.sub_queries))
    print('=' * 80)

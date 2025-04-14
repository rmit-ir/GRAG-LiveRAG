# Technical Context: RAG System

## Technical Problem Statement

Traditional language models have technical limitations:
- Fixed knowledge cutoffs
- Inability to access real-time information
- Potential for hallucination or outdated information
- Limited domain-specific knowledge without fine-tuning

The LiveRAG system addresses these technical challenges by implementing:
1. Real-time retrieval from vector databases
2. Context augmentation for language model responses
3. Up-to-date information delivery through retrieval mechanisms

## Technical Implementation Focus

- Accurate and contextually relevant retrieval mechanisms
- Efficient integration of retrieved information into prompts
- Fast response times despite the additional retrieval step
- Support for diverse query types
- Concise response generation (200 token limit)
- High-quality supporting passage selection

## Technical Use Cases

1. **Vector-Based Retrieval**: Implementing efficient vector search across multiple providers
2. **Embedding Generation**: Converting text to vector representations
3. **Context Selection**: Identifying and ranking the most relevant passages
4. **Prompt Engineering**: Constructing effective prompts with retrieved context
5. **Response Generation**: Generating concise, accurate responses with the LLM

## Technical Evaluation Metrics

1. **Relevance**: Technical measure of answer quality
   - Graded on a four-point scale:
     - **2:** Correct and relevant (no irrelevant information)
     - **1:** Correct but contains irrelevant information
     - **0:** No answer provided (abstention)
     - **-1:** Incorrect answer

2. **Faithfulness**: Technical measure of grounding in retrieved passages
   - Graded on a three-point scale:
     - **1:** Full support. All answer parts are grounded
     - **0:** Partial support. Not all answer parts are grounded
     - **-1:** No support. All answer parts are not grounded

## Technical Requirements

1. **Vector Search Implementation**:
   - Must support multiple index providers (Pinecone, OpenSearch)
   - Must efficiently retrieve relevant passages

2. **LLM Integration**:
   - Must exclusively use Falcon3-10B-Instruct for answer generation
   - No fine-tuning allowed (use "as is")
   - Response limited to 200 tokens

3. **DataMorgana Integration**:
   - Generate diverse Q&A benchmarks
   - Support customization of question and user characteristics
   - Enable comprehensive evaluation across various scenarios

# https://aws.amazon.com/bedrock/pricing/

# Pricing per 1000 tokens for different models (in USD)
# These prices should be updated as AWS changes their pricing
BEDROCK_MODEL_PRICING = {
    # Claude 3.7 models
    "anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    # Claude 3.5 models
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_price": 1.00,  # $1.00 per 1M input tokens ($0.001 per 1K)
        "output_price": 5.00   # $5.00 per 1M output tokens ($0.005 per 1K)
    },
    # Claude 3 models
    "anthropic.claude-3-opus-20240229-v1:0": {
        "input_price": 15.00,  # $15.00 per 1M input tokens ($0.015 per 1K)
        "output_price": 75.00  # $75.00 per 1M output tokens ($0.075 per 1K)
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.003 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.015 per 1K)
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input_price": 0.25,  # $0.25 per 1M input tokens ($0.00025 per 1K)
        "output_price": 1.25   # $1.25 per 1M output tokens ($0.00125 per 1K)
    },
    # Claude 2 models
    "anthropic.claude-v2:1": {
        "input_price": 8.00,  # $8.00 per 1M input tokens ($0.008 per 1K)
        "output_price": 24.00  # $24.00 per 1M output tokens ($0.024 per 1K)
    },
    "anthropic.claude-v2:0": {
        "input_price": 8.00,  # $8.00 per 1M input tokens ($0.008 per 1K)
        "output_price": 24.00  # $24.00 per 1M output tokens ($0.024 per 1K)
    },
    "anthropic.claude-instant-v1": {
        "input_price": 0.80,  # $0.80 per 1M input tokens ($0.0008 per 1K)
        "output_price": 2.40   # $2.40 per 1M output tokens ($0.0024 per 1K)
    },
    # Cohere models
    "cohere.command-text-v14": {
        "input_price": 1.50,  # $1.50 per 1M input tokens ($0.0015 per 1K)
        "output_price": 2.00   # $2.00 per 1M output tokens ($0.0020 per 1K)
    },
    "cohere.command-light-text-v14": {
        "input_price": 0.30,  # $0.30 per 1M input tokens ($0.0003 per 1K)
        "output_price": 0.60   # $0.60 per 1M output tokens ($0.0006 per 1K)
    },
    "cohere.command-r-plus-v1:0": {
        "input_price": 3.00,  # $3.00 per 1M input tokens ($0.0030 per 1K)
        "output_price": 15.00  # $15.00 per 1M output tokens ($0.0150 per 1K)
    },
    "cohere.command-r-v1:0": {
        "input_price": 0.50,  # $0.50 per 1M input tokens ($0.0005 per 1K)
        "output_price": 1.50   # $1.50 per 1M output tokens ($0.0015 per 1K)
    },
    "cohere.embed-english-v3:0": {
        "input_price": 0.10,  # $0.10 per 1M input tokens ($0.0001 per 1K)
        "output_price": 0.00   # Embedding models don't have output tokens
    },
    "cohere.embed-multilingual-v3:0": {
        "input_price": 0.10,  # $0.10 per 1M input tokens ($0.0001 per 1K)
        "output_price": 0.00   # Embedding models don't have output tokens
    },
    # Meta Llama models
    "meta.llama3-70b-instruct-v1:0": {
        "input_price": 0.72,  # $0.72 per 1M input tokens ($0.00072 per 1K)
        "output_price": 0.72   # $0.72 per 1M output tokens ($0.00072 per 1K)
    },
    "meta.llama3-8b-instruct-v1:0": {
        "input_price": 0.22,  # $0.22 per 1M input tokens ($0.00022 per 1K)
        "output_price": 0.22   # $0.22 per 1M output tokens ($0.00022 per 1K)
    },
    "meta.llama3-1-70b-instruct-v1:0": {
        "input_price": 0.72,  # $0.72 per 1M input tokens ($0.00072 per 1K)
        "output_price": 0.72   # $0.72 per 1M output tokens ($0.00072 per 1K)
    },
    "meta.llama3-1-8b-instruct-v1:0": {
        "input_price": 0.22,  # $0.22 per 1M input tokens ($0.00022 per 1K)
        "output_price": 0.22   # $0.22 per 1M output tokens ($0.00022 per 1K)
    },
    "meta.llama3-1-405b-instruct-v1:0": {
        "input_price": 2.40,  # $2.40 per 1M input tokens ($0.0024 per 1K)
        "output_price": 2.40   # $2.40 per 1M output tokens ($0.0024 per 1K)
    },
    "meta.llama3-2-1b-instruct-v1:0": {
        "input_price": 0.10,  # $0.10 per 1M input tokens ($0.0001 per 1K)
        "output_price": 0.10   # $0.10 per 1M output tokens ($0.0001 per 1K)
    },
    "meta.llama3-2-3b-instruct-v1:0": {
        "input_price": 0.15,  # $0.15 per 1M input tokens ($0.00015 per 1K)
        "output_price": 0.15   # $0.15 per 1M output tokens ($0.00015 per 1K)
    },
    "meta.llama3-2-11b-instruct-v1:0": {
        "input_price": 0.16,  # $0.16 per 1M input tokens ($0.00016 per 1K)
        "output_price": 0.16   # $0.16 per 1M output tokens ($0.00016 per 1K)
    },
    "meta.llama3-2-90b-instruct-v1:0": {
        "input_price": 0.72,  # $0.72 per 1M input tokens ($0.00072 per 1K)
        "output_price": 0.72   # $0.72 per 1M output tokens ($0.00072 per 1K)
    },
    # Use cross-regional inference model id: us.meta.llama3-3-70b-instruct-v1:0
    # see https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/inference-profiles
    "meta.llama3-3-70b-instruct-v1:0": {
        "input_price": 0.72,  # $0.72 per 1M input tokens ($0.00072 per 1K)
        "output_price": 0.72   # $0.72 per 1M output tokens ($0.00072 per 1K)
    },
    "meta.llama4-scout-17b-instruct-v1:0": {
        "input_price": 0.17,  # $0.17 per 1M input tokens ($0.00017 per 1K)
        "output_price": 0.66   # $0.66 per 1M output tokens ($0.00066 per 1K)
    },
    "meta.llama4-maverick-17b-instruct-v1:0": {
        "input_price": 0.24,  # $0.24 per 1M input tokens ($0.00024 per 1K)
        "output_price": 0.97   # $0.97 per 1M output tokens ($0.00097 per 1K)
    },
    # Mistral AI models
    "mistral.mistral-7b-instruct-v0:2": {
        "input_price": 0.15,  # $0.15 per 1M input tokens ($0.00015 per 1K)
        "output_price": 0.20   # $0.20 per 1M output tokens ($0.0002 per 1K)
    },
    "mistral.mixtral-8x7b-instruct-v0:1": {
        "input_price": 0.45,  # $0.45 per 1M input tokens ($0.00045 per 1K)
        "output_price": 0.70   # $0.70 per 1M output tokens ($0.0007 per 1K)
    },
    "mistral.mistral-large-2402-v1:0": {
        "input_price": 8.00,  # $8.00 per 1M input tokens ($0.008 per 1K)
        "output_price": 24.00  # $24.00 per 1M output tokens ($0.024 per 1K)
    },
    "mistral.mistral-large-2407-v1:0": {
        "input_price": 8.00,  # $8.00 per 1M input tokens ($0.008 per 1K)
        "output_price": 24.00  # $24.00 per 1M output tokens ($0.024 per 1K)
    },
    # Amazon models
    "amazon.titan-text-express-v1": {
        "input_price": 0.30,  # $0.30 per 1M input tokens ($0.0003 per 1K)
        "output_price": 0.40   # $0.40 per 1M output tokens ($0.0004 per 1K)
    },
    "amazon.titan-text-lite-v1": {
        "input_price": 0.30,  # $0.30 per 1M input tokens ($0.0003 per 1K)
        "output_price": 0.40   # $0.40 per 1M output tokens ($0.0004 per 1K)
    },
    "amazon.titan-embed-text-v2:0": {
        "input_price": 0.02,  # $0.02 per 1M input tokens ($0.00002 per 1K)
        "output_price": 0.00   # Embedding models don't have output tokens
    },
    # Amazon Nova models
    "amazon.nova-micro-v1:0": {
        "input_price": 0.035,  # $0.035 per 1M input tokens ($0.000035 per 1K)
        "output_price": 0.14   # $0.14 per 1M output tokens ($0.00014 per 1K)
    },
    "amazon.nova-lite-v1:0": {
        "input_price": 0.06,  # $0.06 per 1M input tokens ($0.00006 per 1K)
        "output_price": 0.24   # $0.24 per 1M output tokens ($0.00024 per 1K)
    },
    "amazon.nova-pro-v1:0": {
        "input_price": 0.80,  # $0.80 per 1M input tokens ($0.0008 per 1K)
        "output_price": 3.20   # $3.20 per 1M output tokens ($0.0032 per 1K)
    },
    "amazon.nova-premier-v1:0": {
        "input_price": 2.50,  # $2.50 per 1M input tokens ($0.0025 per 1K)
        "output_price": 12.50  # $12.50 per 1M output tokens ($0.0125 per 1K)
    },
    # AI21 Labs models
    "ai21.jamba-1-5-large-v1:0": {
        "input_price": 2.00,  # $2.00 per 1M input tokens ($0.002 per 1K)
        "output_price": 8.00   # $8.00 per 1M output tokens ($0.008 per 1K)
    },
    "ai21.jamba-1-5-mini-v1:0": {
        "input_price": 0.20,  # $0.20 per 1M input tokens ($0.0002 per 1K)
        "output_price": 0.40   # $0.40 per 1M output tokens ($0.0004 per 1K)
    },
    "ai21.j2-mid-v1": {
        "input_price": 12.50,  # $12.50 per 1M input tokens ($0.0125 per 1K)
        "output_price": 12.50  # $12.50 per 1M output tokens ($0.0125 per 1K)
    },
    "ai21.j2-ultra-v1": {
        "input_price": 18.80,  # $18.80 per 1M input tokens ($0.0188 per 1K)
        "output_price": 18.80  # $18.80 per 1M output tokens ($0.0188 per 1K)
    },
    # DeepSeek models
    "deepseek.r1-v1:0": {
        "input_price": 1.35,  # $1.35 per 1M input tokens ($0.00135 per 1K)
        "output_price": 5.40   # $5.40 per 1M output tokens ($0.0054 per 1K)
    },
    # Writer models
    "writer.palmyra-x4-v1:0": {
        "input_price": 2.50,  # $2.50 per 1M input tokens ($0.0025 per 1K)
        "output_price": 10.00  # $10.00 per 1M output tokens ($0.010 per 1K)
    },
    "writer.palmyra-x5-v1:0": {
        "input_price": 0.60,  # $0.60 per 1M input tokens ($0.0006 per 1K)
        "output_price": 6.00   # $6.00 per 1M output tokens ($0.006 per 1K)
    },
    # Default pricing if model not found
    "default": {
        "input_price": 3.00,  # Default to Claude 3.5 Sonnet pricing
        "output_price": 15.00
    }
}

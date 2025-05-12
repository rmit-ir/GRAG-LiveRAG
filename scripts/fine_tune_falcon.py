import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# Set up HuggingFace token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Successfully logged in to Hugging Face Hub")
else:
    print("Warning: HF_TOKEN environment variable not found")

# Set the cache directory
HF_HOME = os.environ.get("HF_HOME")
if HF_HOME:
    cache_dir = os.path.join(HF_HOME, "hub")
    print(f"Using cache directory: {cache_dir}")
else:
    cache_dir = None
    print("Warning: HF_HOME environment variable not found, using default cache")

# Define paths and parameters
MODEL_NAME = "tiiuae/Falcon3-10B-Instruct"
OUTPUT_DIR = "./fine_tuned_falcon3"
DATA_PATH = "../data/live_rag_questions/combined_judgments_df_prompts.tsv"  # Replace with your TSV file path

# A100 40GB Optimized Settings
MAX_LENGTH = 1024
BATCH_SIZE = 2  # Small batch size for large model
GRADIENT_ACCUMULATION = 16  # Effective batch size = 32
LORA_R = 8  # Reduced LoRA rank dimension
LORA_ALPHA = 16  # Alpha parameter for LoRA

print(
    f"Training on: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
)
(
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    if torch.cuda.is_available()
    else print("No GPU detected")
)

# Load tokenizer with cache directory
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=HF_TOKEN,
    force_download=True,
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization for A100 efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # More memory efficient than 8-bit
    bnb_4bit_quant_type="nf4",  # Use normalized float 4 for better quality
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=HF_TOKEN,
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
for name, _ in model.named_modules():
    if any(x in name for x in ["query", "key", "value", "dense", "proj"]):
        print(name)

# Configure LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj", 
        "mlp.down_proj"
    ],  # Updated to match actual Falcon layer names
    modules_to_save=["lm_head"],
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters info
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(
    f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}% of {total_params:,} total)"
)

# Load TSV data file
print(f"Loading dataset from {DATA_PATH}...")
try:
    # Load the TSV file with pandas
    df = pd.read_csv(DATA_PATH, sep="\t", encoding="utf-8")

    # Verify the required columns exist
    if not all(col in df.columns for col in ["prompt", "response"]):
        missing = [col for col in ["prompt", "response"] if col not in df.columns]
        raise ValueError(f"Missing required columns in TSV file: {missing}")

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_pandas(df)

    # Create a train/validation split if no validation set provided
    if "validation" not in dataset:
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Dataset loaded: {dataset}")
    print(f"Number of training examples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Number of validation examples: {len(dataset['validation'])}")

    # Preview some data
    print("\nData preview:")
    for i, example in enumerate(dataset["train"].select(range(2))):
        print(f"Example {i+1}:")
        print(f"  Prompt: {example['prompt'][:100]}...")
        print(f"  Response: {example['response'][:100]}...")

except Exception as e:
    print(f"Error loading dataset: {e}")
    raise


# Function to preprocess the dataset
def preprocess_function(examples):
    # Format: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    prompt_text = examples["prompt"]
    response_text = examples["response"]

    formatted_inputs = []

    for prompt, response in zip(prompt_text, response_text):
        # Handle potential NaN values
        prompt = "" if pd.isna(prompt) else str(prompt).strip()
        response = "" if pd.isna(response) else str(response).strip()

        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        formatted_inputs.append(formatted_text)

    # Tokenize with efficient padding and truncation
    tokenized_inputs = tokenizer(
        formatted_inputs,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="np",  # Use numpy for more memory efficiency during preprocessing
    )

    # Set input_ids as labels for causal language modeling
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


# Process dataset with batched operations for speed
print("Tokenizing and preparing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=64,  # Process in larger chunks for efficiency
    num_proc=4,  # Parallel processing
    remove_columns=dataset["train"].column_names,
)

# Create data collator for batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # Not using masked language modeling
)

# Define training arguments optimized for A100 40GB
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,  # Only keep the 3 most recent checkpoints
    logging_steps=10,
    learning_rate=1e-4,
    bf16=True,  # Use bfloat16 precision (better than fp16 for A100)
    optim="adamw_torch",
    warmup_ratio=0.03,  # Warm up over 3% of steps
    weight_decay=0.01,
    gradient_checkpointing=True,  # Save memory with gradient checkpointing
    report_to="tensorboard",
    remove_unused_columns=False,
    push_to_hub=False,  # Set to True if you want to upload the model to HF Hub
    ddp_find_unused_parameters=False,  # More efficient distributed training
    torch_compile=True,  # Use PyTorch 2.0 compile for speed if available
    max_grad_norm=0.3,  # Reduce grad norm for stability
    hub_token=HF_TOKEN,  # Add token for potential push to hub
    hub_strategy="checkpoint",  # Only upload at checkpoints to save bandwidth
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation", None),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed! Model saved to:", OUTPUT_DIR)


# Function to test the model (optional)
def test_model():
    print("\nTesting the fine-tuned model...")
    from peft import PeftModel

    def generate_response(prompt, model_path=OUTPUT_DIR):
        # Load the fine-tuned model for inference
        inference_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=HF_TOKEN,
        )
        inference_model = PeftModel.from_pretrained(inference_model, model_path)

        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(
            inference_model.device
        )

        outputs = inference_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract just the assistant's response
        response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        return response

    # Test with a sample from the dataset (if available)
    if len(dataset["train"]) > 0:
        sample_prompt = dataset["train"][0]["prompt"]
        print(f"Sample prompt: {sample_prompt[:100]}...")
        response = generate_response(sample_prompt)
        print(f"Generated response: {response}")


# Uncomment to run test after training
# test_model()

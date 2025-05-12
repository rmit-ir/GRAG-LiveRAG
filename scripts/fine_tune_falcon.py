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
DATA_PATH = "./data/live_rag_questions/combined_judgments_df_prompts.tsv"  # Replace with your TSV file path

# Multi-GPU A100 Optimized Settings
MAX_LENGTH = 1024
BATCH_SIZE = 4  # Increased batch size per GPU for multi-GPU setup
GRADIENT_ACCUMULATION = 8  # Reduced as we have more GPUs (8*4*8 = 256 effective batch)
LORA_R = 8
LORA_ALPHA = 16

# Print GPU information
print(f"Number of available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Load tokenizer with cache directory  
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="left",
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=HF_TOKEN,
)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization for distributed A100 efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_map="auto",  # Let the model decide how to distribute across GPUs
    trust_remote_code=True,
    cache_dir=cache_dir,
    token=HF_TOKEN,
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

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
    ],
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

# Define DeepSpeed configuration for 8 GPUs
deepspeed_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "steps_per_print": 10,
    "train_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION * 8,  # Total batch across all 8 GPUs
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
    "gradient_clipping": 0.3
}

# Save DeepSpeed config to file
import json
with open('ds_config.json', 'w') as f:
    json.dump(deepspeed_config, f)

# Define training arguments optimized for 8x A100 GPUs
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=3,
    save_strategy="epoch",  # Changed from "steps" to "epoch"
    save_total_limit=3,     # Keep only the 3 most recent checkpoints
    logging_steps=10,
    learning_rate=1e-4,
    bf16=True,
    optim="adamw_torch",
    warmup_ratio=0.03,
    weight_decay=0.01,
    gradient_checkpointing=True,
    report_to="none",  # Disable tensorboard to avoid dependencies
    remove_unused_columns=False,
    push_to_hub=False,
    ddp_find_unused_parameters=False,
    torch_compile=False,  # Disable torch.compile as it can be unstable with distributed training
    max_grad_norm=0.3,
    hub_token=HF_TOKEN,
    hub_strategy="checkpoint",
    # DeepSpeed and distributed training settings
    deepspeed="ds_config.json",
    local_rank=-1,  # This will be set by the deepspeed launcher
    fp16=False,  # Use bf16 instead
    sharded_ddp="zero_dp_3",  # Use ZeRO-3 optimization
    # Load best model at the end of training
    load_best_model_at_end=True,  # Added to load best model at end
    metric_for_best_model="loss",  # Use loss as metric to determine best model
    greater_is_better=False,      # Lower loss is better
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
print("Starting distributed fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving model...")
# Make sure to save on the main process only
if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning completed! Model saved to:", OUTPUT_DIR)

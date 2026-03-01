# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers>=4.46.0,<4.48",
#     "trl[peft]>=0.11.0,<0.12",
#     "peft>=0.13.0,<0.15",
#     "datasets",
#     "bitsandbytes",
#     "wandb",
#     "accelerate",
#     "huggingface_hub",
#     "rich",
#     "sentencepiece",
#     "jinja2",
# ]
# ///
"""
Fine-tune a Mistral model for security vulnerability detection using TRL + QLoRA.
Runs on Hugging Face Jobs infrastructure with W&B tracking.

The training dataset is loaded from a HF dataset repo, and LoRA adapters
are pushed to HF Hub when training completes.
"""

import os

import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --- Configuration from environment ---
HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
DATASET_REPO = os.environ["DATASET_REPO"]  # e.g. "username/security-vuln-dataset"
OUTPUT_REPO = os.environ["OUTPUT_REPO"]  # e.g. "username/mistral-nemo-secure-scan"
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

login(token=HF_TOKEN)

# --- W&B setup ---
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_PROJECT"] = "mistral-vibe-security"
    os.environ["WANDB_LOG_MODEL"] = "end"
else:
    os.environ["WANDB_DISABLED"] = "true"

# --- Load dataset ---
print(f"Loading dataset from {DATASET_REPO}...")
dataset = load_dataset(DATASET_REPO)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Validate dataset structure
if "messages" not in train_dataset.column_names:
    available_columns = train_dataset.column_names
    raise ValueError(
        f"Dataset missing 'messages' column. Available columns: {available_columns}"
    )

print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")
print(f"Dataset columns: {train_dataset.column_names}")
print(f"Sample messages length: {len(train_dataset[0]['messages'])} turns")

# --- QLoRA config (4-bit quantization) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# --- Load model + tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

# --- LoRA config ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Training config ---
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    max_seq_length=MAX_SEQ_LENGTH,
    report_to="wandb" if WANDB_API_KEY else "none",
    hub_model_id=OUTPUT_REPO,
    push_to_hub=True,
    hub_token=HF_TOKEN,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# --- Train ---
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()

# --- Push LoRA adapters to Hub ---
print(f"Pushing LoRA adapters to {OUTPUT_REPO}...")
trainer.push_to_hub()

# --- Log model ID as W&B artifact ---
if WANDB_API_KEY:
    artifact = wandb.Artifact("security-scan-lora", type="model")
    artifact.add_dir("./output")
    wandb.log_artifact(artifact)
    wandb.finish()

print(f"Done! Fine-tuned model available at: https://huggingface.co/{OUTPUT_REPO}")

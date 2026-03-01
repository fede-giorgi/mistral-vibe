#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and upload to HF Hub.
Runs on HF Jobs (A10G) to avoid local memory constraints.
"""

import sys
import os

sys.stdout.reconfigure(line_buffering=True)

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
ADAPTER_REPO = "ratnam1510/mistral-small-secure-scan"
MERGED_REPO = "ratnam1510/ministral-8b-security-scanner"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)

    api = HfApi(token=token)

    # Create target repo
    api.create_repo(repo_id=MERGED_REPO, exist_ok=True, private=False)
    print(f"Target repo: {MERGED_REPO}", flush=True)

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load base model in bf16
    print("Loading base model in bf16...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    # Load and merge LoRA
    print(f"Loading LoRA adapter from {ADAPTER_REPO}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)

    print("Merging LoRA into base model...", flush=True)
    model = model.merge_and_unload()

    # Save merged model locally
    save_dir = "/tmp/merged_model"
    print(f"Saving merged model to {save_dir}...", flush=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Report sizes
    total_size = sum(
        f.stat().st_size
        for f in __import__("pathlib").Path(save_dir).rglob("*")
        if f.is_file()
    )
    print(f"Merged model size: {total_size / 1e9:.2f} GB", flush=True)

    # Upload to HF Hub
    print(f"Uploading to {MERGED_REPO}...", flush=True)
    api.upload_folder(
        folder_path=save_dir,
        repo_id=MERGED_REPO,
        repo_type="model",
        commit_message="Upload merged Ministral-8B + security LoRA model",
    )

    # Add model card
    model_card = """---
library_name: transformers
license: apache-2.0
base_model: mistralai/Ministral-8B-Instruct-2410
tags:
  - security
  - vulnerability-detection
  - code-analysis
  - mistral
  - fine-tuned
pipeline_tag: text-generation
---

# Ministral-8B Security Vulnerability Scanner

A fine-tuned version of [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) specialized for **security vulnerability detection** in code.

## Model Details

- **Base model:** mistralai/Ministral-8B-Instruct-2410
- **Fine-tuning method:** QLoRA (r=16, alpha=32)
- **Target modules:** q_proj, k_proj, v_proj, o_proj
- **Training data:** ~864 security vulnerability examples
- **LoRA adapter:** [ratnam1510/mistral-small-secure-scan](https://huggingface.co/ratnam1510/mistral-small-secure-scan)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ratnam1510/ministral-8b-security-scanner")
tokenizer = AutoTokenizer.from_pretrained("ratnam1510/ministral-8b-security-scanner")

messages = [
    {"role": "system", "content": "You are an expert security vulnerability analyst."},
    {"role": "user", "content": "Analyze this code for vulnerabilities:\\n```python\\nimport os\\nos.system(user_input)\\n```"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## API Usage (Inference API)

```python
from huggingface_hub import InferenceClient

client = InferenceClient("ratnam1510/ministral-8b-security-scanner")
response = client.text_generation(
    "Analyze this code for security vulnerabilities: import os; os.system(input())",
    max_new_tokens=512,
)
print(response)
```

## Evaluation

Evaluated against the base Ministral-8B model using LLM-as-judge (GLM-4.5-air):

| Metric | Base | Fine-tuned | Delta |
|--------|------|------------|-------|
| Overall | 1.55/5 | 1.81/5 | +0.26 |
| Vuln ID | 1.90 | 2.31 | +0.41 |
| Severity | 1.60 | 2.62 | +1.02 |

## Limitations

This is an early iteration. The model shows marginal improvements over the base model.
Dataset quality improvements are planned for future iterations.
"""
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=MERGED_REPO,
        repo_type="model",
    )

    print(
        f"\nDONE! Model uploaded to: https://huggingface.co/{MERGED_REPO}", flush=True
    )


if __name__ == "__main__":
    main()

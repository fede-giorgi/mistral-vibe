# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers>=4.48.0",
#     "peft>=0.13.0",
#     "accelerate",
#     "huggingface_hub",
#     "sentencepiece",
# ]
# ///
"""
Merge LoRA adapter into base model and upload to HF Hub.
Runs on HF Jobs (A10G) to avoid local memory constraints.
"""

import sys
import os
import traceback

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=== merge_and_upload.py starting ===", flush=True)

try:
    import torch

    print(
        f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}",
        flush=True,
    )

    from huggingface_hub import HfApi, login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("All imports done.", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
ADAPTER_REPO = "ratnam1510/mistral-small-secure-scan"
MERGED_REPO = "ratnam1510/ministral-8b-security-scanner"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN not set!", flush=True)
        sys.exit(1)

    login(token=token)
    print("Logged in to HF Hub.", flush=True)

    api = HfApi(token=token)

    # Create target repo
    api.create_repo(repo_id=MERGED_REPO, exist_ok=True, private=False)
    print(f"Target repo: {MERGED_REPO}", flush=True)

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=token, trust_remote_code=True
    )
    print("  Tokenizer loaded.", flush=True)

    # Load base model in bf16 on CPU (needs ~16GB RAM, A10G has 46GB)
    print("Loading base model in bf16 on CPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=token,
        trust_remote_code=True,
    )
    print("  Base model loaded.", flush=True)

    # Load and merge LoRA
    print(f"Loading LoRA adapter from {ADAPTER_REPO}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=token)
    print("  LoRA adapter loaded.", flush=True)

    print("Merging LoRA into base model...", flush=True)
    merged = model.merge_and_unload()
    del model, base_model  # free RAM
    import gc

    gc.collect()
    print("  Merge complete. RAM freed.", flush=True)

    # Push directly to Hub (avoids large intermediate disk write)
    print(f"Pushing merged model to {MERGED_REPO}...", flush=True)
    merged.push_to_hub(
        MERGED_REPO,
        token=token,
        commit_message="Upload merged Ministral-8B + security LoRA model",
        max_shard_size="2GB",
    )
    print("  Model pushed.", flush=True)

    print(f"Pushing tokenizer to {MERGED_REPO}...", flush=True)
    tokenizer.push_to_hub(MERGED_REPO, token=token)
    print("  Tokenizer pushed.", flush=True)

    # Add model card
    print("Uploading model card...", flush=True)
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
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

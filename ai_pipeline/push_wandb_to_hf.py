#!/usr/bin/env python3
"""Push a W&B LoRA artifact to Hugging Face as a merged model.

Downloads the adapter from W&B, loads base model + LoRA, merges, and pushes
to a new HF repo. After this you can run remediation with:
  uv run python ai_pipeline/7_remediation.py --model USERNAME/REPO_NAME path/to/file.py

Requires: WANDB_API_KEY, HUGGINGFACE_HUB_TOKEN (or HF_TOKEN).
Memory: ~16GB RAM for a 7B model (loads in float16 on CPU). If you hit OOM,
run this script on Colab or a machine with more RAM.

Usage:
  uv run python ai_pipeline/push_wandb_to_hf.py \\
    --wandb-artifact entity/project/artifact_name:v0 \\
    --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
    --hf-repo username/repo-name
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load ai_pipeline/.env
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push W&B LoRA artifact to Hugging Face as a merged model",
    )
    parser.add_argument(
        "--wandb-artifact",
        required=True,
        help="W&B artifact path (e.g. ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model name (e.g. mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="Hugging Face repo id to create/update (e.g. username/security-scan-merged)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache dir for W&B download (default: XDG_CACHE_HOME/wandb-remediation)",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        print("Error: set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in ai_pipeline/.env", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("WANDB_API_KEY"):
        print("Error: set WANDB_API_KEY in ai_pipeline/.env", file=sys.stderr)
        sys.exit(1)

    # Reuse W&B download and adapter resolution from analyzer_wandb
    from ai_pipeline.remediation.analyzer_wandb import (
        _ensure_deps,
        _download_artifact,
        _find_adapter_dir,
    )
    _ensure_deps()

    import torch
    from huggingface_hub import HfApi, login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    login(token=hf_token)
    cache_root = Path(args.cache_dir or os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "wandb-remediation"

    print("Downloading W&B artifact...")
    adapter_path = _download_artifact(args.wandb_artifact, cache_root)
    adapter_dir = _find_adapter_dir(adapter_path)
    print(f"Adapter at: {adapter_dir}")

    print("Loading base model (float16 on CPU; needs ~14GB RAM for 7B)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging adapter into base...")
    model = model.merge_and_unload()

    print(f"Pushing to https://huggingface.co/{args.hf_repo} ...")
    api = HfApi(token=hf_token)
    try:
        api.create_repo(args.hf_repo, exist_ok=True, private=False)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            print(
                "\n403 Forbidden: your Hugging Face token cannot create repos. "
                "Create a token with **Write** access at https://huggingface.co/settings/tokens "
                "and set HUGGINGFACE_HUB_TOKEN in ai_pipeline/.env",
                file=sys.stderr,
            )
        raise
    tokenizer.push_to_hub(args.hf_repo, token=hf_token)
    model.push_to_hub(args.hf_repo, token=hf_token, max_shard_size="2GB")
    print("Done.")
    print(f"\nRun remediation with:")
    print(f"  uv run python ai_pipeline/7_remediation.py --model {args.hf_repo} path/to/file.py")


if __name__ == "__main__":
    main()

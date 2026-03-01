#!/usr/bin/env python3
"""
Launch the fine-tuning job on Hugging Face Jobs infrastructure.
Uploads the dataset to HF Hub, then kicks off training on a remote GPU.

Usage:
    export HF_TOKEN="hf_..."
    export WANDB_API_KEY="..."
    uv run python scripts/launch_finetune.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, run_uv_job


def upload_dataset(api: HfApi, repo_id: str, token: str) -> None:
    """Upload train/val/test JSONL files to a HF dataset repo."""
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    for split in ["train", "val", "test"]:
        local_path = Path(f"ai_pipeline/dataset/{split}.jsonl")
        if local_path.exists():
            print(f"Uploading {local_path} -> {repo_id}/{split}.jsonl")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"data/{split}.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch fine-tuning on HF Jobs")
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset-repo",
        default=None,
        help="HF dataset repo ID (default: <username>/security-vuln-dataset)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HF model repo for LoRA adapters (default: <username>/mistral-nemo-secure-scan)",
    )
    parser.add_argument(
        "--flavor",
        default="t4-small",
        help="GPU flavor: t4-small, a10g-small, a10g-large, a100-large",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--timeout", default="3h", help="Job timeout")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    if not hf_token:
        print("Error: HF_TOKEN environment variable is required.")
        print("Get one at https://huggingface.co/settings/tokens")
        raise SystemExit(1)

    login(token=hf_token)
    api = HfApi(token=hf_token)
    username = api.whoami()["name"]

    dataset_repo = args.dataset_repo or f"{username}/security-vuln-dataset"
    output_repo = args.output_repo or f"{username}/mistral-small-secure-scan"

    # Step 1: Upload dataset to HF Hub
    print(f"\n📦 Uploading dataset to {dataset_repo}...")
    upload_dataset(api, dataset_repo, hf_token)

    # Step 2: Launch training job on HF infrastructure
    print(f"\n🚀 Launching fine-tuning job on {args.flavor}...")
    print(f"   Model: {args.model}")
    print(f"   Dataset: {dataset_repo}")
    print(f"   Output: {output_repo}")
    print(f"   Epochs: {args.epochs}")

    job = run_uv_job(
        "scripts/finetune_hf.py",
        flavor=args.flavor,
        env={
            "MODEL_NAME": args.model,
            "DATASET_REPO": dataset_repo,
            "OUTPUT_REPO": output_repo,
            "EPOCHS": str(args.epochs),
            "LEARNING_RATE": str(args.learning_rate),
        },
        secrets={
            "HF_TOKEN": hf_token,
            "WANDB_API_KEY": wandb_key,
        },
        timeout=args.timeout,
    )

    print(f"\n✅ Job launched successfully!")
    print(f"   Job ID: {job.id}")
    print(f"   Monitor: {job.url}")
    print(f"   W&B: https://wandb.ai (project: mistral-vibe-security)")
    print(f"\n   Once complete, model will be at: https://huggingface.co/{output_repo}")


if __name__ == "__main__":
    main()

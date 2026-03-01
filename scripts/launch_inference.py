#!/usr/bin/env python3
"""
Launch the inference job on Hugging Face Jobs infrastructure.

Re-uploads the dataset (in case it changed), then kicks off inference
on a remote GPU using the fine-tuned LoRA adapter.

Usage:
    export HF_TOKEN="hf_..."
    uv run python scripts/launch_inference.py
    uv run python scripts/launch_inference.py --sample-size 50 --flavor a10g-small
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, run_uv_job


def upload_dataset(api: HfApi, repo_id: str, token: str) -> None:
    """Upload train/val/test JSONL files to the HF dataset repo."""
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    split_renames = {"train": "train", "val": "validation", "test": "test"}
    for local_split, hub_split in split_renames.items():
        local_path = Path(f"ai_pipeline/dataset/{local_split}.jsonl")
        if local_path.exists():
            print(f"  Uploading {local_path} -> {repo_id}/data/{hub_split}.jsonl")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"data/{hub_split}.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch inference on HF Jobs")
    parser.add_argument(
        "--dataset-repo",
        default=None,
        help="HF dataset repo (default: <username>/security-vuln-dataset)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HF model repo with LoRA adapter (default: <username>/mistral-small-secure-scan)",
    )
    parser.add_argument(
        "--flavor",
        default="t4-small",
        help="GPU flavor: t4-small ($0.60/hr), a10g-small ($1.05/hr), a10g-large, a100-large",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of test examples to run inference on",
    )
    parser.add_argument(
        "--timeout", default="1h", help="Job timeout (inference is fast, 1h is plenty)"
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip re-uploading the dataset"
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    if not hf_token:
        print("Error: HF_TOKEN environment variable is required.")
        print("Get one at https://huggingface.co/settings/tokens")
        raise SystemExit(1)

    login(token=hf_token)
    api = HfApi(token=hf_token)
    username = api.whoami()["name"]

    dataset_repo = args.dataset_repo or f"{username}/security-vuln-dataset"
    output_repo = args.output_repo or f"{username}/mistral-small-secure-scan"

    # Step 1: Re-upload dataset (test set might have changed)
    if not args.skip_upload:
        print(f"\nUploading dataset to {dataset_repo}...")
        upload_dataset(api, dataset_repo, hf_token)
        print("  Done.")
    else:
        print("\nSkipping dataset upload (--skip-upload).")

    # Step 2: Launch inference job
    print(f"\nLaunching inference job on {args.flavor}...")
    print(f"  Dataset: {dataset_repo}")
    print(f"  LoRA adapter: {output_repo}")
    print(f"  Sample size: {args.sample_size}")

    job = run_uv_job(
        "scripts/inference_hf.py",
        flavor=args.flavor,
        env={
            "DATASET_REPO": dataset_repo,
            "OUTPUT_REPO": output_repo,
            "SAMPLE_SIZE": str(args.sample_size),
        },
        secrets={"HF_TOKEN": hf_token},
        timeout=args.timeout,
    )

    print(f"\nJob launched!")
    print(f"  Job ID: {job.id}")
    print(f"  Monitor: {job.url}")
    print(f"\n  When complete, results will be at:")
    print(
        f"    https://huggingface.co/datasets/{dataset_repo}/blob/main/eval/inference_results.json"
    )
    print(f"\n  Then run the Gemini judge locally:")
    print(f"    GEMINI_API_KEY=... uv run python scripts/judge_gemini.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Launch the merge-and-upload job on HF Jobs.

Merges the LoRA adapter into the base model and uploads
the full merged model to HF Hub.

Usage:
    uv run python scripts/launch_merge.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi, login, run_uv_job


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    if not hf_token:
        print("Error: HF_TOKEN not found.")
        raise SystemExit(1)

    login(token=hf_token)

    print("Launching merge job on a10g-small...")

    job = run_uv_job(
        "scripts/merge_and_upload.py",
        flavor="a10g-small",
        secrets={"HF_TOKEN": hf_token},
        timeout="2h",
    )

    print(f"Job launched!")
    print(f"  Job ID: {job.id}")
    print(f"  Monitor: {job.url}")
    print(f"\n  When complete, model will be at:")
    print(f"    https://huggingface.co/ratnam1510/ministral-8b-security-scanner")


if __name__ == "__main__":
    main()

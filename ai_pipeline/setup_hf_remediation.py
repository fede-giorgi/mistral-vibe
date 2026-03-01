#!/usr/bin/env python3
"""Hugging Face remediation setup: check env, suggest repo id, and run remediation.

Run from project root. Ensures .env is loaded from ai_pipeline/ and validates
HUGGINGFACE_HUB_TOKEN / MISTRAL_API_KEY, then prints the exact command to run
or runs a quick test.

Usage:
  uv run python ai_pipeline/setup_hf_remediation.py
  uv run python ai_pipeline/setup_hf_remediation.py --model username/repo-name
  uv run python ai_pipeline/setup_hf_remediation.py --model username/repo-name --test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Load ai_pipeline/.env when run from project root
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)


def check_env() -> tuple[bool, list[str]]:
    errors: list[str] = []
    hf = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "").strip()
    mistral = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if not hf:
        errors.append("HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) is not set in ai_pipeline/.env")
    if not mistral:
        errors.append("MISTRAL_API_KEY is not set in ai_pipeline/.env")
    return len(errors) == 0, errors


def get_hf_username() -> str | None:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"))
        return api.whoami()["name"]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Hugging Face remediation setup and run with your model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model repo id (username/repo-name). If omitted, only checks env and suggests command.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick remediation test on ai_pipeline/test_repo/auth.py",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="File to analyze (default for --test: ai_pipeline/test_repo/auth.py)",
    )
    args = parser.parse_args()

    ok, errors = check_env()
    if not ok:
        print("Setup issues (fix ai_pipeline/.env or export the vars):")
        for e in errors:
            print(f"  - {e}")
        print("\nCreate ai_pipeline/.env with:")
        print("  HUGGINGFACE_HUB_TOKEN=hf_...")
        print("  MISTRAL_API_KEY=...")
        print("\nGet an HF token: https://huggingface.co/settings/tokens")
        sys.exit(1)

    username = get_hf_username()
    default_repo = f"{username}/mistral-small-secure-scan" if username else "USERNAME/mistral-small-secure-scan"

    if not args.model:
        print("Environment looks good.")
        if username:
            print(f"  HF user: {username}")
            print(f"  Suggested repo id: {default_repo}")
        print("\nRun remediation with your Hugging Face model:")
        print(f"  uv run python ai_pipeline/7_remediation.py --model {default_repo} path/to/file.py")
        print("\nReplace the repo id if your model has a different name (e.g. from launch_finetune.py --output-repo).")
        return

    if args.test or args.file:
        file_path = args.file or "ai_pipeline/test_repo/auth.py"
        if not Path(file_path).is_file():
            print(f"Error: file not found: {file_path}")
            sys.exit(1)
        cmd = [
            sys.executable,
            "ai_pipeline/7_remediation.py",
            "--model",
            args.model,
            file_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        repo_root = Path(__file__).resolve().parent.parent
        sys.exit(subprocess.run(cmd, cwd=repo_root).returncode)

    print(f"To run remediation with --model {args.model}:")
    print(f"  uv run python ai_pipeline/7_remediation.py --model {args.model} path/to/file.py")
    print("Or add --test to run on ai_pipeline/test_repo/auth.py")


if __name__ == "__main__":
    main()

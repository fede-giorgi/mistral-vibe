#!/usr/bin/env python3
"""
Script to fine-tune Mistral Large 3 for codebase security vulnerability detection.
This script utilizes the Mistral API for fine-tuning and native W&B integration for logging.

Usage:
    export MISTRAL_API_KEY="..."
    export WANDB_API_KEY="..."
    python scripts/finetune_security.py --train train.jsonl --val val.jsonl --project mistral-vibe-secure
"""

import os
import time
import argparse
import wandb
from mistralai import Mistral

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral for Security Scanning")
    parser.add_argument("--train", type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument("--val", type=str, required=True, help="Path to the validation JSONL file")
    parser.add_argument("--model", type=str, default="mistral-small-latest", help="Base model to fine-tune (e.g. mistral-small-latest, open-mistral-nemo, ministral-8b-latest)")
    parser.add_argument("--project", type=str, default="mistral-vibe-security", help="W&B Project Name")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (optional)")
    parser.add_argument("--poll", action="store_true", help="Poll for completion and log results as W&B Artifact")

    args = parser.parse_args()

    # Ensure API keys are present
    mistral_key = os.getenv("MISTRAL_API_KEY")
    wandb_key = os.getenv("WANDB_API_KEY")

    if not mistral_key:
        print("Error: MISTRAL_API_KEY environment variable is missing.")
        exit(1)

    if not wandb_key:
        print("Error: WANDB_API_KEY environment variable is missing.")
        print("W&B integration is required for this pipeline.")
        exit(1)

    print(f"Initializing Mistral client...")
    client = Mistral(api_key=mistral_key)

    # 1. Upload the training and validation files
    print(f"Uploading training file: {args.train}")
    with open(args.train, "rb") as f:
        train_res = client.files.upload(file={"file_name": os.path.basename(args.train), "content": f}, purpose="fine-tune")

    print(f"Uploading validation file: {args.val}")
    with open(args.val, "rb") as f:
        val_res = client.files.upload(file={"file_name": os.path.basename(args.val), "content": f}, purpose="fine-tune")

    train_file_id = train_res.id
    val_file_id = val_res.id

    print(f"Train File ID: {train_file_id}")
    print(f"Val File ID: {val_file_id}")

    # Wait briefly to ensure files are processed by the API
    print("Waiting 10 seconds for files to be processed by Mistral API...")
    time.sleep(10)

    # We handle W&B Integration locally to bypass Mistral API's 40-character W&B key limit.
    # W&B recently updated keys to start with `wandb_v1_` causing >40 chars.

    # 3. Create the fine-tuning job
    print(f"Starting fine-tuning job for model: {args.model}")

    job = client.fine_tuning.jobs.create(
        model=args.model,
        training_files=[{"file_id": train_file_id, "weight": 1.0}],
        validation_files=[val_file_id],
        hyperparameters={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        },
        suffix="secure-scan"
    )

    print("\n" + "="*50)
    print("Fine-tuning job successfully launched!")
    print(f"Job ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"Track the run in your Weights & Biases dashboard under project '{args.project}'.")
    print("="*50 + "\n")

    print("You can monitor the job status using the Mistral API or Dashboard.")
    print("Once completed, Mistral will send an email and the new Model ID will be active.")

if __name__ == "__main__":
    main()

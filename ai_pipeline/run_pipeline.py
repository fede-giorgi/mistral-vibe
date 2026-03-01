#!/usr/bin/env python3
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    print("=== Pipeline Step 1: Fetching GitHub Security Advisories ===")
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        # Fallback to fetching it on mac
        import subprocess
        try:
            token = subprocess.check_output(["gh", "auth", "token"]).decode("utf-8").strip()
        except:
            print("Error: Please set GITHUB_TOKEN environment variable")
            sys.exit(1)
            
    fetch_github = load_module("fetch_github", "ai_pipeline/1_fetch_github.py")
    advisories = fetch_github.fetch_github_advisories(token, limit=1500)
    print(f"-> Fetched {len(advisories)} GitHub Advisories with fix commits.")

    print("\n=== Pipeline Step 2: Fetching BigVul Summaries ===")
    fetch_bigvul = load_module("fetch_bigvul", "ai_pipeline/2_fetch_bigvul.py")
    df = fetch_bigvul.get_stratified_sample(fetch_bigvul.df, 5000) 
    bigvul_data = fetch_bigvul.process_to_json(df, num_rows=len(df))
    print(f"-> Sampled {len(bigvul_data)} vulnerable code blocks from BigVul.")
    
    print("\n=== Pipeline Step 3: Pairing Datasets ===")
    pair_datasets = load_module("pair_datasets", "ai_pipeline/3_pair_datasets.py")
    paired_output_file = os.path.join("ai_pipeline", "paired_data.json")
    pair_datasets.pair_datasets_and_export(advisories, bigvul_data, paired_output_file)
    
    print("\n=== Pipeline Step 4: Merging with SARD and Splitting ===")
    merge_split = load_module("merge_split", "ai_pipeline/4_merge_and_split.py")
    sard_file = os.path.join("ai_pipeline", "chat_dataset.jsonl")
    output_dir = os.path.join("ai_pipeline", "dataset")
    merge_split.merge_and_split_datasets(paired_output_file, sard_file, output_dir)
    
    print("\nPipeline Complete! Datasets are ready for fine-tuning.")

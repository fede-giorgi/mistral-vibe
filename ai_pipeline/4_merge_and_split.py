#!/usr/bin/env python3
"""
Merge and split datasets with proper shuffling.
This script combines paired GitHub+BigVul data with SARD data,
shuffles them together with a fixed seed, and splits into train/val/test sets.
"""

import os
import json
import random
from typing import List, Dict, Any

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line: {line[:50]}...")
    return data

def save_jsonl_file(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save list of dictionaries to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def merge_and_split_datasets(paired_data_file: str, sard_data_file: str, output_dir: str, random_seed: int = 42) -> None:
    """
    Merge paired GitHub+BigVul data with SARD data, shuffle, and split into train/val/test sets.
    
    Args:
        paired_data_file: Path to the paired GitHub+BigVul JSONL file
        sard_data_file: Path to the SARD dataset JSONL file
        output_dir: Directory to save train/val/test splits
        random_seed: Random seed for reproducible shuffling
    """
    print("Loading paired GitHub+BigVul data...")
    paired_data = load_jsonl_file(paired_data_file)
    print(f"Loaded {len(paired_data)} paired examples")
    
    print("Loading SARD data...")
    sard_data = load_jsonl_file(sard_data_file)
    print(f"Loaded {len(sard_data)} SARD examples")
    
    # Combine datasets
    combined_data = paired_data + sard_data
    print(f"Combined dataset size: {len(combined_data)} examples")
    
    # Shuffle with fixed seed for reproducibility
    print(f"Shuffling combined dataset with seed {random_seed}...")
    random.seed(random_seed)
    random.shuffle(combined_data)
    
    # Split into train/val/test (80/10/10)
    total_size = len(combined_data)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:train_size + val_size]
    test_data = combined_data[train_size + val_size:]
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_data)} examples ({len(train_data)/total_size:.1%})")
    print(f"  Val: {len(val_data)} examples ({len(val_data)/total_size:.1%})")
    print(f"  Test: {len(test_data)} examples ({len(test_data)/total_size:.1%})")
    
    # Save splits
    print(f"Saving splits to {output_dir}...")
    save_jsonl_file(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl_file(val_data, os.path.join(output_dir, "val.jsonl"))
    save_jsonl_file(test_data, os.path.join(output_dir, "test.jsonl"))
    
    print("Dataset merging and splitting complete!")

if __name__ == "__main__":
    # Default paths when running standalone
    paired_file = os.path.join("ai_pipeline", "paired_data.json")
    sard_file = os.path.join("ai_pipeline", "chat_dataset.jsonl")
    output_dir = os.path.join("ai_pipeline", "dataset")
    
    merge_and_split_datasets(paired_file, sard_file, output_dir)
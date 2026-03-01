import json
import random
import os

def process_and_split():
    """
    The 'Brain' of the pipeline:
    1. Loads raw data from all sources.
    2. Pairs GitHub Advisories with BigVul samples using CWE IDs.
    3. Formats everything into Mistral-compatible ChatML format.
    4. Merges with SARD synthetic data.
    5. Applies a global shuffle and splits into Train/Val/Test sets.
    """
    print("Starting data processing and pairing...")

    # Define absolute paths relative to the script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "dataset")

    # 1. Load Raw Data
    github_path = os.path.join(data_dir, "raw_github.json")
    bigvul_path = os.path.join(data_dir, "raw_bigvul.json")
    sard_path = os.path.join(data_dir, "raw_sard.json")

    with open(github_path, "r") as f: github = json.load(f)
    with open(bigvul_path, "r") as f: bigvul = json.load(f)
    with open(sard_path, "r") as f: sard = json.load(f)

    unified_dataset = []

    # 2. Logic: Pair GitHub Advisories with BigVul by CWE
    # Create a map to quickly look up BigVul samples by their violation type (CWE)
    cwe_map = {}
    for item in bigvul:
        cwe = str(item.get("violation_type", "")).strip()
        if cwe not in cwe_map:
            cwe_map[cwe] = []
        cwe_map[cwe].append(item)

    # For each advisory, find matching code samples and format into a conversation
    for adv in github:
        for cwe in adv["cwes"]:
            if cwe in cwe_map:
                # Pick a random sample for this specific CWE to ensure variety
                sample = random.choice(cwe_map[cwe])

                # Format into Mistral ChatML structure
                unified_dataset.append({
                    "messages": [
                        {"role": "system", "content": "You are a senior security researcher analyzing code for vulnerabilities."},
                        {"role": "user", "content": f"Analyze this code for {adv['summary']}:\n\n{sample['vulnerable_code']}"},
                        {"role": "assistant", "content": f"VIOLATION: {cwe}\nREASON: {adv['description']}\nFIX:\n{sample['safe_code']}"}
                    ]
                })

    # 3. Add SARD data (already in conversational format)
    unified_dataset.extend(sard)

    # 4. Global Shuffle & Split (The Golden Rule of Data Science)
    # Using a fixed seed (42) ensures reproducible splits
    random.seed(42)
    random.shuffle(unified_dataset)

    n = len(unified_dataset)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = unified_dataset[:train_end]
    val = unified_dataset[train_end:val_end]
    test = unified_dataset[val_end:]

    # 5. Export to JSONL (one JSON object per line)
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        file_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(file_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    print(f"FINAL DATASET READY: {len(train)} train, {len(val)} val, {len(test)} test.")

if __name__ == "__main__":
    process_and_split()

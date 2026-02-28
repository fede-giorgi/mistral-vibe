import os
import json
import random
import pandas as pd
from typing import List, Dict

def pair_datasets_and_export(advisories: List[Dict], bigvul_data: List[Dict], output_dir: str):
    """
    Pairs advisories with BigVul based on CWE overlap.
    Formats records for Mistral Fine-Tuning: JSONL with roles.
    """
    print("Pairing datasets based on CWE overlaps...")
    dataset = []

    # Map CWEs to available BigVul examples
    cwe_to_bigvul = {}
    for item in bigvul_data:
        cwe = str(item.get("violation_type", "")).strip()
        # Normalize CWE format if needed
        if not cwe.startswith("CWE-"):
            cwe = f"CWE-{cwe}"

        if cwe not in cwe_to_bigvul:
            cwe_to_bigvul[cwe] = []
        cwe_to_bigvul[cwe].append(item)

    # Pair advisories
    for adv in advisories:
        adv_cwes = adv.get("cwes", [])
        paired = False

        for cwe in adv_cwes:
            # We assume github CWEs might come as 'CWE-xx'
            if cwe in cwe_to_bigvul and cwe_to_bigvul[cwe]:
                # Pick a random overlapping vulnerable code context
                match = random.choice(cwe_to_bigvul[cwe])
                vul_code = match["vulnerable_code"]

                # Format for Mistral:
                messages = [
                    {"role": "system", "content": "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation."},
                    {"role": "user", "content": f"Analyze the following code for security vulnerabilities:\n\n```\n{vul_code}\n```"},
                    {"role": "assistant", "content": f"{adv['summary']}\n\n{adv['description']}"}
                ]
                dataset.append({"messages": messages})
                paired = True
                break

    print(f"Successfully generated {len(dataset)} paired examples.")

    if not dataset:
        print("No matches found! Please ensure CWE IDs match formats between datasets.")
        return

    # 80 / 10 / 10 Split
    random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))

    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]

    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    write_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    write_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))

    print(f"Exported to {output_dir}:")
    print(f" - train.jsonl ({len(train_data)} examples)")
    print(f" - val.jsonl ({len(val_data)} examples)")
    print(f" - test.jsonl ({len(test_data)} examples)")

if __name__ == "__main__":
    # Test stub: If we want to run this standalone, we'd load the exported JSONs from steps 1 & 2
    print("This script is meant to be imported, linking steps 1 and 2.")

import pandas as pd
import json
import os

def fetch_bigvul_raw(sample_size=1000):
    """
    Fetches the BigVul dataset directly from Hugging Face using pandas.
    Extracts raw columns and saves them to a local JSON for processing.
    """
    # Absolute path setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "data", "raw_bigvul.json")

    # Hugging Face dataset path
    hf_path = "hf://datasets/bstee615/bigvul/data/train-00000-of-00001-c6410a8bb202ca06.parquet"

    print(f" > Downloading BigVul dataset from Hugging Face...")

    try:
        # Load parquet directly from the web
        df = pd.read_parquet(hf_path)

        # Original script used stratified sampling, let's take a safe sample to avoid memory issues
        # and ensure we have a good mix of vulnerable (1) and safe (0) code.
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        # Map raw columns to our internal names for the Brain (Step 4)
        # CWE ID -> violation_type
        # func_before -> vulnerable_code
        # func_after -> safe_code
        raw_data = []
        for _, row in df.iterrows():
            raw_data.append({
                "violation_type": str(row.get('CWE ID', 'security_flaw')),
                "vulnerable_code": row.get('func_before', ''),
                "safe_code": row.get('func_after', ''),
                "vul": row.get('vul', 0)
            })

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save to raw JSON
        with open(save_path, "w") as f:
            json.dump(raw_data, f)

        print(f"Successfully downloaded and saved {len(raw_data)} BigVul samples.")

    except Exception as e:
        print(f"Failed to fetch BigVul from Hugging Face: {e}")
        print("Make sure you have 'fsspec' and 'huggingface_hub' installed: pip install fsspec huggingface_hub")

if __name__ == "__main__":
    fetch_bigvul_raw()

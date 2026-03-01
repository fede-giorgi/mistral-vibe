import json
import os

def prepare_sard_raw():
    """
    Reads the SARD dataset from a JSONL file and converts it into a standardized
    raw JSON format. This ensures consistency across all data extractors.
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define absolute paths for input and output
    input_path = os.path.join(base_dir, "chat_dataset.jsonl")
    save_path = os.path.join(base_dir, "data", "raw_sard.json")

    print(f" > Preparing SARD dataset from {input_path}...")

    sard_data = []

    # Check if the source file exists before attempting to read
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            for line in f:
                # Skip empty lines and parse valid JSON lines
                if line.strip():
                    sard_data.append(json.loads(line))
    else:
        print(f"Warning: SARD source file not found at {input_path}. Proceeding with empty list.")

    # Ensure the target directory (data/) exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the consolidated list into a single JSON file
    with open(save_path, "w") as f:
        json.dump(sard_data, f)

    print(f"Prepared {len(sard_data)} SARD samples in {save_path}")

if __name__ == "__main__":
    prepare_sard_raw()

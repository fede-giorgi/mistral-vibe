import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

def run_pipeline():
    """
    Main Orchestrator for the Ward Data Pipeline.
    Sequentially executes extraction, preparation, and merging scripts.
    Ensures the 'cwd' is set correctly so relative paths work across steps.
    """
    # Get the absolute path of the directory where this script lives
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the execution sequence
    scripts = [
        "1_fetch_github.py",   # Step 1: Raw GitHub Advisory extraction
        "2_fetch_bigvul.py",   # Step 2: Raw BigVul sample extraction
        "3_prepare_sard.py",   # Step 3: Raw SARD synthetic data preparation
        "4_merge_and_split.py" # Step 4: The Brain - Pairing, Unification & Shuffling
    ]

    print("WARD PIPELINE: Starting full data orchestration...")

    for script in scripts:
        script_path = os.path.join(base_dir, script)

        if not os.path.exists(script_path):
            print(f"Error: Script {script} not found at {script_path}")
            return

        print(f"\n--- Running Stage: {script} ---")

        try:
            # Execute the script setting the base_dir as the working directory
            subprocess.run(["python", script_path], check=True, cwd=base_dir)
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed at stage {script}. Error: {e}")
            return

    print("\n" + "="*40)
    print("ALL DONE! The dataset is unified, shuffled, and ready.")
    print("Check 'ai_pipeline/dataset/' for the final .jsonl files.")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()

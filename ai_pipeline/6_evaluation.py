#%%
import json
import os
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import mistralai
from mistralai.client import MistralClient
import wandb
from tqdm import tqdm
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("MISTRAL_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL", "mistral-large-latest")
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "il-vostro-model-id-custom")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "ai_pipeline/dataset/test.jsonl")
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Initialize clients
client = MistralClient(api_key=API_KEY)
wandb.init(project="ward-security-benchmark", name="base-vs-finetuned")


#%%
def get_model_response(model_id: str, prompt: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """
    Get response from Mistral model with retry logic.

    Args:
        model_id: ID of the model to query
        prompt: Input prompt for the model
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary containing model response or error information
    """
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": f"Analyze this code for security violations:\n\n{prompt}"}]
            response = client.chat(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"}
            )

            # Validate response structure
            if not response.choices or not hasattr(response.choices[0], 'message'):
                raise ValueError("Invalid response structure from API")

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                return {"error": str(e), "attempts": max_retries}

            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate test data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of test data entries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found at {file_path}")

    test_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                # Basic validation
                if not isinstance(entry.get("messages"), list) or len(entry["messages"]) < 2:
                    logger.warning(f"Invalid entry format at line {line_num}")
                    continue

                test_data.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {str(e)}")
                continue

    if not test_data:
        raise ValueError("No valid test data found in file")

    return test_data

def evaluate_model_performance(test_data: List[Dict[str, Any]], sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate both base and finetuned models on test data.

    Args:
        test_data: List of test entries
        sample_size: Number of samples to evaluate (None for all)

    Returns:
        Dictionary containing evaluation results and metrics
    """
    results = []
    data_subset = test_data[:sample_size] if sample_size else test_data

    for entry in tqdm(data_subset, desc="Evaluating models"):
        try:
            vulnerable_code = entry["messages"][0]["content"]
            ground_truth = json.loads(entry["messages"][1]["content"])

            # Query both models
            base_out = get_model_response(BASE_MODEL, vulnerable_code)
            ft_out = get_model_response(FINETUNED_MODEL, vulnerable_code)

            # Check if responses contain expected fields
            gt_violation = ground_truth.get("violation_type")
            ft_correct = ft_out.get("violation_type") == gt_violation
            base_correct = base_out.get("violation_type") == gt_violation

            results.append({
                "code": vulnerable_code,
                "ground_truth": ground_truth,
                "base_model": base_out,
                "finetuned_model": ft_out,
                "ft_correct_type": ft_correct,
                "base_correct_type": base_correct
            })

        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            continue

    return results

def create_wandb_table(results: List[Dict[str, Any]]) -> wandb.Table:
    """
    Create Weights & Biases table for visualization.

    Args:
        results: Evaluation results

    Returns:
        W&B Table object
    """
    table = wandb.Table(columns=["Code", "GT Violation", "Base Output", "FT Output", "FT Correct"])

    for r in results:
        table.add_data(
            r["code"][:100] + "..." if len(r["code"]) > 100 else r["code"],
            r["ground_truth"].get("violation_type", "N/A"),
            str(r["base_model"]),
            str(r["finetuned_model"]),
            r["ft_correct_type"]
        )

    return table

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from evaluation results.

    Args:
        results: Evaluation results

    Returns:
        Dictionary containing accuracy metrics
    """
    if not results:
        return {"ft_accuracy": 0.0, "base_accuracy": 0.0}

    ft_accuracy = sum(1 for r in results if r["ft_correct_type"]) / len(results)
    base_accuracy = sum(1 for r in results if r["base_correct_type"]) / len(results)

    return {
        "ft_accuracy": round(ft_accuracy, 4),
        "base_accuracy": round(base_accuracy, 4)
    }

def main():
    """
    Main evaluation pipeline.
    """
    try:
        # Load test data
        logger.info("Loading test data...")
        test_data = load_test_data(TEST_DATA_PATH)
        logger.info(f"Loaded {len(test_data)} test samples")

        # Evaluate models (using subset of 100 for speed)
        logger.info("Evaluating models...")
        results = evaluate_model_performance(test_data, sample_size=100)

        # Create visualization table
        logger.info("Creating visualization table...")
        table = create_wandb_table(results)
        wandb.log({"comparison_table": table})

        # Calculate and log metrics
        metrics = calculate_metrics(results)
        wandb.log(metrics)

        logger.info(f"Benchmark completed successfully!")
        logger.info(f"FT Accuracy: {metrics['ft_accuracy']} vs Base: {metrics['base_accuracy']}")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        wandb.log({"error": str(e)})
        raise

if __name__ == "__main__":
    main()

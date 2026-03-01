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
from enum import Enum

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

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FixScore(Enum):
    INCORRECT = 1
    PARTIALLY_CORRECT = 2
    REDUCES_RISK = 3
    CORRECT = 4
    EXCELLENT = 5

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
            messages = [{"role": "user", "content": f"Analyze this code for security violations:

{prompt}"}]
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

def evaluate_vulnerability_detection(model_output: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate vulnerability detection (binary classification).

    Args:
        model_output: Model's response
        ground_truth: Ground truth labels

    Returns:
        Detection evaluation metrics
    """
    gt_vulnerable = ground_truth.get("is_vulnerable", False)
    model_vulnerable = model_output.get("is_vulnerable", False)

    return {
        "tp": int(gt_vulnerable and model_vulnerable),
        "fp": int(not gt_vulnerable and model_vulnerable),
        "tn": int(not gt_vulnerable and not model_vulnerable),
        "fn": int(gt_vulnerable and not model_vulnerable),
        "correct": gt_vulnerable == model_vulnerable
    }

def evaluate_severity_classification(model_output: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate severity classification (categorical).

    Args:
        model_output: Model's response
        ground_truth: Ground truth labels

    Returns:
        Severity evaluation metrics
    """
    gt_severity = ground_truth.get("severity")
    model_severity = model_output.get("severity")

    return {
        "correct": gt_severity == model_severity,
        "gt_severity": gt_severity,
        "model_severity": model_severity
    }

def evaluate_fix_quality(model_output: Dict[str, Any], ground_truth: Dict[str, Any], code: str) -> Dict[str, Any]:
    """
    Evaluate fix quality using LLM-as-a-judge approach.

    Args:
        model_output: Model's response
        ground_truth: Ground truth labels
        code: Original vulnerable code

    Returns:
        Fix quality evaluation
    """
    # Get the proposed fix from model output
    proposed_fix = model_output.get("fix", "")
    
    # Create evaluation prompt for judge model
    judge_prompt = f"""
    Evaluate the quality of this security fix:

    Original vulnerable code:
    {code}

    Proposed fix:
    {proposed_fix}

    Ground truth vulnerability: {ground_truth.get("violation_type", "Unknown")}
    Ground truth severity: {ground_truth.get("severity", "Unknown")}

    Score the fix on a 1-5 scale:
    1 = incorrect or dangerous
    2 = partially correct but flawed
    3 = reduces risk but incomplete
    4 = correct and secure
    5 = correct, secure, and well implemented

    Provide:
    1. Score (1-5)
    2. Brief justification
    3. Confidence level (0-1)
    """

    # Use base model as judge (could also use dedicated judge model)
    judge_response = get_model_response(BASE_MODEL, judge_prompt)
    
    # Parse judge response
    score = judge_response.get("score", 3)
    justification = judge_response.get("justification", "No justification provided")
    confidence = judge_response.get("confidence", 0.7)

    return {
        "score": score,
        "justification": justification,
        "confidence": confidence
    }

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
            code = entry["messages"][0]["content"]
            ground_truth = json.loads(entry["messages"][1]["content"])

            # Query both models
            base_out = get_model_response(BASE_MODEL, code)
            ft_out = get_model_response(FINETUNED_MODEL, code)

            # Skip if errors occurred
            if "error" in base_out or "error" in ft_out:
                logger.warning(f"Skipping entry due to API errors")
                continue

            # Evaluate vulnerability detection
            base_detection = evaluate_vulnerability_detection(base_out, ground_truth)
            ft_detection = evaluate_vulnerability_detection(ft_out, ground_truth)

            # Evaluate severity (only for vulnerable cases)
            base_severity = evaluate_severity_classification(base_out, ground_truth) if ground_truth.get("is_vulnerable") else None
            ft_severity = evaluate_severity_classification(ft_out, ground_truth) if ground_truth.get("is_vulnerable") else None

            # Evaluate fix quality (only for vulnerable cases)
            base_fix = evaluate_fix_quality(base_out, ground_truth, code) if ground_truth.get("is_vulnerable") else None
            ft_fix = evaluate_fix_quality(ft_out, ground_truth, code) if ground_truth.get("is_vulnerable") else None

            results.append({
                "code": code,
                "ground_truth": ground_truth,
                "base_model": base_out,
                "finetuned_model": ft_out,
                "base_detection": base_detection,
                "ft_detection": ft_detection,
                "base_severity": base_severity,
                "ft_severity": ft_severity,
                "base_fix": base_fix,
                "ft_fix": ft_fix
            })

        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            continue

    return results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics from evaluation results.

    Args:
        results: Evaluation results

    Returns:
        Dictionary containing all metrics
    """
    if not results:
        return {
            "detection_accuracy": 0.0,
            "detection_precision": 0.0,
            "detection_recall": 0.0,
            "detection_f1": 0.0,
            "detection_fpr": 0.0,
            "severity_accuracy": 0.0,
            "avg_fix_score": 0.0
        }

    # Detection metrics
    total_tp = sum(r["ft_detection"]["tp"] for r in results)
    total_fp = sum(r["ft_detection"]["fp"] for r in results)
    total_tn = sum(r["ft_detection"]["tn"] for r in results)
    total_fn = sum(r["ft_detection"]["fn"] for r in results)

    total_correct = sum(r["ft_detection"]["correct"] for r in results)
    detection_accuracy = total_correct / len(results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

    # Severity metrics (only for vulnerable cases)
    vulnerable_results = [r for r in results if r["ground_truth"].get("is_vulnerable")]
    severity_correct = sum(1 for r in vulnerable_results if r["ft_severity"]["correct"]) if vulnerable_results else 0
    severity_accuracy = severity_correct / len(vulnerable_results) if vulnerable_results else 0

    # Fix quality metrics (only for vulnerable cases)
    avg_fix_score = sum(r["ft_fix"]["score"] for r in vulnerable_results) / len(vulnerable_results) if vulnerable_results else 0

    return {
        "detection_accuracy": round(detection_accuracy, 4),
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "detection_f1": round(f1, 4),
        "detection_fpr": round(fpr, 4),
        "severity_accuracy": round(severity_accuracy, 4),
        "avg_fix_score": round(avg_fix_score, 4)
    }

def create_wandb_table(results: List[Dict[str, Any]]) -> wandb.Table:
    """
    Create Weights & Biases table for visualization.

    Args:
        results: Evaluation results

    Returns:
        W&B Table object
    """
    table = wandb.Table(columns=[
        "Code", "GT Vulnerable", "GT Severity", "Base Correct", 
        "FT Correct", "FT Severity Correct", "FT Fix Score"
    ])

    for r in results:
        gt_vuln = r["ground_truth"].get("is_vulnerable", False)
        gt_severity = r["ground_truth"].get("severity", "N/A")
        
        base_correct = r["base_detection"]["correct"]
        ft_correct = r["ft_detection"]["correct"]
        ft_severity_correct = r["ft_severity"]["correct"] if r["ft_severity"] else None
        ft_fix_score = r["ft_fix"]["score"] if r["ft_fix"] else None

        table.add_data(
            r["code"][:100] + "..." if len(r["code"]) > 100 else r["code"],
            gt_vuln,
            gt_severity,
            base_correct,
            ft_correct,
            ft_severity_correct,
            ft_fix_score
        )

    return table

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

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(results)

        # Create visualization table
        logger.info("Creating visualization table...")
        table = create_wandb_table(results)
        wandb.log({"comparison_table": table})

        # Log all metrics
        wandb.log(metrics)

        logger.info(f"Benchmark completed successfully!")
        logger.info(f"Detection Accuracy: {metrics['detection_accuracy']}")
        logger.info(f"Detection F1: {metrics['detection_f1']}")
        logger.info(f"Severity Accuracy: {metrics['severity_accuracy']}")
        logger.info(f"Avg Fix Score: {metrics['avg_fix_score']}")

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        wandb.log({"error": str(e)})
        raise

if __name__ == "__main__":
    main()
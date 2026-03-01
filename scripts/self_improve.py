"""
Self-improving fine-tuning orchestrator.

Runs the full Train -> Evaluate -> Optimize loop:
  1. TRAIN: Launch fine-tuning job on HF Jobs (or wait for existing one)
  2. EVAL:  Run the fine-tuned model against the test set with LLM-as-judge
  3. OPTIMIZE: Analyze failures, generate improved + synthetic training data
  4. Repeat with augmented dataset

Each iteration is tracked in W&B with metrics, results, and dataset versions.

Usage:
    export HF_TOKEN="hf_..."
    export MISTRAL_API_KEY="..."
    export WANDB_API_KEY="..."

    # Run evaluation + optimization on existing trained model:
    uv run python scripts/self_improve.py --skip-train --iteration 1

    # Full loop from scratch:
    uv run python scripts/self_improve.py --iterations 3

    # Continue from iteration 2 with augmented data:
    uv run python scripts/self_improve.py --start-iteration 2 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream output."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def wait_for_job(job_id: str, poll_interval: int = 60) -> bool:
    """Poll HF job status until completion."""
    print(f"\nWaiting for job {job_id}...")
    print(f"  Monitor: https://huggingface.co/jobs/ratnam1510/{job_id}")

    while True:
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    f"""
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
job = api.get_job('{job_id}')
print(job.status)
""",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            status = result.stdout.strip().lower()
            print(f"  Status: {status}")

            match status:
                case "completed" | "success":
                    print("  Job completed successfully!")
                    return True
                case "failed" | "error":
                    print("  Job failed!")
                    return False
                case "cancelled" | "stopped":
                    print("  Job was cancelled/stopped.")
                    return False
                case _:
                    time.sleep(poll_interval)
        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(poll_interval)


def step_train(
    iteration: int,
    flavor: str,
    model_name: str,
    dataset_repo: str,
    output_repo: str,
    augmented_path: Path | None = None,
) -> str | None:
    """Launch training job and return job ID."""
    print(f"\n{'=' * 60}")
    print(f"STEP 1: TRAIN (iteration {iteration})")
    print(f"{'=' * 60}")

    # If we have augmented data from a previous optimization, swap it in
    train_path = Path("ai_pipeline/dataset/train.jsonl")
    if augmented_path and augmented_path.exists():
        backup = train_path.with_suffix(f".iter{iteration - 1}.bak")
        shutil.copy2(train_path, backup)
        shutil.copy2(augmented_path, train_path)
        print(f"  Swapped in augmented dataset: {augmented_path}")
        print(f"  Original backed up to: {backup}")

    # Launch the job
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/launch_finetune.py",
                "--flavor",
                flavor,
                "--model",
                model_name,
                "--timeout",
                "6h",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse job ID from output
        for line in result.stdout.splitlines():
            if "Job ID:" in line:
                job_id = line.split("Job ID:")[-1].strip()
                print(f"  Launched job: {job_id}")
                return job_id
        print(f"  Output: {result.stdout}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Training launch failed: {e.stderr}")
        return None


def step_evaluate(
    iteration: int, model_id: str, judge_model: str, sample_size: int
) -> dict | None:
    """Run evaluation and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"STEP 2: EVALUATE (iteration {iteration})")
    print(f"{'=' * 60}")

    try:
        run_cmd([
            "uv",
            "run",
            "python",
            "scripts/evaluate.py",
            "--model",
            model_id,
            "--judge-model",
            judge_model,
            "--sample-size",
            str(sample_size),
            "--iteration",
            str(iteration),
        ])

        # Load results
        results_path = Path(f"ai_pipeline/eval_results/iter_{iteration}.json")
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            return data.get("metrics", {})
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Evaluation failed: {e}")
        return None


def step_optimize(
    iteration: int, optimizer_model: str, max_improved: int, synthetic_per_weakness: int
) -> Path | None:
    """Run optimization and return path to augmented dataset."""
    print(f"\n{'=' * 60}")
    print(f"STEP 3: OPTIMIZE (iteration {iteration})")
    print(f"{'=' * 60}")

    try:
        run_cmd([
            "uv",
            "run",
            "python",
            "scripts/optimize.py",
            "--iteration",
            str(iteration),
            "--optimizer-model",
            optimizer_model,
            "--max-improved",
            str(max_improved),
            "--synthetic-per-weakness",
            str(synthetic_per_weakness),
        ])

        augmented = Path(f"ai_pipeline/dataset/augmented_iter_{iteration + 1}.jsonl")
        if augmented.exists():
            return augmented
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Optimization failed: {e}")
        return None


def print_iteration_summary(iteration: int, metrics: dict | None) -> None:
    """Print a summary of the iteration."""
    print(f"\n{'=' * 60}")
    print(f"ITERATION {iteration} SUMMARY")
    print(f"{'=' * 60}")
    if metrics:
        print(f"  avg_overall:    {metrics.get('avg_overall', 'N/A')}")
        print(f"  avg_relevance:  {metrics.get('avg_relevance', 'N/A')}")
        print(
            f"  avg_vuln_id:    {metrics.get('avg_vulnerability_identification', 'N/A')}"
        )
        print(f"  avg_fix:        {metrics.get('avg_fix_suggestion', 'N/A')}")
        print(f"  pct_good (>=4): {metrics.get('pct_good', 'N/A')}")
        print(f"  pct_poor (<=2): {metrics.get('pct_poor', 'N/A')}")
    else:
        print("  No metrics available.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-improving fine-tuning orchestrator"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Number of improvement iterations"
    )
    parser.add_argument(
        "--start-iteration", type=int, default=1, help="Starting iteration number"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (assume model is already trained)",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip optimization (just train + eval)",
    )
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Base model for training",
    )
    parser.add_argument(
        "--eval-model",
        default="ratnam1510/mistral-small-secure-scan",
        help="Model ID to evaluate (the fine-tuned output)",
    )
    parser.add_argument(
        "--judge-model",
        default="mistral-large-latest",
        help="Model used as judge during evaluation",
    )
    parser.add_argument(
        "--optimizer-model",
        default="mistral-large-latest",
        help="Model used to generate improved training data",
    )
    parser.add_argument("--flavor", default="a10g-small", help="GPU flavor for HF Jobs")
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Number of test samples per evaluation",
    )
    parser.add_argument(
        "--wait-for-job",
        type=str,
        default=None,
        help="Wait for a specific job ID instead of launching new training",
    )
    args = parser.parse_args()

    # Validate env
    for var in ["HF_TOKEN"]:
        if not os.environ.get(var):
            print(f"Error: {var} environment variable required.")
            sys.exit(1)

    if not args.skip_train and not os.environ.get("MISTRAL_API_KEY"):
        print("Warning: MISTRAL_API_KEY not set. Eval/optimize steps will fail.")

    print("=" * 60)
    print("SELF-IMPROVING FINE-TUNING LOOP")
    print("=" * 60)
    print(
        f"  Iterations: {args.start_iteration} to {args.start_iteration + args.iterations - 1}"
    )
    print(f"  Base model: {args.model}")
    print(f"  Eval model: {args.eval_model}")
    print(f"  Judge: {args.judge_model}")
    print(f"  GPU: {args.flavor}")
    print(f"  Skip train: {args.skip_train}")

    all_metrics: list[dict] = []
    augmented_path: Path | None = None

    for i in range(args.start_iteration, args.start_iteration + args.iterations):
        print(f"\n\n{'#' * 60}")
        print(f"# ITERATION {i}")
        print(f"{'#' * 60}")

        # Step 1: Train
        if not args.skip_train:
            if args.wait_for_job and i == args.start_iteration:
                # Wait for an already-running job
                success = wait_for_job(args.wait_for_job)
                if not success:
                    print("Training job failed. Stopping.")
                    break
            else:
                job_id = step_train(
                    iteration=i,
                    flavor=args.flavor,
                    model_name=args.model,
                    dataset_repo="ratnam1510/security-vuln-dataset",
                    output_repo="ratnam1510/mistral-small-secure-scan",
                    augmented_path=augmented_path,
                )
                if not job_id:
                    print("Failed to launch training. Stopping.")
                    break
                success = wait_for_job(job_id)
                if not success:
                    print("Training job failed. Stopping.")
                    break

        # Step 2: Evaluate
        metrics = step_evaluate(
            iteration=i,
            model_id=args.eval_model,
            judge_model=args.judge_model,
            sample_size=args.eval_samples,
        )
        all_metrics.append(metrics or {})
        print_iteration_summary(i, metrics)

        # Check if we should stop (good enough)
        if metrics and metrics.get("avg_overall", 0) >= 4.5:
            print(f"\n  Model is performing well (avg_overall >= 4.5). Stopping early.")
            break

        # Step 3: Optimize (generate improved data for next round)
        if not args.skip_optimize and i < args.start_iteration + args.iterations - 1:
            augmented_path = step_optimize(
                iteration=i,
                optimizer_model=args.optimizer_model,
                max_improved=20,
                synthetic_per_weakness=5,
            )

    # Final summary
    print(f"\n\n{'=' * 60}")
    print("FINAL SUMMARY — ALL ITERATIONS")
    print(f"{'=' * 60}")
    for i, m in enumerate(all_metrics, start=args.start_iteration):
        overall = m.get("avg_overall", "N/A")
        good = m.get("pct_good", "N/A")
        print(f"  Iteration {i}: avg_overall={overall}, pct_good={good}")

    if len(all_metrics) >= 2:
        first = all_metrics[0].get("avg_overall", 0) or 0
        last = all_metrics[-1].get("avg_overall", 0) or 0
        delta = last - first
        print(
            f"\n  Improvement: {first:.3f} -> {last:.3f} ({'+' if delta >= 0 else ''}{delta:.3f})"
        )


if __name__ == "__main__":
    main()

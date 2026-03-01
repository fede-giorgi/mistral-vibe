# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mistralai",
#     "wandb",
# ]
# ///
"""
Evaluate fine-tuned security model against test set using LLM-as-judge.

Runs the fine-tuned model on each test example, then uses a strong judge model
(mistral-large) to score the response on multiple dimensions. Results are logged
to W&B for tracking across improvement iterations.

Usage:
    export MISTRAL_API_KEY="..."
    export WANDB_API_KEY="..."
    uv run python scripts/evaluate.py \
        --model "ratnam1510/mistral-small-secure-scan" \
        --iteration 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import wandb
from mistralai import Mistral

# ---------------------------------------------------------------------------
# Scoring rubric used by the judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are an expert security auditor evaluating AI-generated vulnerability analyses.
Score the candidate response on EACH dimension below (1-5 scale).

Dimensions:
1. **vulnerability_identification** — Did it identify a real vulnerability class
   (e.g. CWE category, OWASP type)? (1=wrong/none, 5=precise CWE + description)
2. **severity_accuracy** — Is the severity assessment reasonable?
   (1=wildly off, 3=roughly correct, 5=matches CVSS-aligned ground truth)
3. **explanation_quality** — Is the explanation clear, specific, and actionable?
   (1=vague/generic, 5=cites lines, explains root cause and impact)
4. **fix_suggestion** — Does it suggest a correct remediation?
   (1=no fix or wrong fix, 5=complete, correct, production-ready fix)
5. **relevance** — Does the response actually address the code shown?
   (1=completely unrelated, 5=directly analyses the given snippet)

Return ONLY valid JSON:
{
  "vulnerability_identification": <1-5>,
  "severity_accuracy": <1-5>,
  "explanation_quality": <1-5>,
  "fix_suggestion": <1-5>,
  "relevance": <1-5>,
  "overall": <1-5>,
  "reasoning": "<brief justification>"
}
"""


@dataclass
class EvalResult:
    """Single evaluation result for one test example."""

    index: int
    code_snippet: str
    ground_truth: str
    model_response: str
    scores: dict[str, float] = field(default_factory=dict)
    judge_reasoning: str = ""
    error: str = ""
    latency_ms: float = 0.0


def load_test_data(path: str) -> list[dict]:
    """Load test examples from JSONL."""
    data: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            # Extract code from user message and ground truth from assistant
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(
                m["content"] for m in messages if m["role"] == "assistant"
            )
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"),
                "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation.",
            )
            data.append({
                "system": system_msg,
                "user": user_msg,
                "ground_truth": assistant_msg,
            })
    return data


def query_model(
    client: Mistral,
    model_id: str,
    system: str,
    user: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> tuple[str, float]:
    """Query a model and return (response_text, latency_ms)."""
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.chat.complete(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2048,
            )
            latency = (time.monotonic() - t0) * 1000
            text = resp.choices[0].message.content if resp.choices else ""
            return text or "", latency
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[ERROR] {e}", 0.0
            time.sleep(retry_delay * (attempt + 1))
    return "[ERROR] max retries exceeded", 0.0


def judge_response(
    client: Mistral,
    judge_model: str,
    code: str,
    ground_truth: str,
    candidate: str,
    *,
    max_retries: int = 3,
) -> dict:
    """Use a strong model to score the candidate response."""
    prompt = f"""\
## Code Under Review
```
{code[:3000]}
```

## Ground Truth Analysis
{ground_truth[:2000]}

## Candidate Response to Evaluate
{candidate[:2000]}

Score the candidate response on all dimensions.
"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.complete(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            text = resp.choices[0].message.content if resp.choices else "{}"
            return json.loads(text or "{}")
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "overall": 1}
            time.sleep(3 * (attempt + 1))
    return {"error": "max retries", "overall": 1}


def run_evaluation(
    client: Mistral,
    model_id: str,
    judge_model: str,
    test_data: list[dict],
    sample_size: int | None = None,
) -> list[EvalResult]:
    """Evaluate model on test data with LLM-as-judge scoring."""
    subset = test_data[:sample_size] if sample_size else test_data
    results: list[EvalResult] = []

    for i, example in enumerate(subset):
        print(f"  [{i + 1}/{len(subset)}] Evaluating...")

        # Get model response
        response, latency = query_model(
            client, model_id, example["system"], example["user"]
        )

        # Judge the response
        scores = judge_response(
            client,
            judge_model,
            code=example["user"],
            ground_truth=example["ground_truth"],
            candidate=response,
        )

        result = EvalResult(
            index=i,
            code_snippet=example["user"][:500],
            ground_truth=example["ground_truth"][:500],
            model_response=response[:500],
            scores={k: v for k, v in scores.items() if k not in ("reasoning", "error")},
            judge_reasoning=scores.get("reasoning", ""),
            error=scores.get("error", ""),
            latency_ms=latency,
        )
        results.append(result)

        # Rate limit: be gentle with the API
        time.sleep(1)

    return results


def compute_metrics(results: list[EvalResult]) -> dict[str, float]:
    """Aggregate scores across all results."""
    if not results:
        return {}

    valid = [r for r in results if not r.error]
    if not valid:
        return {"num_evaluated": len(results), "num_errors": len(results)}

    dimensions = [
        "vulnerability_identification",
        "severity_accuracy",
        "explanation_quality",
        "fix_suggestion",
        "relevance",
        "overall",
    ]

    metrics: dict[str, float] = {
        "num_evaluated": len(results),
        "num_valid": len(valid),
        "num_errors": len(results) - len(valid),
    }

    for dim in dimensions:
        values = [
            r.scores[dim] for r in valid if isinstance(r.scores.get(dim), int | float)
        ]
        if values:
            metrics[f"avg_{dim}"] = round(sum(values) / len(values), 3)
            metrics[f"min_{dim}"] = min(values)
            metrics[f"max_{dim}"] = max(values)

    # Score distribution for overall
    overall_vals = [
        r.scores.get("overall", 0) for r in valid if r.scores.get("overall")
    ]
    if overall_vals:
        metrics["pct_good"] = round(
            sum(1 for v in overall_vals if v >= 4) / len(overall_vals), 3
        )
        metrics["pct_poor"] = round(
            sum(1 for v in overall_vals if v <= 2) / len(overall_vals), 3
        )

    return metrics


def log_to_wandb(
    metrics: dict[str, float], results: list[EvalResult], iteration: int, model_id: str
) -> None:
    """Log evaluation results to W&B."""
    # Log scalar metrics
    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=iteration)

    # Log results table
    columns = [
        "index",
        "code_snippet",
        "ground_truth",
        "model_response",
        "overall",
        "vuln_id",
        "severity",
        "explanation",
        "fix",
        "relevance",
        "reasoning",
        "latency_ms",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        table.add_data(
            r.index,
            r.code_snippet[:200],
            r.ground_truth[:200],
            r.model_response[:200],
            r.scores.get("overall", 0),
            r.scores.get("vulnerability_identification", 0),
            r.scores.get("severity_accuracy", 0),
            r.scores.get("explanation_quality", 0),
            r.scores.get("fix_suggestion", 0),
            r.scores.get("relevance", 0),
            r.judge_reasoning[:200],
            r.latency_ms,
        )
    wandb.log({f"eval/results_iter_{iteration}": table}, step=iteration)

    # Save detailed results as artifact
    artifact = wandb.Artifact(
        f"eval-results-iter-{iteration}",
        type="evaluation",
        metadata={"model": model_id, "iteration": iteration, **metrics},
    )
    results_path = Path(f"/tmp/eval_results_iter_{iteration}.json")
    results_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    artifact.add_file(str(results_path))
    wandb.log_artifact(artifact)


def save_results_local(
    results: list[EvalResult], metrics: dict, iteration: int
) -> Path:
    """Save results locally for the optimizer to consume."""
    out_dir = Path("ai_pipeline/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"iter_{iteration}.json"
    out_path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "metrics": metrics,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Results saved to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned security model")
    parser.add_argument(
        "--model",
        default="ratnam1510/mistral-small-secure-scan",
        help="Fine-tuned model ID (HF repo or Mistral model ID)",
    )
    parser.add_argument(
        "--judge-model",
        default="mistral-large-latest",
        help="Strong model used as judge",
    )
    parser.add_argument(
        "--test-data",
        default="ai_pipeline/dataset/test.jsonl",
        help="Path to test JSONL",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of test examples to evaluate (None=all)",
    )
    parser.add_argument(
        "--iteration", type=int, default=1, help="Current improvement iteration number"
    )
    parser.add_argument(
        "--wandb-project", default="mistral-vibe-security", help="W&B project name"
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable required.")
        raise SystemExit(1)

    wandb_key = os.environ.get("WANDB_API_KEY", "")

    # Init W&B
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project=args.wandb_project,
            name=f"eval-iter-{args.iteration}",
            config={
                "model": args.model,
                "judge_model": args.judge_model,
                "iteration": args.iteration,
                "sample_size": args.sample_size,
            },
            tags=["evaluation", f"iter-{args.iteration}"],
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"

    client = Mistral(api_key=api_key)

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    print(f"  Loaded {len(test_data)} test examples")

    # Run evaluation
    print(f"Evaluating model: {args.model}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Sample size: {args.sample_size}")
    results = run_evaluation(
        client, args.model, args.judge_model, test_data, args.sample_size
    )

    # Compute metrics
    metrics = compute_metrics(results)
    print("\n--- Evaluation Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Save results
    results_path = save_results_local(results, metrics, args.iteration)

    # Log to W&B
    if wandb_key:
        log_to_wandb(metrics, results, args.iteration, args.model)
        wandb.finish()
        print(f"\n  W&B run: https://wandb.ai (project: {args.wandb_project})")

    print(f"\nEvaluation complete. Results at: {results_path}")
    print(f"  avg_overall: {metrics.get('avg_overall', 'N/A')}")
    print(f"  pct_good (>=4): {metrics.get('pct_good', 'N/A')}")
    print(f"  pct_poor (<=2): {metrics.get('pct_poor', 'N/A')}")


if __name__ == "__main__":
    main()

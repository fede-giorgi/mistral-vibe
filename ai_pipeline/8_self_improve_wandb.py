#!/usr/bin/env python3
"""Step 8 — W&B MCP-driven self-improvement loop.

This script is designed to be called by the Cursor agent as part of
a W&B MCP self-improvement workflow. It supports subcommands that the
agent invokes between W&B MCP queries to close the evaluate-analyze-improve loop.

Subcommands:
  eval      Run evaluation on the fine-tuned model, log results to W&B
  analyze   Load last eval results, print failure breakdown (agent reads this)
  optimize  Generate improved training data for weak areas
  report    Create a W&B report comparing iterations

The agent workflow (driven by W&B MCP):
  1. Agent calls `eval` → results logged to W&B
  2. Agent uses W&B MCP `query_wandb_tool` to inspect metrics
  3. Agent uses W&B MCP `query_weave_traces_tool` to find failing traces
  4. Agent calls `analyze` to get structured failure breakdown
  5. Agent decides what to improve (prompts, training data, code)
  6. Agent calls `optimize` to generate better training data
  7. Agent re-runs `eval` and compares via W&B MCP
  8. Agent calls `report` to create a W&B report

Env vars needed:
  WANDB_API_KEY          — W&B logging (required)
  GEMINI_API_KEY         — Gemini judge (required for default judge)
  HUGGINGFACE_HUB_TOKEN  — HF Inference API for fine-tuned model
  MISTRAL_API_KEY        — Mistral API (if using Mistral model or judge)

Usage:
  # Load env vars
  set -a && source ai_pipeline/.env && set +a

  # Run evaluation: HF model + Gemini judge (default)
  uv run python ai_pipeline/8_self_improve_wandb.py eval \
    --model username/repo-name --iteration 1

  # Or with Mistral model + Gemini judge
  uv run python ai_pipeline/8_self_improve_wandb.py eval \
    --model ft:your-model-id --judge-model gemini-2.0-flash --iteration 1

  # Analyze failures from last eval
  uv run python ai_pipeline/8_self_improve_wandb.py analyze --iteration 1

  # Generate improved training data
  uv run python ai_pipeline/8_self_improve_wandb.py optimize --iteration 1

  # Create W&B report comparing iterations 1 and 2
  uv run python ai_pipeline/8_self_improve_wandb.py report --iterations 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

WANDB_PROJECT = "mistral-vibe-security"
EVAL_RESULTS_DIR = Path("ai_pipeline/eval_results")

JUDGE_SYSTEM = """\
You are an expert security auditor evaluating AI-generated vulnerability analyses.
Score the candidate response on EACH dimension below (1-5 scale).

Dimensions:
1. **vulnerability_identification** — Correct CWE / vulnerability class?
   (1=wrong/none, 5=precise CWE + description)
2. **severity_accuracy** — Reasonable severity assessment?
   (1=wildly off, 3=roughly correct, 5=matches CVSS-aligned ground truth)
3. **explanation_quality** — Clear, specific, actionable?
   (1=vague/generic, 5=cites lines, explains root cause and impact)
4. **fix_suggestion** — Correct remediation?
   (1=no fix or wrong fix, 5=complete, correct, production-ready fix)
5. **relevance** — Addresses the code shown?
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
    index: int
    code_snippet: str
    ground_truth: str
    model_response: str
    scores: dict[str, float] = field(default_factory=dict)
    judge_reasoning: str = ""
    error: str = ""
    latency_ms: float = 0.0


def _load_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)


def _load_test_data(path: str) -> list[dict]:
    data: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            user_msg = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"),
                "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation.",
            )
            data.append({"system": system_msg, "user": user_msg, "ground_truth": assistant_msg})
    return data


def _query_model_mistral(client, model_id: str, system: str, user: str) -> tuple[str, float]:
    """Call model via Mistral API."""
    for attempt in range(3):
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
            if attempt == 2:
                return f"[ERROR] {e}", 0.0
            time.sleep(5 * (attempt + 1))
    return "[ERROR] max retries exceeded", 0.0


def _query_model_hf(model_id: str, system: str, user: str) -> tuple[str, float]:
    """Call model via HF dedicated Inference Endpoint (OpenAI-compatible) or serverless API."""
    import httpx

    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "")
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip().rstrip("/")

    if endpoint_url:
        url = f"{endpoint_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 2048,
            "temperature": 0.2,
        }
    else:
        url = f"https://router.huggingface.co/hf/models/{model_id}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": f"{system}\n\n{user}", "parameters": {"max_new_tokens": 2048, "return_full_text": False, "temperature": 0.2}}

    for attempt in range(5):
        try:
            t0 = time.monotonic()
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload, headers=headers)
            latency = (time.monotonic() - t0) * 1000
            response.raise_for_status()
            data = response.json()
            if endpoint_url:
                return data["choices"][0]["message"]["content"].strip(), latency
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip(), latency
            return str(data), latency
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503 and attempt < 4:
                logger.warning("HF model loading, retrying in %ds...", 10 * (attempt + 1))
                time.sleep(10 * (attempt + 1))
                continue
            return f"[ERROR] {e}", 0.0
        except Exception as e:
            if attempt == 4:
                return f"[ERROR] {e}", 0.0
            time.sleep(5 * (attempt + 1))
    return "[ERROR] max retries exceeded", 0.0


def _judge_gemini(gemini_client, gemini_model: str, code: str, ground_truth: str, candidate: str) -> dict:
    """Score response using Gemini as judge."""
    prompt = (
        f"## Code Under Review\n```\n{code[:4000]}\n```\n\n"
        f"## Ground Truth Analysis\n{ground_truth[:3000]}\n\n"
        f"## Candidate Response to Evaluate\n{candidate[:3000]}\n\n"
        "Score the candidate response on all dimensions. Return ONLY valid JSON."
    )
    for attempt in range(6):
        try:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=[{"role": "user", "parts": [{"text": JUDGE_SYSTEM + "\n\n" + prompt}]}],
            )
            text = (response.text or "{}").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if attempt == 5:
                return {"error": err_str, "overall": 1}
            wait = 60 * (attempt + 1) if is_rate_limit else 4 * (attempt + 1)
            logger.warning("Judge attempt %d failed: %s, retrying in %ds", attempt + 1, err_str[:80], wait)
            time.sleep(wait)
    return {"error": "max retries", "overall": 1}


def _judge_mistral(client, judge_model: str, code: str, ground_truth: str, candidate: str) -> dict:
    """Score response using Mistral as judge."""
    prompt = (
        f"## Code Under Review\n```\n{code[:3000]}\n```\n\n"
        f"## Ground Truth Analysis\n{ground_truth[:2000]}\n\n"
        f"## Candidate Response to Evaluate\n{candidate[:2000]}\n\n"
        "Score the candidate response on all dimensions."
    )
    for attempt in range(3):
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
        except Exception as e:
            if attempt == 2:
                return {"error": str(e), "overall": 1}
            time.sleep(3 * (attempt + 1))
    return {"error": "max retries", "overall": 1}


def _compute_metrics(results: list[EvalResult]) -> dict[str, float]:
    valid = [r for r in results if not r.error]
    if not valid:
        return {"num_evaluated": len(results), "num_errors": len(results)}

    dimensions = [
        "vulnerability_identification", "severity_accuracy",
        "explanation_quality", "fix_suggestion", "relevance", "overall",
    ]
    metrics: dict[str, float] = {
        "num_evaluated": len(results),
        "num_valid": len(valid),
        "num_errors": len(results) - len(valid),
    }
    for dim in dimensions:
        values = [r.scores[dim] for r in valid if isinstance(r.scores.get(dim), int | float)]
        if values:
            metrics[f"avg_{dim}"] = round(sum(values) / len(values), 3)
            metrics[f"min_{dim}"] = min(values)
            metrics[f"max_{dim}"] = max(values)

    overall_vals = [r.scores.get("overall", 0) for r in valid if r.scores.get("overall")]
    if overall_vals:
        metrics["pct_good"] = round(sum(1 for v in overall_vals if v >= 4) / len(overall_vals), 3)
        metrics["pct_poor"] = round(sum(1 for v in overall_vals if v <= 2) / len(overall_vals), 3)
    return metrics


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation and log results to W&B."""
    import wandb

    # Determine model backend: HF Inference API (username/repo) or Mistral API
    use_hf_model = "/" in args.model and not args.model.startswith("ft:")
    # Determine judge backend: Gemini (gemini-*) or Mistral
    use_gemini_judge = args.judge_model.startswith("gemini")

    mistral_client = None
    if not use_hf_model or not use_gemini_judge:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("Error: MISTRAL_API_KEY required (for Mistral model or judge).", file=sys.stderr)
            sys.exit(1)
        from mistralai import Mistral
        mistral_client = Mistral(api_key=api_key)

    gemini_client = None
    if use_gemini_judge:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            print("Error: GEMINI_API_KEY required for Gemini judge. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
            sys.exit(1)
        from google import genai
        gemini_client = genai.Client(api_key=gemini_key)

    if use_hf_model:
        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "")
        if not token:
            print("Error: HUGGINGFACE_HUB_TOKEN required for HF model.", file=sys.stderr)
            sys.exit(1)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"self-improve-eval-iter-{args.iteration}",
        config={
            "model": args.model,
            "model_backend": "hf" if use_hf_model else "mistral",
            "judge_model": args.judge_model,
            "judge_backend": "gemini" if use_gemini_judge else "mistral",
            "iteration": args.iteration,
            "sample_size": args.sample_size,
            "step": "eval",
        },
        tags=["self-improve", "evaluation", f"iter-{args.iteration}"],
    )

    test_data = _load_test_data(args.test_data)
    subset = test_data[: args.sample_size] if args.sample_size else test_data
    print(f"Evaluating {len(subset)} samples")
    print(f"  Model: {args.model} ({'HF Inference API' if use_hf_model else 'Mistral API'})")
    print(f"  Judge: {args.judge_model} ({'Gemini' if use_gemini_judge else 'Mistral'})")

    results: list[EvalResult] = []
    for i, example in enumerate(subset):
        print(f"  [{i + 1}/{len(subset)}] Evaluating...")

        # Query the fine-tuned model
        if use_hf_model:
            response, latency = _query_model_hf(args.model, example["system"], example["user"])
        else:
            response, latency = _query_model_mistral(mistral_client, args.model, example["system"], example["user"])

        # Judge the response
        if use_gemini_judge:
            scores = _judge_gemini(gemini_client, args.judge_model, example["user"], example["ground_truth"], response)
        else:
            scores = _judge_mistral(mistral_client, args.judge_model, example["user"], example["ground_truth"], response)

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

        # Log per-sample metrics for Weave trace queryability
        wandb.log({
            "sample/index": i,
            "sample/overall": scores.get("overall", 0),
            "sample/vuln_id": scores.get("vulnerability_identification", 0),
            "sample/severity": scores.get("severity_accuracy", 0),
            "sample/explanation": scores.get("explanation_quality", 0),
            "sample/fix": scores.get("fix_suggestion", 0),
            "sample/relevance": scores.get("relevance", 0),
            "sample/latency_ms": latency,
        })
        time.sleep(1)

    metrics = _compute_metrics(results)

    # Log aggregate metrics
    wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    # Log results table
    columns = [
        "index", "code_snippet", "ground_truth", "model_response",
        "overall", "vuln_id", "severity", "explanation", "fix", "relevance",
        "reasoning", "latency_ms",
    ]
    table = wandb.Table(columns=columns)
    for r in results:
        table.add_data(
            r.index, r.code_snippet[:200], r.ground_truth[:200], r.model_response[:200],
            r.scores.get("overall", 0), r.scores.get("vulnerability_identification", 0),
            r.scores.get("severity_accuracy", 0), r.scores.get("explanation_quality", 0),
            r.scores.get("fix_suggestion", 0), r.scores.get("relevance", 0),
            r.judge_reasoning[:200], r.latency_ms,
        )
    wandb.log({f"eval/results_iter_{args.iteration}": table})

    # Save as artifact
    artifact = wandb.Artifact(
        f"eval-results-iter-{args.iteration}",
        type="evaluation",
        metadata={"model": args.model, "iteration": args.iteration, **metrics},
    )
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"iter_{args.iteration}.json"
    out_path.write_text(
        json.dumps({"iteration": args.iteration, "metrics": metrics, "results": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    artifact.add_file(str(out_path))
    wandb.log_artifact(artifact)

    wandb.finish()

    print("\n--- Evaluation Metrics ---")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")
    print(f"\nResults saved to {out_path}")
    print(f"W&B run: https://wandb.ai (project: {WANDB_PROJECT})")
    print(
        "\nNext: use W&B MCP query_wandb_tool to inspect metrics, "
        "then run `analyze` to see failure breakdown."
    )


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
def cmd_analyze(args: argparse.Namespace) -> None:
    """Load eval results and print structured failure analysis for the agent."""
    path = EVAL_RESULTS_DIR / f"iter_{args.iteration}.json"
    if not path.exists():
        print(f"Error: no eval results at {path}. Run `eval` first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    results = data["results"]

    # Categorize failures
    patterns: dict[str, list[dict]] = {
        "low_relevance": [],
        "poor_explanation": [],
        "wrong_vulnerability": [],
        "missing_fix": [],
        "low_overall": [],
    }
    for r in results:
        scores = r.get("scores", {})
        if not scores:
            continue
        if scores.get("relevance", 5) <= 2:
            patterns["low_relevance"].append(r)
        if scores.get("explanation_quality", 5) <= 2:
            patterns["poor_explanation"].append(r)
        if scores.get("vulnerability_identification", 5) <= 2:
            patterns["wrong_vulnerability"].append(r)
        if scores.get("fix_suggestion", 5) <= 2:
            patterns["missing_fix"].append(r)
        if scores.get("overall", 5) <= 2:
            patterns["low_overall"].append(r)

    print(f"=== Failure Analysis — Iteration {args.iteration} ===")
    print(f"Samples evaluated: {metrics.get('num_evaluated', 0)}")
    print(f"avg_overall: {metrics.get('avg_overall', 'N/A')}")
    print(f"pct_good (>=4): {metrics.get('pct_good', 'N/A')}")
    print(f"pct_poor (<=2): {metrics.get('pct_poor', 'N/A')}")
    print()
    print("Failure breakdown:")
    for name, cases in patterns.items():
        print(f"  {name}: {len(cases)} cases")

    # Identify top weaknesses
    weakness_areas = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
    top_weaknesses = [name for name, cases in weakness_areas if cases]
    if top_weaknesses:
        print(f"\nTop weakness areas: {', '.join(top_weaknesses)}")
        print("\nRecommended actions:")
        for name in top_weaknesses[:3]:
            match name:
                case "low_relevance":
                    print(f"  - {name}: Responses don't match the code. Fix training data alignment.")
                case "wrong_vulnerability":
                    print(f"  - {name}: Wrong CWE identified. Add diverse CWE examples.")
                case "missing_fix":
                    print(f"  - {name}: No remediation suggested. Add fix-focused examples.")
                case "poor_explanation":
                    print(f"  - {name}: Vague explanations. Add detailed analysis examples.")
                case "low_overall":
                    print(f"  - {name}: General quality issue. Improve all dimensions.")
    else:
        print("\nNo major failure patterns detected. Model is performing well.")

    # Output JSON for agent consumption
    analysis_path = EVAL_RESULTS_DIR / f"analysis_iter_{args.iteration}.json"
    analysis = {
        "iteration": args.iteration,
        "metrics": metrics,
        "failure_patterns": {k: len(v) for k, v in patterns.items()},
        "top_weaknesses": top_weaknesses,
        "total_failures": sum(len(v) for v in patterns.values()),
    }
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nAnalysis saved to {analysis_path}")
    print(
        "\nNext: use W&B MCP query_weave_traces_tool to inspect failing traces, "
        "then run `optimize` to generate improved data."
    )


# ---------------------------------------------------------------------------
# Subcommand: optimize
# ---------------------------------------------------------------------------
def cmd_optimize(args: argparse.Namespace) -> None:
    """Generate improved training data based on failure analysis."""
    import wandb
    from mistralai import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY required.", file=sys.stderr)
        sys.exit(1)

    path = EVAL_RESULTS_DIR / f"iter_{args.iteration}.json"
    if not path.exists():
        print(f"Error: no eval results at {path}. Run `eval` first.", file=sys.stderr)
        sys.exit(1)

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project=WANDB_PROJECT,
            name=f"self-improve-optimize-iter-{args.iteration}",
            config={"iteration": args.iteration, "step": "optimize", "optimizer_model": args.optimizer_model},
            tags=["self-improve", "optimization", f"iter-{args.iteration}"],
        )

    client = Mistral(api_key=api_key)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    metrics = data["metrics"]

    # Find failures
    failures = sorted(
        [r for r in results if r.get("scores", {}).get("overall", 5) <= 3],
        key=lambda r: r.get("scores", {}).get("overall", 5),
    )
    print(f"Found {len(failures)} low-scoring examples to improve")

    improved: list[dict] = []
    for r in failures[: args.max_improved]:
        scores = r.get("scores", {})
        if scores.get("relevance", 5) <= 2:
            reason = "Response is not relevant to the code shown"
        elif scores.get("vulnerability_identification", 5) <= 2:
            reason = "Wrong vulnerability type identified"
        elif scores.get("fix_suggestion", 5) <= 2:
            reason = "No actionable fix suggested"
        else:
            reason = "Low overall quality"

        print(f"  Improving example {r['index']} (overall={scores.get('overall', '?')}, reason={reason})...")
        prompt = (
            f"The following code was analyzed by a security model, but the analysis was poor.\n\n"
            f"## Failure reason: {reason}\n\n"
            f"## Code:\n```\n{r.get('code_snippet', '')[:3000]}\n```\n\n"
            f"## Original (poor) response:\n{r.get('model_response', '')[:1500]}\n\n"
            "Generate an IMPROVED, ACCURATE security analysis. Return ONLY the assistant message content."
        )
        try:
            resp = client.chat.complete(
                model=args.optimizer_model,
                messages=[
                    {"role": "system", "content": "You are an expert security training data curator."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )
            text = resp.choices[0].message.content if resp.choices else None
            if text:
                improved.append({
                    "messages": [
                        {"role": "system", "content": "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation."},
                        {"role": "user", "content": r.get("code_snippet", "")},
                        {"role": "assistant", "content": text},
                    ]
                })
        except Exception as e:
            logger.warning("Error improving example %d: %s", r["index"], e)
        time.sleep(1)

    print(f"Generated {len(improved)} improved examples")

    # Write augmented dataset
    output_dir = Path("ai_pipeline/dataset")
    train_path = output_dir / "train.jsonl"
    existing: list[dict] = []
    if train_path.exists():
        with train_path.open(encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]

    augmented = existing + improved
    augmented_path = output_dir / f"augmented_iter_{args.iteration + 1}.jsonl"
    with augmented_path.open("w", encoding="utf-8") as f:
        for ex in augmented:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Augmented dataset: {augmented_path}")
    print(f"  Original: {len(existing)}, + Improved: {len(improved)}, = Total: {len(augmented)}")

    if wandb_key:
        wandb.log({
            "optimize/improved_count": len(improved),
            "optimize/total_failures": len(failures),
            "optimize/augmented_total": len(augmented),
        })
        artifact = wandb.Artifact(
            f"augmented-dataset-iter-{args.iteration + 1}",
            type="dataset",
            metadata={"iteration": args.iteration + 1, "improved": len(improved), "total": len(augmented)},
        )
        artifact.add_file(str(augmented_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    print(
        "\nNext: retrain with augmented data, then re-run `eval --iteration "
        f"{args.iteration + 1}` and compare via W&B MCP."
    )


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> None:
    """Create a W&B report comparing iterations."""
    import wandb

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print("Error: WANDB_API_KEY required for W&B report.", file=sys.stderr)
        sys.exit(1)

    wandb.login(key=wandb_key)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"self-improve-report-iters-{'-'.join(str(i) for i in args.iterations)}",
        config={"iterations": args.iterations, "step": "report"},
        tags=["self-improve", "report"],
    )

    # Load metrics for each iteration
    all_metrics: dict[int, dict] = {}
    for it in args.iterations:
        path = EVAL_RESULTS_DIR / f"iter_{it}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            all_metrics[it] = data["metrics"]
        else:
            print(f"Warning: no results for iteration {it}")

    if not all_metrics:
        print("Error: no iteration results found.", file=sys.stderr)
        sys.exit(1)

    # Log comparison table
    columns = ["iteration", "avg_overall", "avg_vuln_id", "avg_severity", "avg_explanation", "avg_fix", "avg_relevance", "pct_good", "pct_poor"]
    table = wandb.Table(columns=columns)
    for it, m in sorted(all_metrics.items()):
        table.add_data(
            it,
            m.get("avg_overall", 0), m.get("avg_vulnerability_identification", 0),
            m.get("avg_severity_accuracy", 0), m.get("avg_explanation_quality", 0),
            m.get("avg_fix_suggestion", 0), m.get("avg_relevance", 0),
            m.get("pct_good", 0), m.get("pct_poor", 0),
        )
    wandb.log({"report/iteration_comparison": table})

    # Log delta if we have at least 2 iterations
    iters = sorted(all_metrics.keys())
    if len(iters) >= 2:
        first, last = all_metrics[iters[0]], all_metrics[iters[-1]]
        delta = {
            f"delta/{k}": last.get(k, 0) - first.get(k, 0)
            for k in ["avg_overall", "avg_vulnerability_identification", "avg_fix_suggestion", "pct_good"]
            if k in last
        }
        wandb.log(delta)
        print(f"\nImprovement from iter {iters[0]} to {iters[-1]}:")
        for k, v in delta.items():
            print(f"  {k}: {'+' if v >= 0 else ''}{v:.3f}")

    wandb.finish()
    print(f"\nReport logged to W&B project: {WANDB_PROJECT}")
    print("Use W&B MCP create_wandb_report_tool to generate a shareable report.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Step 8: W&B MCP-driven self-improvement loop",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation, log to W&B")
    p_eval.add_argument("--iteration", type=int, default=1)
    p_eval.add_argument("--model", default="ratnam1510/mistral-small-secure-scan")
    p_eval.add_argument("--judge-model", default="gemini-2.0-flash", help="Judge model: gemini-* for Gemini, anything else for Mistral")
    p_eval.add_argument("--test-data", default="ai_pipeline/dataset/test.jsonl")
    p_eval.add_argument("--sample-size", type=int, default=20)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze failures from last eval")
    p_analyze.add_argument("--iteration", type=int, default=1)

    # optimize
    p_optimize = subparsers.add_parser("optimize", help="Generate improved training data")
    p_optimize.add_argument("--iteration", type=int, default=1)
    p_optimize.add_argument("--optimizer-model", default="mistral-large-latest")
    p_optimize.add_argument("--max-improved", type=int, default=20)

    # report
    p_report = subparsers.add_parser("report", help="Create W&B report comparing iterations")
    p_report.add_argument("--iterations", type=int, nargs="+", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    match args.command:
        case "eval":
            cmd_eval(args)
        case "analyze":
            cmd_analyze(args)
        case "optimize":
            cmd_optimize(args)
        case "report":
            cmd_report(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
# ]
# ///
"""
Robust LLM-as-judge scoring via OpenRouter. Scores both base and fine-tuned
model inference results, saving progress after every sample so interrupted
runs can resume.

Usage:
    OPENROUTER_API_KEY="sk-or-..." uv run python scripts/judge_openrouter.py

Outputs:
    ai_pipeline/eval_results/judge_base.json       — per-sample scores (base)
    ai_pipeline/eval_results/judge_finetuned.json   — per-sample scores (fine-tuned)
    ai_pipeline/eval_results/comparison_report.md   — side-by-side comparison
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_MODEL = "z-ai/glm-4.5-air:free"
DELAY_BETWEEN_REQUESTS = 3  # seconds
REQUEST_TIMEOUT = 90  # seconds per API call
MAX_RETRIES = 4

OUT_DIR = Path("ai_pipeline/eval_results")
BASE_INPUT = OUT_DIR / "inference_results_base.json"
FT_INPUT = OUT_DIR / "inference_results_finetuned.json"
BASE_PROGRESS = OUT_DIR / "judge_base_progress.json"
FT_PROGRESS = OUT_DIR / "judge_finetuned_progress.json"
BASE_FINAL = OUT_DIR / "judge_base.json"
FT_FINAL = OUT_DIR / "judge_finetuned.json"
COMPARISON_REPORT = OUT_DIR / "comparison_report.md"

JUDGE_SYSTEM = """\
You are an expert security auditor evaluating AI-generated vulnerability analyses.
Score the candidate response on EACH dimension below (1-5 scale).

Dimensions:
1. vulnerability_identification — Did it identify a real vulnerability class? (1=wrong/none, 5=precise CWE)
2. severity_accuracy — Is the severity assessment reasonable? (1=wildly off, 5=matches ground truth)
3. explanation_quality — Is the explanation clear and actionable? (1=vague, 5=cites lines, root cause)
4. fix_suggestion — Does it suggest correct remediation? (1=no/wrong fix, 5=production-ready)
5. relevance — Does the response address the code shown? (1=unrelated, 5=directly analyses snippet)

Return ONLY valid JSON (no markdown fences, no extra text):
{"vulnerability_identification": <1-5>, "severity_accuracy": <1-5>, "explanation_quality": <1-5>, "fix_suggestion": <1-5>, "relevance": <1-5>, "overall": <1-5>, "reasoning": "<brief justification>"}"""


def call_judge(client: OpenAI, code: str, ground_truth: str, candidate: str) -> dict:
    """Score one sample. Returns dict with scores or error."""
    prompt = (
        f"## Code Under Review\n```\n{code[:3000]}\n```\n\n"
        f"## Ground Truth Analysis\n{ground_truth[:2000]}\n\n"
        f"## Candidate Response to Evaluate\n{candidate[:2000]}\n\n"
        "Score the candidate. Return ONLY valid JSON."
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                timeout=REQUEST_TIMEOUT,
            )
            text = resp.choices[0].message.content or "{}"
            text = text.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            parsed = json.loads(text)
            # Validate we got numeric scores
            if not isinstance(parsed.get("overall"), int | float):
                raise ValueError(f"Missing 'overall' score in: {text[:100]}")
            return parsed
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower() or "quota" in err.lower()
            if attempt == MAX_RETRIES - 1:
                print(
                    f"    FAILED after {MAX_RETRIES} attempts: {err[:150]}", flush=True
                )
                return {"error": err[:300], "overall": 0}
            wait = 60 * (attempt + 1) if is_rate else 10 * (attempt + 1)
            tag = "Rate limited" if is_rate else "Error"
            print(
                f"    {tag}, retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {err[:100]}",
                flush=True,
            )
            time.sleep(wait)
    return {"error": "max retries exhausted", "overall": 0}


def load_progress(path: Path) -> list[dict]:
    """Load previously scored samples."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_progress(path: Path, results: list[dict]) -> None:
    """Save scored samples to disk."""
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def score_file(
    client: OpenAI, input_path: Path, progress_path: Path, label: str
) -> list[dict]:
    """Score all samples in input_path, resuming from progress_path."""
    with input_path.open(encoding="utf-8") as f:
        samples = json.load(f)

    results = load_progress(progress_path)
    done = len(results)

    if done >= len(samples):
        print(
            f"\n[{label}] All {len(samples)} samples already scored. Skipping.",
            flush=True,
        )
        return results

    print(f"\n{'=' * 60}", flush=True)
    print(
        f"[{label}] Scoring samples {done + 1}..{len(samples)} ({len(samples) - done} remaining)",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)

    for i in range(done, len(samples)):
        sample = samples[i]
        print(f"\n[{label}] [{i + 1}/{len(samples)}] Judging...", flush=True)
        t0 = time.time()

        scores = call_judge(
            client,
            code=sample["code_snippet"],
            ground_truth=sample["ground_truth"],
            candidate=sample["model_response"],
        )

        elapsed = time.time() - t0
        result = {
            "index": sample.get("index", i),
            "scores": {
                k: v for k, v in scores.items() if k not in ("reasoning", "error")
            },
            "reasoning": scores.get("reasoning", ""),
            "error": scores.get("error", ""),
            "judge_time_s": round(elapsed, 1),
            "code_snippet": sample["code_snippet"][:200],
            "model_response_preview": sample["model_response"][:200],
            "inference_latency_ms": sample.get("latency_ms", 0),
        }
        results.append(result)

        overall = scores.get("overall", "?")
        err_tag = " [ERROR]" if result["error"] else ""
        print(f"  -> {overall}/5 in {elapsed:.1f}s{err_tag}", flush=True)
        if result.get("reasoning"):
            print(f"     {result['reasoning'][:120]}", flush=True)

        # Save after EVERY sample
        save_progress(progress_path, results)

        # Delay between requests (skip after last)
        if i < len(samples) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from scored results."""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {
            "num_evaluated": len(results),
            "num_valid": 0,
            "num_errors": len(results),
        }

    dims = [
        "vulnerability_identification",
        "severity_accuracy",
        "explanation_quality",
        "fix_suggestion",
        "relevance",
        "overall",
    ]
    metrics: dict[str, float | int] = {
        "num_evaluated": len(results),
        "num_valid": len(valid),
        "num_errors": len(results) - len(valid),
    }
    for dim in dims:
        vals = [
            r["scores"][dim]
            for r in valid
            if isinstance(r["scores"].get(dim), int | float)
        ]
        if vals:
            metrics[f"avg_{dim}"] = round(sum(vals) / len(vals), 2)
            metrics[f"min_{dim}"] = min(vals)
            metrics[f"max_{dim}"] = max(vals)

    overalls = [
        r["scores"].get("overall", 0) for r in valid if r["scores"].get("overall")
    ]
    if overalls:
        metrics["pct_good"] = round(
            sum(1 for v in overalls if v >= 4) / len(overalls), 3
        )
        metrics["pct_poor"] = round(
            sum(1 for v in overalls if v <= 2) / len(overalls), 3
        )

    return metrics


def generate_comparison(base_results: list[dict], ft_results: list[dict]) -> str:
    """Generate a side-by-side comparison report."""
    base_m = compute_metrics(base_results)
    ft_m = compute_metrics(ft_results)

    dims = [
        "overall",
        "vulnerability_identification",
        "severity_accuracy",
        "explanation_quality",
        "fix_suggestion",
        "relevance",
    ]

    lines = [
        "# Base vs Fine-tuned Model Comparison",
        "",
        f"**Judge model:** {OPENROUTER_MODEL}",
        f"**Base samples:** {base_m.get('num_valid', 0)}/{base_m.get('num_evaluated', 0)} scored",
        f"**Fine-tuned samples:** {ft_m.get('num_valid', 0)}/{ft_m.get('num_evaluated', 0)} scored",
        "",
        "## Score Comparison (1-5 scale)",
        "",
        "| Dimension | Base Avg | Fine-tuned Avg | Delta |",
        "|-----------|----------|----------------|-------|",
    ]
    for dim in dims:
        b = base_m.get(f"avg_{dim}", "N/A")
        f_ = ft_m.get(f"avg_{dim}", "N/A")
        if isinstance(b, (int, float)) and isinstance(f_, (int, float)):
            delta = round(f_ - b, 2)
            arrow = "+" if delta > 0 else ""
            lines.append(
                f"| {dim.replace('_', ' ').title()} | {b} | {f_} | {arrow}{delta} |"
            )
        else:
            lines.append(f"| {dim.replace('_', ' ').title()} | {b} | {f_} | N/A |")

    lines.extend([
        "",
        "## Quality Distribution",
        "",
        f"| Metric | Base | Fine-tuned |",
        f"|--------|------|------------|",
        f"| % Good (>=4) | {base_m.get('pct_good', 'N/A')} | {ft_m.get('pct_good', 'N/A')} |",
        f"| % Poor (<=2) | {base_m.get('pct_poor', 'N/A')} | {ft_m.get('pct_poor', 'N/A')} |",
        "",
        "## Per-Example Side-by-Side",
        "",
    ])

    n = min(len(base_results), len(ft_results))
    for i in range(n):
        br = base_results[i]
        fr = ft_results[i]
        b_overall = br.get("scores", {}).get("overall", "?")
        f_overall = fr.get("scores", {}).get("overall", "?")
        lines.extend([
            f"### Example {i + 1}",
            f"- **Base:** {b_overall}/5 — {br.get('reasoning', '')[:150]}",
            f"- **Fine-tuned:** {f_overall}/5 — {fr.get('reasoning', '')[:150]}",
            "",
        ])

    # Summary
    b_avg = base_m.get("avg_overall", 0)
    f_avg = ft_m.get("avg_overall", 0)
    if isinstance(b_avg, (int, float)) and isinstance(f_avg, (int, float)):
        if f_avg > b_avg:
            verdict = f"Fine-tuned model is BETTER by {round(f_avg - b_avg, 2)} points."
        elif f_avg < b_avg:
            verdict = f"Fine-tuned model is WORSE by {round(b_avg - f_avg, 2)} points."
        else:
            verdict = "Both models score the same."
    else:
        verdict = "Cannot determine — incomplete data."

    lines.extend(["", "## Verdict", "", verdict, ""])
    return "\n".join(lines)


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY env var required.", flush=True)
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15.0),
    )

    # --- Score base model ---
    base_results: list[dict] = []
    if BASE_INPUT.exists():
        base_results = score_file(client, BASE_INPUT, BASE_PROGRESS, "BASE")
        base_m = compute_metrics(base_results)
        BASE_FINAL.write_text(
            json.dumps({"metrics": base_m, "results": base_results}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[BASE] avg_overall={base_m.get('avg_overall', 'N/A')}", flush=True)
    else:
        print(f"\nSkipping base — {BASE_INPUT} not found", flush=True)

    # --- Score fine-tuned model ---
    ft_results: list[dict] = []
    if FT_INPUT.exists():
        ft_results = score_file(client, FT_INPUT, FT_PROGRESS, "FINETUNED")
        ft_m = compute_metrics(ft_results)
        FT_FINAL.write_text(
            json.dumps({"metrics": ft_m, "results": ft_results}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[FINETUNED] avg_overall={ft_m.get('avg_overall', 'N/A')}", flush=True)
    else:
        print(f"\nSkipping fine-tuned — {FT_INPUT} not found", flush=True)

    # --- Comparison report ---
    if base_results and ft_results:
        report = generate_comparison(base_results, ft_results)
        COMPARISON_REPORT.write_text(report, encoding="utf-8")
        print(f"\nComparison report: {COMPARISON_REPORT}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()

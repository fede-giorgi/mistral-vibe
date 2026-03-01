#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-genai",
#     "huggingface_hub",
# ]
# ///
"""
Score inference results using Gemini as an LLM-as-judge.

Downloads inference_results.json from HF dataset repo, sends each result
to Gemini for scoring on 5 security-analysis dimensions, then generates
a markdown report and JSON metrics file.

Usage:
    export GEMINI_API_KEY="..."
    uv run python scripts/judge_gemini.py
    uv run python scripts/judge_gemini.py --sample-size 50 --gemini-model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from google import genai
from huggingface_hub import HfApi, hf_hub_download, login

# ---------------------------------------------------------------------------
# Judge rubric (same as evaluate_local.py for consistency)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are an expert security auditor evaluating AI-generated vulnerability analyses.
Score the candidate response on EACH dimension below (1-5 scale).

Dimensions:
1. **vulnerability_identification** — Did it identify a real vulnerability class \
(e.g. CWE category, OWASP type)? (1=wrong/none, 5=precise CWE + description)
2. **severity_accuracy** — Is the severity assessment reasonable? \
(1=wildly off, 3=roughly correct, 5=matches CVSS-aligned ground truth)
3. **explanation_quality** — Is the explanation clear, specific, and actionable? \
(1=vague/generic, 5=cites lines, explains root cause and impact)
4. **fix_suggestion** — Does it suggest a correct remediation? \
(1=no fix or wrong fix, 5=complete, correct, production-ready fix)
5. **relevance** — Does the response actually address the code shown? \
(1=completely unrelated, 5=directly analyses the given snippet)

Return ONLY valid JSON (no markdown fences):
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
class JudgedResult:
    index: int
    code_snippet: str
    ground_truth: str
    model_response: str
    latency_ms: float
    scores: dict[str, float] = field(default_factory=dict)
    judge_reasoning: str = ""
    error: str = ""


def download_inference_results(dataset_repo: str, token: str) -> list[dict]:
    """Download inference_results.json from HF dataset repo."""
    print(f"Downloading inference results from {dataset_repo}...")
    path = hf_hub_download(
        repo_id=dataset_repo,
        filename="eval/inference_results.json",
        repo_type="dataset",
        token=token,
    )
    with Path(path).open(encoding="utf-8") as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} inference results.")
    return results


def judge_with_gemini(
    client: genai.Client,
    model_name: str,
    code: str,
    ground_truth: str,
    candidate: str,
    *,
    max_retries: int = 3,
) -> dict:
    """Use Gemini to score the candidate response."""
    prompt = f"""\
## Code Under Review
```
{code[:4000]}
```

## Ground Truth Analysis
{ground_truth[:3000]}

## Candidate Response to Evaluate
{candidate[:3000]}

Score the candidate response on all dimensions. Return ONLY valid JSON."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": [{"text": JUDGE_SYSTEM + "\n\n" + prompt}],
                    }
                ],
            )
            text = response.text or "{}"
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                return {"error": str(e), "overall": 1}
            time.sleep(2 * (attempt + 1))
    return {"error": "max retries", "overall": 1}


def compute_metrics(results: list[JudgedResult]) -> dict[str, float]:
    """Aggregate scores across all judged results."""
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

    # Average latency
    latencies = [r.latency_ms for r in valid if r.latency_ms > 0]
    if latencies:
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

    return metrics


def generate_report(metrics: dict, results: list[JudgedResult], iteration: int) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# Security Model Evaluation Report",
        f"**Iteration:** {iteration}",
        "**Model:** Ministral-8B + LoRA (security-scan-lora:v0)",
        "**Judge:** Gemini",
        f"**Samples evaluated:** {metrics.get('num_valid', 0)}/{metrics.get('num_evaluated', 0)}",
        f"**Avg latency:** {metrics.get('avg_latency_ms', 'N/A')}ms",
        "",
        "## Overall Scores (1-5 scale)",
        "",
        "| Dimension | Avg | Min | Max |",
        "|-----------|-----|-----|-----|",
    ]
    for dim in [
        "overall",
        "vulnerability_identification",
        "severity_accuracy",
        "explanation_quality",
        "fix_suggestion",
        "relevance",
    ]:
        avg = metrics.get(f"avg_{dim}", "N/A")
        mn = metrics.get(f"min_{dim}", "N/A")
        mx = metrics.get(f"max_{dim}", "N/A")
        label = dim.replace("_", " ").title()
        lines.append(f"| {label} | {avg} | {mn} | {mx} |")

    lines.extend([
        "",
        f"**% Good (score >= 4):** {metrics.get('pct_good', 'N/A')}",
        f"**% Poor (score <= 2):** {metrics.get('pct_poor', 'N/A')}",
        "",
        "## Per-Example Results",
        "",
    ])

    for r in results:
        overall = r.scores.get("overall", "?")
        lines.extend([
            f"### Example {r.index + 1} (overall: {overall}/5, latency: {r.latency_ms:.0f}ms)",
            "",
            f"**Code snippet:** `{r.code_snippet[:120]}...`",
            "",
            f"**Model response:** {r.model_response[:400]}...",
            "",
            f"**Judge reasoning:** {r.judge_reasoning}",
            "",
            f"**Scores:** vuln_id={r.scores.get('vulnerability_identification', '?')}, "
            f"severity={r.scores.get('severity_accuracy', '?')}, "
            f"explanation={r.scores.get('explanation_quality', '?')}, "
            f"fix={r.scores.get('fix_suggestion', '?')}, "
            f"relevance={r.scores.get('relevance', '?')}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score inference results with Gemini judge"
    )
    parser.add_argument(
        "--dataset-repo",
        default=None,
        help="HF dataset repo with inference_results.json (default: <username>/security-vuln-dataset)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
        help="Gemini model to use as judge",
    )
    parser.add_argument(
        "--iteration", type=int, default=1, help="Iteration number for the report"
    )
    parser.add_argument(
        "--output-dir",
        default="ai_pipeline/eval_results",
        help="Directory for evaluation output files",
    )
    parser.add_argument(
        "--local-file",
        default=None,
        help="Use a local inference_results.json instead of downloading from HF",
    )
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("Error: GEMINI_API_KEY environment variable is required.")
        raise SystemExit(1)

    # Get HF token (needed for downloading results unless --local-file)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()

    # Download or load inference results
    if args.local_file:
        print(f"Loading local inference results from {args.local_file}...")
        with Path(args.local_file).open(encoding="utf-8") as f:
            raw_results = json.load(f)
    else:
        if not hf_token:
            print("Error: HF_TOKEN required to download results from HF Hub.")
            raise SystemExit(1)
        login(token=hf_token)
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]
        dataset_repo = args.dataset_repo or f"{username}/security-vuln-dataset"
        raw_results = download_inference_results(dataset_repo, hf_token)

    print(f"\nScoring {len(raw_results)} results with {args.gemini_model}...")
    gemini_client = genai.Client(api_key=gemini_key)

    judged: list[JudgedResult] = []
    for i, result in enumerate(raw_results):
        print(
            f"\n[{i + 1}/{len(raw_results)}] Judging example {result.get('index', i)}..."
        )
        scores = judge_with_gemini(
            gemini_client,
            args.gemini_model,
            code=result["code_snippet"],
            ground_truth=result["ground_truth"],
            candidate=result["model_response"],
        )

        jr = JudgedResult(
            index=result.get("index", i),
            code_snippet=result["code_snippet"],
            ground_truth=result["ground_truth"],
            model_response=result["model_response"],
            latency_ms=result.get("latency_ms", 0.0),
            scores={k: v for k, v in scores.items() if k not in ("reasoning", "error")},
            judge_reasoning=scores.get("reasoning", ""),
            error=scores.get("error", ""),
        )
        judged.append(jr)

        overall = scores.get("overall", "?")
        print(f"  Score: {overall}/5 — {scores.get('reasoning', '')[:100]}")

        # Rate limit: Gemini free tier is 15 RPM for flash
        time.sleep(4.5)

    # Compute metrics
    metrics = compute_metrics(judged)
    print(f"\n{'=' * 60}")
    print("EVALUATION METRICS")
    print(f"{'=' * 60}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"iter_{args.iteration}.json"
    results_path.write_text(
        json.dumps(
            {
                "iteration": args.iteration,
                "metrics": metrics,
                "results": [asdict(r) for r in judged],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = generate_report(metrics, judged, args.iteration)
    report_path = out_dir / f"report_iter_{args.iteration}.md"
    report_path.write_text(report, encoding="utf-8")

    # Also upload metrics back to HF dataset repo
    if not args.local_file and hf_token:
        try:
            api = HfApi(token=hf_token)
            username = api.whoami()["name"]
            dataset_repo = args.dataset_repo or f"{username}/security-vuln-dataset"
            api.upload_file(
                path_or_fileobj=str(results_path),
                path_in_repo=f"eval/judge_results_iter_{args.iteration}.json",
                repo_id=dataset_repo,
                repo_type="dataset",
                token=hf_token,
            )
            print(f"\nJudge results uploaded to {dataset_repo}")
        except Exception as e:
            print(f"\nWarning: Failed to upload judge results: {e}")

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    print(
        f"\nOverall: avg={metrics.get('avg_overall', 'N/A')}, "
        f"good={metrics.get('pct_good', 'N/A')}, poor={metrics.get('pct_poor', 'N/A')}"
    )


if __name__ == "__main__":
    main()

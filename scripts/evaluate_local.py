# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers>=4.46.0",
#     "peft>=0.13.0",
#     "bitsandbytes",
#     "accelerate",
#     "sentencepiece",
#     "google-genai",
# ]
# ///
"""
Evaluate fine-tuned LoRA model locally with Gemini as judge.

Loads base Ministral-8B + LoRA adapter, runs inference on test set,
then uses Gemini to score each response on 5 dimensions.

Usage:
    export GEMINI_API_KEY="..."
    uv run python scripts/evaluate_local.py --sample-size 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from google import genai
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Judge rubric
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
class EvalResult:
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


def load_model(base_model: str, adapter_path: str, device: str):
    """Load base model + LoRA adapter."""
    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model: {base_model}...")
    if device == "mps":
        # MPS: no bitsandbytes, no device_map="auto" (causes accelerate bug)
        # Load to CPU first, attach LoRA, then move to MPS
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16
        )
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Moving model to MPS...")
        model = model.to("mps")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto"
        )
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(
    model, tokenizer, system: str, user: str, max_new_tokens: int = 1024
) -> tuple[str, float]:
    """Generate a response from the fine-tuned model."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = (time.monotonic() - t0) * 1000

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip(), latency


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
            # Strip markdown fences if present
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


def compute_metrics(results: list[EvalResult]) -> dict[str, float]:
    """Aggregate scores."""
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
    return metrics


def generate_report(metrics: dict, results: list[EvalResult], iteration: int) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# Security Model Evaluation Report",
        f"**Iteration:** {iteration}",
        f"**Model:** Ministral-8B + LoRA (security-scan-lora:v0)",
        f"**Judge:** Gemini",
        f"**Samples evaluated:** {metrics.get('num_valid', 0)}/{metrics.get('num_evaluated', 0)}",
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
            f"**Code snippet:** `{r.code_snippet[:100]}...`",
            "",
            f"**Model response:** {r.model_response[:300]}...",
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
    parser = argparse.ArgumentParser(description="Local eval with Gemini judge")
    parser.add_argument("--base-model", default="mistralai/Ministral-8B-Instruct-2410")
    parser.add_argument("--adapter-path", default="ai_pipeline/lora_adapter")
    parser.add_argument("--test-data", default="ai_pipeline/dataset/test.jsonl")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--gemini-model", default="gemini-2.0-flash")
    parser.add_argument("--output-dir", default="ai_pipeline/eval_results")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("Error: GEMINI_API_KEY environment variable required.")
        raise SystemExit(1)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if device == "mps":
        print("NOTE: MPS cannot use 4-bit quantization. Loading in float16.")
        print("  Ministral-8B in fp16 needs ~16GB. Your M4 24GB should handle it.")

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter_path, device)
    print(f"Model loaded on {model.device}")

    # Init Gemini
    gemini_client = genai.Client(api_key=gemini_key)

    # Load test data
    test_data = load_test_data(args.test_data)
    subset = test_data[: args.sample_size]
    print(f"\nEvaluating {len(subset)} test examples...")

    results: list[EvalResult] = []
    for i, example in enumerate(subset):
        print(f"\n[{i + 1}/{len(subset)}] Generating response...")

        # Get model response
        response, latency = generate_response(
            model, tokenizer, example["system"], example["user"]
        )
        print(f"  Response ({latency:.0f}ms): {response[:100]}...")

        # Judge with Gemini
        print("  Judging with Gemini...")
        scores = judge_with_gemini(
            gemini_client,
            args.gemini_model,
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

        overall = scores.get("overall", "?")
        print(f"  Score: {overall}/5 — {scores.get('reasoning', '')[:80]}")

    # Compute metrics
    metrics = compute_metrics(results)
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
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = generate_report(metrics, results, args.iteration)
    report_path = out_dir / f"report_iter_{args.iteration}.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    print(
        f"\nOverall: avg={metrics.get('avg_overall', 'N/A')}, "
        f"good={metrics.get('pct_good', 'N/A')}, poor={metrics.get('pct_poor', 'N/A')}"
    )


if __name__ == "__main__":
    main()

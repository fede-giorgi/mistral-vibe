# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mistralai",
# ]
# ///
"""
Analyze evaluation results and generate improved training data.

This script closes the self-improving loop by:
1. Loading eval results from the previous iteration
2. Identifying failure patterns (low-scoring examples)
3. Using a strong model to generate better training examples
4. Producing an augmented dataset for the next training round

Usage:
    export MISTRAL_API_KEY="..."
    uv run python scripts/optimize.py --iteration 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from mistralai import Mistral

# ---------------------------------------------------------------------------
# Prompt for the optimizer model to generate improved training examples
# ---------------------------------------------------------------------------
OPTIMIZER_SYSTEM = """\
You are an expert security training data curator. Your job is to create
high-quality training examples for a security vulnerability detection model.

Each example must follow this EXACT chat format:
{
  "messages": [
    {"role": "system", "content": "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation."},
    {"role": "user", "content": "Analyze the following code for security vulnerabilities:\\n\\n```\\n<code>\\n```"},
    {"role": "assistant", "content": "<detailed, accurate vulnerability analysis that directly addresses the code shown>"}
  ]
}

Rules for the assistant response:
1. MUST directly reference the code shown (cite specific functions, variables, lines)
2. MUST identify the correct CWE category
3. MUST assess severity (LOW/MEDIUM/HIGH/CRITICAL) with justification
4. MUST explain the root cause and attack vector
5. MUST suggest a specific fix with code
6. MUST be factually accurate — do NOT hallucinate vulnerabilities
"""


def load_eval_results(iteration: int) -> dict:
    """Load evaluation results from a specific iteration."""
    path = Path(f"ai_pipeline/eval_results/iter_{iteration}.json")
    if not path.exists():
        raise FileNotFoundError(f"No eval results found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def identify_failure_patterns(results: list[dict]) -> dict[str, list[dict]]:
    """Categorize failures by type for targeted improvement."""
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

    return patterns


def generate_improved_example(
    client: Mistral,
    model: str,
    code_snippet: str,
    failure_reason: str,
    original_response: str,
) -> dict | None:
    """Use a strong model to generate a better training example for a failure case."""
    prompt = f"""\
The following code was analyzed by a security model, but the analysis was poor.

## Failure reason: {failure_reason}

## Code:
```
{code_snippet[:3000]}
```

## Original (poor) response:
{original_response[:1500]}

Generate an IMPROVED, ACCURATE security analysis of this code. Your response
must directly address the specific code shown. If the code has no clear
vulnerability, say so honestly — do not fabricate issues.

Return ONLY the assistant message content (the vulnerability analysis text).
"""
    try:
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        improved_text = resp.choices[0].message.content if resp.choices else None
        if not improved_text:
            return None

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation.",
                },
                {"role": "user", "content": code_snippet},
                {"role": "assistant", "content": improved_text},
            ]
        }
    except Exception as e:
        print(f"    Error generating example: {e}")
        return None


def generate_synthetic_examples(
    client: Mistral, model: str, weakness_area: str, count: int = 5
) -> list[dict]:
    """Generate entirely new training examples targeting a weakness area."""
    prompt = f"""\
Generate {count} diverse, realistic security vulnerability training examples.

Focus area: {weakness_area}

For each example, provide:
1. A realistic code snippet (C, C++, Python, JavaScript, or Java) that contains
   a real vulnerability related to {weakness_area}
2. A detailed, accurate analysis of the vulnerability

Return a JSON array where each element has:
{{
  "code": "<the vulnerable code>",
  "analysis": "<detailed vulnerability analysis>"
}}

Make the code realistic — use real library calls, real patterns from production code.
Vary the languages and specific vulnerability subtypes.
"""
    try:
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        text = resp.choices[0].message.content if resp.choices else "[]"
        data = json.loads(text or "[]")

        # Handle both array and object-with-array responses
        examples_raw = data if isinstance(data, list) else data.get("examples", [])

        examples: list[dict] = []
        for item in examples_raw:
            code = item.get("code", "")
            analysis = item.get("analysis", "")
            if code and analysis:
                examples.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation.",
                        },
                        {
                            "role": "user",
                            "content": f"Analyze the following code for security vulnerabilities:\n\n```\n{code}\n```",
                        },
                        {"role": "assistant", "content": analysis},
                    ]
                })
        return examples
    except Exception as e:
        print(f"    Error generating synthetic examples: {e}")
        return []


def create_optimization_report(
    patterns: dict[str, list[dict]],
    metrics: dict,
    new_examples_count: int,
    improved_examples_count: int,
) -> str:
    """Generate a human-readable optimization report."""
    report_lines = [
        "# Optimization Report",
        f"## Iteration Metrics",
        f"- Average overall score: {metrics.get('avg_overall', 'N/A')}",
        f"- % good (>=4): {metrics.get('pct_good', 'N/A')}",
        f"- % poor (<=2): {metrics.get('pct_poor', 'N/A')}",
        "",
        "## Failure Pattern Analysis",
    ]
    for pattern_name, examples in patterns.items():
        report_lines.append(f"- **{pattern_name}**: {len(examples)} cases")

    report_lines.extend([
        "",
        "## Actions Taken",
        f"- Improved {improved_examples_count} failing examples",
        f"- Generated {new_examples_count} synthetic examples",
        "",
        "## Recommendations for Next Iteration",
    ])

    # Auto-generate recommendations based on failure patterns
    if len(patterns["low_relevance"]) > 3:
        report_lines.append(
            "- HIGH PRIORITY: Model responses are not relevant to the code shown. "
            "The dataset likely has mismatched code-vulnerability pairs. "
            "Consider regenerating these pairs with correct matching."
        )
    if len(patterns["missing_fix"]) > 3:
        report_lines.append(
            "- MEDIUM: Model is not suggesting fixes. "
            "Add more examples with explicit remediation code."
        )
    if len(patterns["wrong_vulnerability"]) > 3:
        report_lines.append(
            "- HIGH: Model misidentifies vulnerability types. "
            "Add more diverse CWE examples to training set."
        )

    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize training data based on eval results"
    )
    parser.add_argument(
        "--iteration", type=int, required=True, help="Iteration to analyze"
    )
    parser.add_argument(
        "--optimizer-model",
        default="mistral-large-latest",
        help="Model used to generate improved examples",
    )
    parser.add_argument(
        "--max-improved", type=int, default=20, help="Max failing examples to fix"
    )
    parser.add_argument(
        "--synthetic-per-weakness",
        type=int,
        default=5,
        help="Synthetic examples per weakness area",
    )
    parser.add_argument(
        "--output-dir",
        default="ai_pipeline/dataset",
        help="Directory for augmented dataset",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY required.")
        raise SystemExit(1)

    client = Mistral(api_key=api_key)

    # 1. Load eval results
    print(f"Loading eval results for iteration {args.iteration}...")
    eval_data = load_eval_results(args.iteration)
    metrics = eval_data["metrics"]
    results = eval_data["results"]
    print(f"  {len(results)} results, avg_overall={metrics.get('avg_overall', 'N/A')}")

    # 2. Identify failure patterns
    print("Analyzing failure patterns...")
    patterns = identify_failure_patterns(results)
    for name, examples in patterns.items():
        print(f"  {name}: {len(examples)} cases")

    # 3. Fix failing examples
    print(f"\nGenerating improved examples (max {args.max_improved})...")
    improved_examples: list[dict] = []
    failures = sorted(
        [r for r in results if r.get("scores", {}).get("overall", 5) <= 3],
        key=lambda r: r.get("scores", {}).get("overall", 5),
    )

    for r in failures[: args.max_improved]:
        # Determine primary failure reason
        scores = r.get("scores", {})
        if scores.get("relevance", 5) <= 2:
            reason = "Response is not relevant to the code shown"
        elif scores.get("vulnerability_identification", 5) <= 2:
            reason = "Wrong vulnerability type identified"
        elif scores.get("fix_suggestion", 5) <= 2:
            reason = "No actionable fix suggested"
        else:
            reason = "Low overall quality"

        print(
            f"  Fixing example {r['index']} (overall={scores.get('overall', '?')}, reason={reason})..."
        )
        improved = generate_improved_example(
            client,
            args.optimizer_model,
            code_snippet=r.get("code_snippet", ""),
            failure_reason=reason,
            original_response=r.get("model_response", ""),
        )
        if improved:
            improved_examples.append(improved)
        time.sleep(1)

    print(f"  Generated {len(improved_examples)} improved examples")

    # 4. Generate synthetic examples for weak areas
    print("\nGenerating synthetic examples for weakness areas...")
    synthetic_examples: list[dict] = []
    weakness_areas = [name for name, cases in patterns.items() if len(cases) >= 2]

    cwe_focus_map = {
        "low_relevance": "code relevance — ensure analysis matches the specific code shown",
        "poor_explanation": "clear explanations with root cause analysis and impact assessment",
        "wrong_vulnerability": "diverse CWE categories including injection, memory safety, auth bypass, and crypto issues",
        "missing_fix": "remediation suggestions with corrected code examples",
        "low_overall": "comprehensive vulnerability analysis covering identification, severity, explanation, and fix",
    }

    for weakness in weakness_areas:
        focus = cwe_focus_map.get(weakness, weakness)
        print(f"  Generating {args.synthetic_per_weakness} examples for: {weakness}...")
        new_examples = generate_synthetic_examples(
            client, args.optimizer_model, focus, args.synthetic_per_weakness
        )
        synthetic_examples.extend(new_examples)
        time.sleep(2)

    print(f"  Generated {len(synthetic_examples)} synthetic examples")

    # 5. Write augmented dataset
    output_dir = Path(args.output_dir)
    augmented_path = output_dir / f"augmented_iter_{args.iteration + 1}.jsonl"

    # Load existing training data
    train_path = output_dir / "train.jsonl"
    existing: list[dict] = []
    if train_path.exists():
        with train_path.open(encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]

    all_examples = existing + improved_examples + synthetic_examples
    with augmented_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nAugmented dataset written to {augmented_path}")
    print(f"  Original: {len(existing)} examples")
    print(f"  + Improved: {len(improved_examples)}")
    print(f"  + Synthetic: {len(synthetic_examples)}")
    print(f"  = Total: {len(all_examples)} examples")

    # 6. Write optimization report
    report = create_optimization_report(
        patterns, metrics, len(synthetic_examples), len(improved_examples)
    )
    report_path = Path(
        f"ai_pipeline/eval_results/optimization_report_iter_{args.iteration}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\nOptimization report: {report_path}")
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 7 — Remediation loop entry point.

How to verify it works (no API key needed):
  uv run pytest tests/ai_pipeline/remediation/ -v
  uv run python ai_pipeline/7_remediation.py --mock path/to/file.py

Once you have MISTRAL_API_KEY (and your fine-tuned model ID if applicable),
run without --mock to use the real analyzer and Devstral fixer:
  export MISTRAL_API_KEY=your_key
  uv run python ai_pipeline/7_remediation.py path/to/file.py
  # With fine-tuned model:
  uv run python ai_pipeline/7_remediation.py --model ft:your-model-id path/to/file.py

Usage:
    # Mock analyzer + mock fixer (no API key needed):
    uv run python ai_pipeline/7_remediation.py --mock examples/vuln.py

    # Real fine-tuned model + Devstral fixer (requires MISTRAL_API_KEY):
    uv run python ai_pipeline/7_remediation.py --model ft:your-model-id examples/vuln.py

    # Read from stdin:
    cat vuln.py | uv run python ai_pipeline/7_remediation.py --mock -

    # Multiple iterations, JSON output:
    uv run python ai_pipeline/7_remediation.py --mock --max-iterations 5 --output-json results.json examples/vuln.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ai_pipeline.remediation.analyzer import Analyzer, MistralAnalyzer, MockAnalyzer
from ai_pipeline.remediation.fixer import DevstralFixer, MockFixer
from ai_pipeline.remediation.loop import run_remediation_loop, DEFAULT_MAX_ITERATIONS
from ai_pipeline.remediation.models import CodeEdit, RemediationSummary


def print_summary(summary: RemediationSummary) -> None:
    """Pretty-print the remediation summary to stdout."""
    print("\n" + "=" * 60)
    print("REMEDIATION LOOP SUMMARY")
    print("=" * 60)
    print(f"  Iterations run:       {len(summary.iterations)}")
    print(f"  Initial findings:     {summary.total_findings_initial}")
    print(f"  Resolved:             {summary.total_resolved}")
    print(f"  Still persisting:     {summary.total_persisting}")
    print(f"  New (regressions):    {summary.total_new}")
    print(f"  Converged:            {summary.converged}")
    print("=" * 60)

    for result in summary.iterations:
        print(f"\n--- Iteration {result.iteration} ---")
        print(f"  Before: {len(result.analysis_before.findings)} finding(s)")
        print(f"  Edits applied: {len(result.fixer_output.edits)}")
        print(f"  After:  {len(result.analysis_after.findings)} finding(s)")
        for delta in result.deltas:
            status_icon = {"resolved": "+", "persists": "x", "new": "!"}.get(delta.status, "?")
            label = delta.before.title if delta.before else (delta.after.title if delta.after else delta.finding_id)
            print(f"    [{status_icon}] {delta.status.upper():>10}  {delta.finding_id}: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Remediation loop — analyzer -> Devstral fixer -> re-analyzer -> comparator",
    )
    parser.add_argument(
        "file",
        help="Path to the source file to analyze (use '-' for stdin)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock analyzer instead of the real fine-tuned model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Fine-tuned model ID for the analyzer (overrides FINETUNED_MODEL env var)",
    )
    parser.add_argument(
        "--fixer-model",
        default="devstral-small-latest",
        help="Model ID for Devstral fixer (default: devstral-small-latest)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum remediation iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write full JSON results to this path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.file == "-":
        code = sys.stdin.read()
        file_path = "<stdin>"
    else:
        path = Path(args.file)
        if not path.is_file():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        code = path.read_text(encoding="utf-8")
        file_path = str(path)

    analyzer: Analyzer
    if args.mock:
        analyzer = MockAnalyzer()
    else:
        analyzer = MistralAnalyzer(model_id=args.model)

    # With --mock, use MockFixer too so no API key is needed (avoids "Bearer " header error).
    if args.mock:
        # One edit that matches the given code so the loop runs: apply -> re-analyze -> 0 findings -> converged.
        if code.strip():
            mock_edits = [
                CodeEdit(
                    file_path=file_path,
                    original=code,
                    replacement=code.rstrip() + "\n  # fixed by mock\n",
                    finding_id="F-001",
                    rationale="Mock fix for --mock mode",
                ),
            ]
        else:
            mock_edits = []
        fixer = MockFixer(edits=mock_edits)
    else:
        fixer = DevstralFixer(model_id=args.fixer_model)

    summary = run_remediation_loop(
        code=code,
        file_path=file_path,
        analyzer=analyzer,
        fixer=fixer,
        max_iterations=args.max_iterations,
    )

    print_summary(summary)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()

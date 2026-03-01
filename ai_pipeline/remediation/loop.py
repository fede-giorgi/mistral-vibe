"""Remediation loop orchestrator.

Flow per iteration:
  1. Analyzer (fine-tuned model) scans code -> findings
  2. Devstral receives code + findings -> produces edits
  3. Edits are applied to produce fixed code
  4. Analyzer re-scans fixed code -> new findings
  5. Comparator diffs before/after findings

The loop repeats until all findings are resolved or max iterations reached.
"""

from __future__ import annotations

import logging

from ai_pipeline.remediation.analyzer import Analyzer
from ai_pipeline.remediation.comparator import build_remediation_result
from ai_pipeline.remediation.fixer import Fixer, apply_edits
from ai_pipeline.remediation.models import RemediationResult, RemediationSummary

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3


def run_remediation_loop(
    code: str,
    file_path: str,
    analyzer: Analyzer,
    fixer: Fixer,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> RemediationSummary:
    """Execute the full analyze -> fix -> re-analyze -> compare loop."""
    current_code = code
    iterations: list[RemediationResult] = []
    total_initial = 0

    for i in range(1, max_iterations + 1):
        logger.info("=== Remediation iteration %d / %d ===", i, max_iterations)

        # Step 1: Analyze current code
        logger.info("Step 1: Running analyzer on %s …", file_path)
        analysis_before = analyzer.analyze(current_code, file_path)
        finding_count = len(analysis_before.findings)
        logger.info("  Found %d finding(s).", finding_count)

        if i == 1:
            total_initial = finding_count

        if finding_count == 0:
            logger.info("No findings — code is clean. Stopping.")
            result = build_remediation_result(
                iteration=i,
                analysis_before=analysis_before,
                fixer_output=fixer.fix(current_code, [], file_path),
                analysis_after=analysis_before,
            )
            iterations.append(result)
            break

        # Step 2: Ask Devstral to fix
        logger.info("Step 2: Asking Devstral to fix %d finding(s) …", finding_count)
        fixer_output = fixer.fix(current_code, analysis_before.findings, file_path)
        logger.info("  Devstral proposed %d edit(s).", len(fixer_output.edits))

        if not fixer_output.edits:
            logger.warning("  Devstral returned no edits — stopping loop.")
            result = build_remediation_result(
                iteration=i,
                analysis_before=analysis_before,
                fixer_output=fixer_output,
                analysis_after=analysis_before,
            )
            iterations.append(result)
            break

        # Step 3: Apply edits
        logger.info("Step 3: Applying edits …")
        fixed_code = apply_edits(current_code, fixer_output.edits)

        # Step 4: Re-analyze fixed code
        logger.info("Step 4: Re-analyzing fixed code …")
        analysis_after = analyzer.analyze(fixed_code, file_path)
        remaining = len(analysis_after.findings)
        logger.info("  %d finding(s) remaining after fix.", remaining)

        # Step 5: Compare
        logger.info("Step 5: Comparing before/after …")
        result = build_remediation_result(
            iteration=i,
            analysis_before=analysis_before,
            fixer_output=fixer_output,
            analysis_after=analysis_after,
        )
        iterations.append(result)

        logger.info(
            "  Resolved: %d | Persisting: %d | New: %d | All resolved: %s",
            result.resolved_count,
            result.persisting_count,
            result.new_count,
            result.all_resolved,
        )

        if result.all_resolved:
            logger.info("All findings resolved! Stopping.")
            break

        current_code = fixed_code

    total_resolved = sum(r.resolved_count for r in iterations)
    last = iterations[-1] if iterations else None
    total_persisting = last.persisting_count if last else 0
    total_new = last.new_count if last else 0
    converged = last.all_resolved if last else False

    return RemediationSummary(
        iterations=iterations,
        total_findings_initial=total_initial,
        total_resolved=total_resolved,
        total_persisting=total_persisting,
        total_new=total_new,
        converged=converged,
    )

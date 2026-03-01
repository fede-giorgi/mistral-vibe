"""Comparison logic for before/after analyzer findings.

Determines which findings were resolved, which persist, and which are new
(regressions introduced by the fix).
"""

from __future__ import annotations

from ai_pipeline.remediation.models import (
    AnalyzerOutput,
    FindingDelta,
    FixerOutput,
    RemediationResult,
)


def _match_key(cwe: str | None, title: str) -> str:
    """Build a fuzzy-matching key from a finding's identity fields.

    When IDs are auto-generated and differ between runs, we fall back to
    (cwe, title-prefix) so structurally identical findings still match.
    """
    prefix = title[:60].strip().lower() if title else ""
    return f"{cwe or ''}|{prefix}"


def compare(before: AnalyzerOutput, after: AnalyzerOutput) -> list[FindingDelta]:
    """Compare two analyzer outputs and produce a delta list.

    Returns one FindingDelta per unique finding across both runs.
    """
    before_by_id = {f.id: f for f in before.findings}
    after_by_id = {f.id: f for f in after.findings}

    before_by_key = {_match_key(f.cwe, f.title): f for f in before.findings}
    after_by_key = {_match_key(f.cwe, f.title): f for f in after.findings}

    deltas: list[FindingDelta] = []
    matched_after_ids: set[str] = set()

    for fid, bf in before_by_id.items():
        # Exact ID match
        if fid in after_by_id:
            af = after_by_id[fid]
            matched_after_ids.add(af.id)
            deltas.append(FindingDelta(
                finding_id=fid,
                status="persists",
                before=bf,
                after=af,
                notes="Finding still present after fix attempt.",
            ))
            continue

        # Fuzzy match by CWE + title prefix
        bkey = _match_key(bf.cwe, bf.title)
        if bkey in after_by_key:
            af = after_by_key[bkey]
            matched_after_ids.add(af.id)
            deltas.append(FindingDelta(
                finding_id=fid,
                status="persists",
                before=bf,
                after=af,
                notes=f"Matched by CWE+title (after-id: {af.id}).",
            ))
            continue

        # Not found in after → resolved
        deltas.append(FindingDelta(
            finding_id=fid,
            status="resolved",
            before=bf,
            after=None,
            notes="Finding no longer reported after fix.",
        ))

    # Anything in after that wasn't matched → new regression
    for fid, af in after_by_id.items():
        if fid not in matched_after_ids:
            akey = _match_key(af.cwe, af.title)
            if akey not in before_by_key:
                deltas.append(FindingDelta(
                    finding_id=fid,
                    status="new",
                    before=None,
                    after=af,
                    notes="New finding introduced after fix (possible regression).",
                ))

    return deltas


def build_remediation_result(
    iteration: int,
    analysis_before: AnalyzerOutput,
    fixer_output: FixerOutput,
    analysis_after: AnalyzerOutput,
) -> RemediationResult:
    """Convenience builder that runs comparison and aggregates counts."""
    deltas = compare(analysis_before, analysis_after)

    resolved = sum(1 for d in deltas if d.status == "resolved")
    persisting = sum(1 for d in deltas if d.status == "persists")
    new = sum(1 for d in deltas if d.status == "new")

    return RemediationResult(
        iteration=iteration,
        analysis_before=analysis_before,
        fixer_output=fixer_output,
        analysis_after=analysis_after,
        deltas=deltas,
        resolved_count=resolved,
        persisting_count=persisting,
        new_count=new,
        all_resolved=(persisting == 0 and new == 0),
    )

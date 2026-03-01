"""Tests for the remediation loop — run with: uv run pytest tests/ai_pipeline/remediation/ -v

These tests use MockAnalyzer and MockFixer only. No API keys or network required.
You can be 100% sure the pipeline works if all tests pass.
"""

from __future__ import annotations

import pytest

from ai_pipeline.remediation.analyzer import MockAnalyzer
from ai_pipeline.remediation.comparator import build_remediation_result, compare
from ai_pipeline.remediation.fixer import MockFixer, apply_edits
from ai_pipeline.remediation.loop import run_remediation_loop
from ai_pipeline.remediation.models import (
    AnalyzerOutput,
    CodeEdit,
    Finding,
    FixerOutput,
    Severity,
)


# --- MockAnalyzer behaviour ---


def test_mock_analyzer_returns_findings_on_first_call_then_empty() -> None:
    """MockAnalyzer returns 2 findings on first analyze(), 0 on second (simulates fix)."""
    analyzer = MockAnalyzer()
    out1 = analyzer.analyze("any code", "test.py")
    out2 = analyzer.analyze("any code", "test.py")

    assert len(out1.findings) == 2
    assert out1.findings[0].id == "F-001"
    assert out1.findings[1].id == "F-002"
    assert len(out2.findings) == 0


def test_mock_analyzer_with_custom_fixtures() -> None:
    """MockAnalyzer with fixtures returns them on first call, empty on second."""
    fixtures = [
        Finding(
            id="F-A",
            file_path="x.py",
            title="Custom finding",
            explanation="Test",
            cwe="CWE-79",
            severity=Severity.HIGH,
        ),
    ]
    analyzer = MockAnalyzer(fixtures=fixtures)
    out1 = analyzer.analyze("code", "x.py")
    out2 = analyzer.analyze("code", "x.py")

    assert len(out1.findings) == 1
    assert out1.findings[0].id == "F-A"
    assert out1.findings[0].title == "Custom finding"
    assert len(out2.findings) == 0


# --- apply_edits ---


def test_apply_edits_applies_single_edit() -> None:
    code = "x = 1"
    edits = [
        CodeEdit(
            file_path="test.py",
            original="x = 1",
            replacement="x = 2  # fixed",
            finding_id="F-001",
            rationale="test",
        ),
    ]
    result = apply_edits(code, edits)
    assert result == "x = 2  # fixed"


def test_apply_edits_applies_multiple_edits_in_order() -> None:
    code = "a = 1\nb = 2"
    edits = [
        CodeEdit(
            file_path="test.py",
            original="a = 1",
            replacement="a = 10",
            finding_id="F-001",
            rationale="",
        ),
        CodeEdit(
            file_path="test.py",
            original="b = 2",
            replacement="b = 20",
            finding_id="F-002",
            rationale="",
        ),
    ]
    result = apply_edits(code, edits)
    assert result == "a = 10\nb = 20"


def test_apply_edits_skips_non_matching_edit() -> None:
    code = "x = 1"
    edits = [
        CodeEdit(
            file_path="test.py",
            original="y = 2",
            replacement="y = 3",
            finding_id="F-001",
            rationale="",
        ),
    ]
    result = apply_edits(code, edits)
    assert result == "x = 1"


# --- compare / build_remediation_result ---


def test_compare_all_resolved() -> None:
    """When 'after' has no findings, all before findings are resolved."""
    f1 = Finding(id="F-001", file_path="x.py", title="SQL", explanation="e", cwe="CWE-89", severity=Severity.HIGH)
    f2 = Finding(id="F-002", file_path="x.py", title="XSS", explanation="e", cwe="CWE-79", severity=Severity.MEDIUM)
    before = AnalyzerOutput(findings=[f1, f2], model_id="test")
    after = AnalyzerOutput(findings=[], model_id="test")

    deltas = compare(before, after)

    assert len(deltas) == 2
    statuses = {d.finding_id: d.status for d in deltas}
    assert statuses["F-001"] == "resolved"
    assert statuses["F-002"] == "resolved"


def test_compare_all_persist() -> None:
    """When after has same findings (same id), they persist."""
    f1 = Finding(id="F-001", file_path="x.py", title="SQL", explanation="e", cwe="CWE-89", severity=Severity.HIGH)
    before = AnalyzerOutput(findings=[f1], model_id="test")
    after = AnalyzerOutput(findings=[f1], model_id="test")

    deltas = compare(before, after)

    assert len(deltas) == 1
    assert deltas[0].status == "persists"


def test_compare_new_regression() -> None:
    """When after has a finding not in before, it's 'new'."""
    before = AnalyzerOutput(findings=[], model_id="test")
    f_new = Finding(id="F-003", file_path="x.py", title="New bug", explanation="e", cwe="CWE-22", severity=Severity.LOW)
    after = AnalyzerOutput(findings=[f_new], model_id="test")

    deltas = compare(before, after)

    assert len(deltas) == 1
    assert deltas[0].status == "new"
    assert deltas[0].after is f_new


def test_build_remediation_result_counts() -> None:
    f1 = Finding(id="F-001", file_path="x.py", title="A", explanation="e", cwe="CWE-89", severity=Severity.HIGH)
    before = AnalyzerOutput(findings=[f1], model_id="test")
    after = AnalyzerOutput(findings=[], model_id="test")
    fixer_out = FixerOutput(edits=[], model_id="test")

    result = build_remediation_result(1, before, fixer_out, after)

    assert result.resolved_count == 1
    assert result.persisting_count == 0
    assert result.new_count == 0
    assert result.all_resolved is True


# --- Full loop E2E (no API) ---


def test_run_remediation_loop_converges_with_mock_analyzer_and_mock_fixer() -> None:
    """Full loop: MockAnalyzer (2 findings -> 0 on re-run) + MockFixer (1 edit).
    No API calls. Proves the pipeline converges and marks all resolved."""
    code = "x = 1"
    # Edit that actually matches the code so apply_edits runs; re-analyze then sees "fixed" code and mock returns 0
    mock_edits = [
        CodeEdit(
            file_path="<stdin>",
            original="x = 1",
            replacement="x = 1  # fixed",
            finding_id="F-001",
            rationale="test",
        ),
    ]
    analyzer = MockAnalyzer()
    fixer = MockFixer(edits=mock_edits)

    summary = run_remediation_loop(
        code=code,
        file_path="<stdin>",
        analyzer=analyzer,
        fixer=fixer,
        max_iterations=3,
    )

    assert len(summary.iterations) == 1
    assert summary.total_findings_initial == 2
    assert summary.total_resolved == 2
    assert summary.total_persisting == 0
    assert summary.total_new == 0
    assert summary.converged is True
    assert summary.iterations[0].all_resolved is True


def test_run_remediation_loop_stops_when_no_findings() -> None:
    """If the analyzer returns 0 findings immediately, we get one iteration and converged."""
    analyzer = MockAnalyzer(fixtures=[])  # empty fixtures -> always return empty
    fixer = MockFixer(edits=[])

    summary = run_remediation_loop(
        code="clean code",
        file_path="clean.py",
        analyzer=analyzer,
        fixer=fixer,
        max_iterations=3,
    )

    assert len(summary.iterations) == 1
    assert summary.total_findings_initial == 0
    assert summary.converged is True


def test_run_remediation_loop_stops_when_fixer_returns_no_edits() -> None:
    """If fixer returns no edits, loop stops after one iteration without converging."""
    analyzer = MockAnalyzer()  # 2 findings first time
    fixer = MockFixer(edits=[])  # no edits

    summary = run_remediation_loop(
        code="vulnerable code",
        file_path="vuln.py",
        analyzer=analyzer,
        fixer=fixer,
        max_iterations=3,
    )

    assert len(summary.iterations) == 1
    assert summary.total_findings_initial == 2
    assert summary.converged is False
    assert summary.total_persisting == 2

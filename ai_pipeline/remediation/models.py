"""Data contracts for the remediation loop pipeline.

Defines the structured types flowing between analyzer -> fixer -> re-analyzer -> comparator.
"""

from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class Severity(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class Finding(BaseModel):
    """A single vulnerability detected by the analyzer."""

    id: str = Field(description="Unique identifier for this finding (e.g. 'F-001')")
    file_path: str = Field(description="Path to the affected file")
    start_line: int | None = Field(default=None, description="First affected line")
    end_line: int | None = Field(default=None, description="Last affected line")
    cwe: str | None = Field(default=None, description="CWE identifier (e.g. 'CWE-79')")
    severity: Severity = Field(default=Severity.MEDIUM)
    title: str = Field(description="Short description of the vulnerability")
    explanation: str = Field(description="Detailed explanation of the issue and its risk")
    snippet: str | None = Field(default=None, description="Relevant code snippet")


class AnalyzerOutput(BaseModel):
    """Structured output from the security analyzer (fine-tuned model)."""

    findings: list[Finding] = Field(default_factory=list)
    raw_response: str = Field(default="", description="Raw model output before parsing")
    model_id: str = Field(default="", description="Model that produced this analysis")


class CodeEdit(BaseModel):
    """A single edit proposed by the fixer."""

    file_path: str
    original: str = Field(description="Original code to be replaced")
    replacement: str = Field(description="Fixed code")
    finding_id: str = Field(description="ID of the finding this edit addresses")
    rationale: str = Field(default="", description="Why this fix resolves the finding")


class FixerOutput(BaseModel):
    """Structured output from the Devstral fixer."""

    edits: list[CodeEdit] = Field(default_factory=list)
    raw_response: str = Field(default="", description="Raw model output before parsing")
    model_id: str = Field(default="", description="Model that produced these fixes")


class FindingDelta(BaseModel):
    """Comparison result for a single finding between before and after analysis."""

    finding_id: str
    status: str = Field(description="'resolved' | 'persists' | 'regressed' | 'new'")
    before: Finding | None = None
    after: Finding | None = None
    notes: str = ""


class RemediationResult(BaseModel):
    """Full result of one remediation cycle (analyze -> fix -> re-analyze -> compare)."""

    iteration: int = Field(default=1)
    analysis_before: AnalyzerOutput
    fixer_output: FixerOutput
    analysis_after: AnalyzerOutput
    deltas: list[FindingDelta] = Field(default_factory=list)
    resolved_count: int = 0
    persisting_count: int = 0
    new_count: int = 0
    all_resolved: bool = False


class RemediationSummary(BaseModel):
    """Summary across all iterations of the remediation loop."""

    iterations: list[RemediationResult] = Field(default_factory=list)
    total_findings_initial: int = 0
    total_resolved: int = 0
    total_persisting: int = 0
    total_new: int = 0
    converged: bool = False
    final_code: str | None = None

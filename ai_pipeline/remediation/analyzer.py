"""Security analyzer abstraction.

Provides a protocol for the analyzer and two implementations:
- MockAnalyzer: returns fixture findings for development/testing.
- MistralAnalyzer: calls the fine-tuned model via the Mistral API (swap in when ready).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol

from mistralai import Mistral

from ai_pipeline.remediation.models import AnalyzerOutput, Finding, Severity

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_PROMPT = (
    "You are a senior security engineer. Analyze the provided code for security "
    "vulnerabilities. Respond ONLY with a JSON object containing a 'findings' array. "
    "Each finding must have: id (string, e.g. 'F-001'), file_path (string), "
    "start_line (int or null), end_line (int or null), cwe (string or null, e.g. 'CWE-79'), "
    "severity ('low', 'medium', or 'high'), title (string), explanation (string), "
    "snippet (string or null)."
)


class Analyzer(Protocol):
    """Protocol that any analyzer implementation must satisfy."""

    def analyze(self, code: str, file_path: str = "<stdin>") -> AnalyzerOutput: ...


class MockAnalyzer:
    """Returns hardcoded findings for development and testing.

    Pass your own fixtures list, or leave it None to get sensible defaults.
    """

    def __init__(self, fixtures: list[Finding] | None = None) -> None:
        self._fixtures = fixtures
        self._call_count = 0

    def analyze(self, code: str, file_path: str = "<stdin>") -> AnalyzerOutput:
        self._call_count += 1

        if self._fixtures is not None:
            # On re-analysis (second call onwards), return empty to simulate
            # "all findings resolved" — makes the loop testable end-to-end.
            if self._call_count > 1:
                return AnalyzerOutput(findings=[], raw_response="<mock-clean>", model_id="mock")
            return AnalyzerOutput(
                findings=self._fixtures,
                raw_response="<mock>",
                model_id="mock",
            )

        # Default fixtures: return findings on first call, clean on second
        if self._call_count > 1:
            return AnalyzerOutput(findings=[], raw_response="<mock-clean>", model_id="mock")

        return AnalyzerOutput(
            findings=[
                Finding(
                    id="F-001",
                    file_path=file_path,
                    start_line=1,
                    end_line=5,
                    cwe="CWE-89",
                    severity=Severity.HIGH,
                    title="SQL Injection via unsanitized user input",
                    explanation=(
                        "User-controlled input is concatenated directly into a SQL "
                        "query string without parameterization, enabling arbitrary "
                        "SQL execution."
                    ),
                    snippet=code[:500] if code else None,
                ),
                Finding(
                    id="F-002",
                    file_path=file_path,
                    start_line=10,
                    end_line=12,
                    cwe="CWE-79",
                    severity=Severity.MEDIUM,
                    title="Reflected XSS in template rendering",
                    explanation=(
                        "User input is rendered in an HTML template without "
                        "escaping, allowing script injection."
                    ),
                    snippet=None,
                ),
            ],
            raw_response="<mock-default-fixtures>",
            model_id="mock",
        )


class MistralAnalyzer:
    """Calls a Mistral model to analyze code for security vulnerabilities.

    Set ``model_id`` to your fine-tuned model ID once training completes.
    Until then it falls back to ``FINETUNED_MODEL`` env var or ``mistral-large-latest``.
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self._model_id = model_id or os.getenv("FINETUNED_MODEL", "mistral-large-latest")
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = Mistral(api_key=self._api_key)

    def analyze(self, code: str, file_path: str = "<stdin>") -> AnalyzerOutput:
        raw = self._call_model(code)
        findings = self._parse_findings(raw, file_path)
        return AnalyzerOutput(findings=findings, raw_response=raw, model_id=self._model_id)

    def _call_model(self, code: str) -> str:
        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this code for security vulnerabilities:\n\n```\n{code}\n```"},
        ]

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.complete(
                    model=self._model_id,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error("Analyzer call failed after %d attempts: %s", self._max_retries, e)
                    return json.dumps({"findings": [], "error": str(e)})
                logger.warning("Attempt %d failed: %s — retrying in %.1fs", attempt + 1, e, self._retry_delay)
                time.sleep(self._retry_delay)

        return json.dumps({"findings": []})

    @staticmethod
    def _parse_findings(raw: str, file_path: str) -> list[Finding]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse analyzer JSON response")
            return []

        raw_findings = data.get("findings", [])
        findings: list[Finding] = []
        for i, item in enumerate(raw_findings):
            try:
                finding = Finding(
                    id=item.get("id", f"F-{i + 1:03d}"),
                    file_path=item.get("file_path", file_path),
                    start_line=item.get("start_line"),
                    end_line=item.get("end_line"),
                    cwe=item.get("cwe"),
                    severity=Severity(item.get("severity", "medium")),
                    title=item.get("title", "Unknown vulnerability"),
                    explanation=item.get("explanation", ""),
                    snippet=item.get("snippet"),
                )
                findings.append(finding)
            except Exception as e:
                logger.warning("Skipping malformed finding at index %d: %s", i, e)
        return findings

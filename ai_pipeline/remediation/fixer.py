"""Devstral-powered code fixer.

Takes source code + a list of findings from the analyzer and asks Devstral
to produce concrete edits that resolve each vulnerability.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol

from mistralai import Mistral

from ai_pipeline.remediation.models import CodeEdit, Finding, FixerOutput

logger = logging.getLogger(__name__)


class Fixer(Protocol):
    """Protocol for any fixer implementation (real or mock)."""

    def fix(self, code: str, findings: list[Finding], file_path: str = "<stdin>") -> FixerOutput: ...

FIXER_SYSTEM_PROMPT = """\
You are Devstral, an expert secure-code engineer. You receive:
1. Source code with security vulnerabilities.
2. A structured list of findings describing each vulnerability.

Your job is to produce minimal, targeted edits that fix every finding without
breaking functionality. Respond ONLY with a JSON object:

{
  "edits": [
    {
      "file_path": "<path>",
      "original": "<exact original code to replace>",
      "replacement": "<fixed code>",
      "finding_id": "<id of the finding this fixes>",
      "rationale": "<why this edit fixes the vulnerability>"
    }
  ]
}

Rules:
- Each edit must reference a finding_id from the input.
- The "original" field must be an exact substring of the provided source code.
- Keep edits as small as possible — only change what is necessary.
- Do NOT introduce new dependencies unless absolutely required.
- Preserve the coding style and indentation of the original source.
"""


def _format_findings_for_prompt(findings: list[Finding]) -> str:
    items: list[str] = []
    for f in findings:
        parts = [
            f"- **{f.id}** [{f.severity.value.upper()}]: {f.title}",
            f"  CWE: {f.cwe or 'N/A'}",
            f"  Lines: {f.start_line or '?'}-{f.end_line or '?'}",
            f"  Explanation: {f.explanation}",
        ]
        if f.snippet:
            parts.append(f"  Snippet:\n```\n{f.snippet}\n```")
        items.append("\n".join(parts))
    return "\n\n".join(items)


class DevstralFixer:
    """Calls Devstral to produce code edits that fix reported vulnerabilities."""

    def __init__(
        self,
        model_id: str = "devstral-small-latest",
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self._model_id = model_id
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = Mistral(api_key=self._api_key)

    def fix(self, code: str, findings: list[Finding], file_path: str = "<stdin>") -> FixerOutput:
        if not findings:
            return FixerOutput(edits=[], raw_response="", model_id=self._model_id)

        raw = self._call_model(code, findings)
        edits = self._parse_edits(raw, file_path)
        return FixerOutput(edits=edits, raw_response=raw, model_id=self._model_id)

    def _call_model(self, code: str, findings: list[Finding]) -> str:
        findings_text = _format_findings_for_prompt(findings)

        user_prompt = (
            f"Here is the source code:\n\n```\n{code}\n```\n\n"
            f"The following vulnerabilities were found:\n\n{findings_text}\n\n"
            "Produce edits to fix ALL of these vulnerabilities."
        )

        messages = [
            {"role": "system", "content": FIXER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.complete(
                    model=self._model_id,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error("Fixer call failed after %d attempts: %s", self._max_retries, e)
                    return json.dumps({"edits": [], "error": str(e)})
                logger.warning("Attempt %d failed: %s — retrying in %.1fs", attempt + 1, e, self._retry_delay)
                time.sleep(self._retry_delay)

        return json.dumps({"edits": []})

    @staticmethod
    def _parse_edits(raw: str, file_path: str) -> list[CodeEdit]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse fixer JSON response")
            return []

        raw_edits = data.get("edits", [])
        edits: list[CodeEdit] = []
        for i, item in enumerate(raw_edits):
            try:
                edit = CodeEdit(
                    file_path=item.get("file_path", file_path),
                    original=item.get("original", ""),
                    replacement=item.get("replacement", ""),
                    finding_id=item.get("finding_id", f"F-{i + 1:03d}"),
                    rationale=item.get("rationale", ""),
                )
                if edit.original and edit.replacement and edit.original != edit.replacement:
                    edits.append(edit)
                else:
                    logger.warning("Skipping no-op or empty edit at index %d", i)
            except Exception as exc:
                logger.warning("Skipping malformed edit at index %d: %s", i, exc)
        return edits


class MockFixer:
    """Returns predefined edits for testing — no API calls."""

    def __init__(self, edits: list[CodeEdit] | None = None) -> None:
        self._edits = edits or []

    def fix(self, code: str, findings: list[Finding], file_path: str = "<stdin>") -> FixerOutput:
        return FixerOutput(
            edits=list(self._edits),
            raw_response="<mock-fixer>",
            model_id="mock",
        )


def apply_edits(code: str, edits: list[CodeEdit]) -> str:
    """Apply a list of CodeEdits to source code (in-memory, sequential)."""
    result = code
    for edit in edits:
        if edit.original in result:
            result = result.replace(edit.original, edit.replacement, 1)
        else:
            logger.warning(
                "Edit for finding %s could not be applied — original snippet not found in code",
                edit.finding_id,
            )
    return result

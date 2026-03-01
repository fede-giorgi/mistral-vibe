"""Security analyzer abstraction.

Provides a protocol for the analyzer and implementations:
- MockAnalyzer: returns fixture findings for development/testing.
- MistralAnalyzer: calls the Mistral API (base or fine-tuned model).
- HuggingFaceAnalyzer: calls a model on Hugging Face via the Inference API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Protocol

import httpx
from mistralai import Mistral

from ai_pipeline.remediation.models import AnalyzerOutput, Finding, Severity

logger = logging.getLogger(__name__)

HF_INFERENCE_URL = "https://router.huggingface.co/hf/models"

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
    def _strip_markdown_fences(text: str) -> str:
        """Remove ```json ... ``` wrapping if present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    @staticmethod
    def _parse_findings(raw: str, file_path: str) -> list[Finding]:
        cleaned = MistralAnalyzer._strip_markdown_fences(raw)

        # Try structured JSON first
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None

        if data is not None:
            if "findings" in data:
                raw_findings = data["findings"]
            elif "violation_type" in data:
                raw_findings = [data]
            else:
                raw_findings = [data] if data else []

            findings: list[Finding] = []
            for i, item in enumerate(raw_findings):
                try:
                    sev_raw = item.get("severity", "medium").lower()
                    if sev_raw == "critical":
                        sev_raw = "high"
                    finding = Finding(
                        id=item.get("id", f"F-{i + 1:03d}"),
                        file_path=item.get("file_path", file_path),
                        start_line=item.get("start_line"),
                        end_line=item.get("end_line"),
                        cwe=item.get("cwe") or item.get("violation_type"),
                        severity=Severity(sev_raw),
                        title=item.get("title") or item.get("violation_type", "Unknown vulnerability"),
                        explanation=item.get("explanation") or item.get("description") or item.get("risk", ""),
                        snippet=item.get("snippet"),
                    )
                    findings.append(finding)
                except Exception as e:
                    logger.warning("Skipping malformed finding at index %d: %s", i, e)
            return findings

        # Fallback: extract vulnerability mentions from free-text response
        return MistralAnalyzer._parse_freetext(raw, file_path)

    @staticmethod
    def _parse_freetext(raw: str, file_path: str) -> list[Finding]:
        """Extract findings from prose / free-text model output."""
        import re

        vuln_patterns: list[tuple[str, str, str]] = [
            (r"(?i)sql.?injection", "CWE-89", "SQL Injection"),
            (r"(?i)cross.?site.?scripting|XSS", "CWE-79", "Cross-Site Scripting (XSS)"),
            (r"(?i)command.?injection|os.?command", "CWE-78", "OS Command Injection"),
            (r"(?i)path.?traversal|directory.?traversal", "CWE-22", "Path Traversal"),
            (r"(?i)CSRF|cross.?site.?request.?forgery", "CWE-352", "Cross-Site Request Forgery"),
            (r"(?i)insecure.?deserialization", "CWE-502", "Insecure Deserialization"),
            (r"(?i)hard.?coded.?(?:password|credential|secret)", "CWE-798", "Hard-coded Credentials"),
            (r"(?i)(?:debug\s*=\s*True|debug.?mode)", "CWE-489", "Debug Mode Enabled"),
            (r"(?i)buffer.?overflow", "CWE-120", "Buffer Overflow"),
            (r"(?i)(?:insecure.?direct.?object|IDOR)", "CWE-639", "Insecure Direct Object Reference"),
        ]

        findings: list[Finding] = []
        seen_cwes: set[str] = set()
        for pattern, cwe, title in vuln_patterns:
            if re.search(pattern, raw) and cwe not in seen_cwes:
                seen_cwes.add(cwe)
                findings.append(Finding(
                    id=f"F-{len(findings) + 1:03d}",
                    file_path=file_path,
                    start_line=None,
                    end_line=None,
                    cwe=cwe,
                    severity=Severity.HIGH,
                    title=title,
                    explanation=raw[:1500],
                    snippet=None,
                ))

        return findings


class HuggingFaceAnalyzer:
    """Calls a fine-tuned model on Hugging Face via a dedicated Inference Endpoint.

    Supports both dedicated endpoints (OpenAI-compatible vLLM/TGI) and the
    serverless Inference API.  Set HF_ENDPOINT_URL for a dedicated endpoint,
    or fall back to the serverless router.
    """

    def __init__(
        self,
        model_id: str,
        token: str | None = None,
        endpoint_url: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        max_new_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id.strip()
        if "/" not in self._model_id:
            raise ValueError(
                "Hugging Face model_id must be 'username/repo-name' (e.g. myuser/my-security-model)"
            )
        self._token = (token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "")).strip()
        if not self._token:
            raise ValueError(
                "Hugging Face Inference API requires a token. Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
            )
        self._endpoint_url = (endpoint_url or os.getenv("HF_ENDPOINT_URL", "")).strip().rstrip("/")
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._max_new_tokens = max_new_tokens

    def analyze(self, code: str, file_path: str = "<stdin>") -> AnalyzerOutput:
        raw = self._call_model(code)
        findings = MistralAnalyzer._parse_findings(raw, file_path)
        return AnalyzerOutput(findings=findings, raw_response=raw, model_id=self._model_id)

    def _call_model(self, code: str) -> str:
        if self._endpoint_url:
            return self._call_endpoint(code)
        return self._call_serverless(code)

    def _call_endpoint(self, code: str) -> str:
        """Call a dedicated HF Inference Endpoint (OpenAI-compatible chat API)."""
        url = f"{self._endpoint_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "user", "content": f"analyze this code for security violations:\n\n{code}"},
            ],
            "max_tokens": self._max_new_tokens,
            "temperature": 0.0,
        }

        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                logger.debug("Endpoint raw response (first 500 chars): %s", content[:500])
                return content
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt == self._max_retries - 1:
                    logger.error("Endpoint call failed after %d attempts: %s", self._max_retries, err)
                    return json.dumps({"findings": [], "error": err})
                logger.warning("Attempt %d: %s — retrying in %.1fs", attempt + 1, err[:200], self._retry_delay)
                time.sleep(self._retry_delay)

        return json.dumps({"findings": []})

    def _call_serverless(self, code: str) -> str:
        """Fallback: call via HF serverless Inference API."""
        from huggingface_hub import InferenceClient

        prompt = (
            f"{ANALYZER_SYSTEM_PROMPT}\n\n"
            f"Analyze this code for security vulnerabilities:\n\n```\n{code}\n```"
        )

        hf_client = InferenceClient(model=self._model_id, token=self._token, timeout=120)

        for attempt in range(self._max_retries):
            try:
                result = hf_client.text_generation(
                    prompt,
                    max_new_tokens=self._max_new_tokens,
                    temperature=0.2,
                )
                return result.strip()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt == self._max_retries - 1:
                    logger.error("HF serverless failed after %d attempts: %s", self._max_retries, err)
                    return json.dumps({"findings": [], "error": err})
                logger.warning("Attempt %d: %s — retrying in %.1fs", attempt + 1, err[:200], self._retry_delay)
                time.sleep(self._retry_delay)

        return json.dumps({"findings": []})

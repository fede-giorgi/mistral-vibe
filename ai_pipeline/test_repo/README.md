# test_repo

Intentionally vulnerable code used to demonstrate the agentic remediation workflow.

## Demo File

**`vuln_demo.py`** — A concise login function with a classic SQL injection vulnerability.
The fine-tuned model (wardstral-8b) detects the issue, Devstral proposes fixes,
and the loop iterates until all findings are resolved.

## Run the Demo

```bash
# From project root, load environment variables:
set -a && source ai_pipeline/.env && set +a

# Run the full remediation loop (fine-tuned model + Devstral fixer):
uv run python ai_pipeline/7_remediation.py \
  --model ratnam1510/wardstral-8b \
  --max-iterations 3 \
  --output-json ai_pipeline/test_repo/demo_result.json \
  ai_pipeline/test_repo/vuln_demo.py
```

## What Happens

1. **wardstral-8b** (fine-tuned on security data, served via HF Inference Endpoint) analyzes the code
2. It detects SQL Injection and other security issues
3. **Devstral** (Mistral's coding model) proposes code fixes
4. The fixed code is re-analyzed by wardstral-8b
5. The loop repeats until all findings are resolved or max iterations reached

## Other Test Files

- `auth.py` — SQL injection, hardcoded secret, unsanitized output
- `main.py` — Command injection, template without escaping

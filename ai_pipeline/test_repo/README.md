# test_repo

Fake repo used to test the remediation workflow (Step 7).

**Files (intentionally vulnerable):**

- `auth.py` — SQL injection pattern, hardcoded secret, unsanitized output
- `main.py` — command injection, template without escaping

**Run remediation:**

```bash
# From project root, with env loaded:
set -a && source ai_pipeline/.env && set +a

# Mock (no API):
uv run python ai_pipeline/7_remediation.py --mock ai_pipeline/test_repo/auth.py

# W&B artifact:
uv run python ai_pipeline/7_remediation.py \
  --model ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0 \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  ai_pipeline/test_repo/auth.py
```

Replace with your real model/artifact if needed.

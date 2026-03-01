# Plugging your fine-tuned model into remediation

You can use either a **Hugging Face repo** or a **W&B artifact** (the artifact path you got from the run).

---

## Quick setup: Hugging Face (recommended — no local GPU)

1. **Put your tokens in `ai_pipeline/.env`:**
   ```bash
   HUGGINGFACE_HUB_TOKEN=hf_...
   MISTRAL_API_KEY=...
   ```

2. **Check setup and get your repo id:**
   ```bash
   uv run python ai_pipeline/setup_hf_remediation.py
   ```
   This checks your env and, if logged in, suggests your default repo id (e.g. `USERNAME/mistral-small-secure-scan`).

3. **Run remediation** (use your actual HF model repo id):
   ```bash
   uv run python ai_pipeline/7_remediation.py --model USERNAME/REPO_NAME path/to/file.py
   ```
   Example:
   ```bash
   uv run python ai_pipeline/7_remediation.py --model tommasoravasio/mistral-small-secure-scan ai_pipeline/test_repo/auth.py
   ```

   Or run the setup script with your model and `--test` to try it on the test repo:
   ```bash
   uv run python ai_pipeline/setup_hf_remediation.py --model USERNAME/REPO_NAME --test
   ```

If you don’t have a model on Hugging Face yet, either train one with `scripts/launch_finetune.py`, or **if you only have a W&B artifact**, push it to HF once with the script below, then use the HF analyzer.

### I only have a W&B artifact — push it to Hugging Face once

If your fine-tuned model lives only on W&B (e.g. `entity/project/security-scan-lora:v0`), you can merge it with the base model and push to a new HF repo. Then use that repo id with `--model` for remediation (no local GPU needed).

1. **Set in `ai_pipeline/.env`:** `WANDB_API_KEY`, `HUGGINGFACE_HUB_TOKEN`.

2. **Run the push script** (needs ~16GB RAM for a 7B model; if you OOM, run it on [Colab](https://colab.research.google.com) or another machine):
   ```bash
   uv run python ai_pipeline/push_wandb_to_hf.py \
     --wandb-artifact ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0 \
     --base-model mistralai/Mistral-7B-Instruct-v0.3 \
     --hf-repo YOUR_HF_USERNAME/security-scan-merged
   ```
   Replace `YOUR_HF_USERNAME` with your Hugging Face username and the W&B path with your artifact.

3. **Run remediation** with the new repo:
   ```bash
   uv run python ai_pipeline/7_remediation.py --model YOUR_HF_USERNAME/security-scan-merged path/to/file.py
   ```

---

## Option A: Use the W&B artifact path

If you have an artifact path like `ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0`:

1. **Install extra deps** (once):
   ```bash
   uv add wandb torch transformers peft
   ```

2. **Set env**: `HUGGINGFACE_HUB_TOKEN` (to load the base model), `WANDB_API_KEY` (to download the artifact). Optionally `MISTRAL_API_KEY` for the fixer.

3. **Run** (use the **exact** artifact path and the base model you fine-tuned from):
   ```bash
   uv run python ai_pipeline/7_remediation.py \
     --model ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0 \
     --base-model mistralai/Mistral-7B-Instruct-v0.3 \
     path/to/code.py
   ```
   Replace `mistralai/Mistral-7B-Instruct-v0.3` with the base model you used in training (e.g. from the job config or `scripts/launch_finetune.py`).

The first run will download the artifact and the base model; later runs use the cache.

### If you get MPS out-of-memory on Mac (e.g. 7B model)

- **Use 4-bit quantization (recommended)**  
  Install `mps-bitsandbytes` so the W&B analyzer loads the base model in 4-bit on Apple Silicon (~3.5 GB for 7B):
  ```bash
  uv add mps-bitsandbytes
  ```
  Then run as above; 4-bit is used by default on Mac.

- **Force CPU** (slower, no GPU memory limit):
  ```bash
  uv run python ai_pipeline/7_remediation.py --model ... --base-model ... --cpu path/to/code.py
  ```

- **Last resort** (can cause system instability): allow MPS to use more memory:
  ```bash
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python ai_pipeline/7_remediation.py ...
  ```

- **Avoid local inference**  
  Use the Hugging Face analyzer instead (Option B): push your adapter to a HF repo and run with `--model username/repo-name`. The analyzer then uses the HF Inference API and no local GPU is needed.

---

## Option B: Use the Hugging Face repo id

### Step 1: Get your Hugging Face model repo id

The repo id has the form **`username/repo-name`**.

- **If you used the default when launching the job** (e.g. `scripts/launch_finetune.py` without `--output-repo`):
  - Repo id is: **`YOUR_HF_USERNAME/mistral-small-secure-scan`**
  - Replace `YOUR_HF_USERNAME` with your Hugging Face username (see https://huggingface.co/settings or your profile URL).

- **If you passed a custom `--output-repo`** when launching:
  - Repo id is that value (e.g. `myuser/my-custom-scan`).

- **From the W&B run** (optional):
  - Open the W&B run page.
  - In the job logs or config, look for `OUTPUT_REPO` or the line:  
    `Once complete, model will be at: https://huggingface.co/...`
  - The repo id is the part after `huggingface.co/` (e.g. `tommaso/mistral-small-secure-scan`).

- **From Hugging Face**:
  - Go to https://huggingface.co/models and filter by "Your models".
  - Open the model you created for this fine-tune; the repo id is in the URL:  
    `https://huggingface.co/username/repo-name` → use **`username/repo-name`**.

---

### Step 2: Set environment variables

In `ai_pipeline/.env` (or export in the shell):

- **HUGGINGFACE_HUB_TOKEN** — your HF token (for the analyzer).
- **MISTRAL_API_KEY** — your Mistral key (for the Devstral fixer).

Load them before running (e.g. from project root):

```bash
set -a && source ai_pipeline/.env && set +a
```

---

### Step 3: Run remediation with your model

From the project root:

```bash
uv run python ai_pipeline/7_remediation.py --model YOUR_REPO_ID path/to/code.py
```

Example (replace with your repo id):

```bash
uv run python ai_pipeline/7_remediation.py --model tommasoravasio/mistral-small-secure-scan path/to/vulnerable.py
```

The script uses **Hugging Face Inference API** for the analyzer (your fine-tuned model) and **Mistral API** for the fixer (Devstral).

---

## Summary

| You have              | What to use in remediation      |
|-----------------------|---------------------------------|
| W&B artifact path    | `--model entity/project/artifact:v0 --base-model BASE` (install: wandb, torch, transformers, peft; on Mac add mps-bitsandbytes for 4-bit) |
| HF model repo id      | `--model username/repo-name`   |
| Default launch_finetune | HF repo: `username/mistral-small-secure-scan` |

**Mac MPS OOM:** use `--cpu` or install `mps-bitsandbytes` for 4-bit; or use the HF repo (Option B) so inference runs in the cloud.

"""W&B artifact analyzer — loads LoRA adapter from a W&B artifact and runs inference locally.

Requires: wandb, torch, transformers, peft. Install with:
  uv add wandb torch transformers peft
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ai_pipeline.remediation.analyzer import ANALYZER_SYSTEM_PROMPT
from ai_pipeline.remediation.models import AnalyzerOutput, Finding, Severity

logger = logging.getLogger(__name__)

def _normalize_no_split_module_classes(value):
    """Convert no_split_module_classes to list[str] for accelerate (handles set/list of sets)."""
    if value is None:
        return None
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            if isinstance(x, set):
                out.extend(x)
            elif isinstance(x, (list, tuple)):
                out.extend(x)
            else:
                out.append(str(x))
        return out
    return [str(value)]


def _patch_accelerate_get_balanced_memory() -> None:
    """Patch accelerate's get_balanced_memory AND peft's local reference to it.

    peft does `from accelerate.utils.modeling import get_balanced_memory` which
    creates a local binding. Patching only the accelerate module doesn't affect
    peft's already-captured reference. We must patch both.
    """
    try:
        import accelerate.utils.modeling as _acc_modeling

        _orig = _acc_modeling.get_balanced_memory

        def _patched(model, max_memory=None, no_split_module_classes=None, **kwargs):
            no_split_module_classes = _normalize_no_split_module_classes(no_split_module_classes)
            return _orig(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, **kwargs)

        _acc_modeling.get_balanced_memory = _patched

        # Also patch peft's local reference (if peft is already imported)
        try:
            import peft.peft_model as _peft_model
            if hasattr(_peft_model, "get_balanced_memory"):
                _peft_model.get_balanced_memory = _patched
        except ImportError:
            pass
    except Exception as e:
        logger.debug("Could not patch accelerate get_balanced_memory: %s", e)


def _ensure_deps() -> None:
    try:
        # Patch accelerate first
        _patch_accelerate_get_balanced_memory()
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import wandb  # noqa: F401
        # Re-patch after peft import to catch peft's local reference
        _patch_accelerate_get_balanced_memory()
    except ImportError as e:
        raise ImportError(
            "Using a W&B artifact requires: wandb, torch, transformers, peft. "
            "Install with: uv add wandb torch transformers peft"
        ) from e


def _download_artifact(artifact_path: str, cache_root: Path) -> Path:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    safe_name = artifact_path.replace("/", "_").replace(":", "_")
    cache_dir = cache_root / safe_name
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        logger.info("Using cached W&B artifact at %s", cache_dir)
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(cache_dir))
    return cache_dir


def _find_adapter_dir(root: Path) -> Path:
    """Artifact may be ./output with adapter at root or in checkpoint-* subdir."""
    if (root / "adapter_config.json").exists():
        return root
    checkpoints = list(root.glob("checkpoint-*"))
    def _ckpt_num(p: Path) -> int:
        parts = p.name.split("-")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    checkpoints.sort(key=_ckpt_num)
    for cp in reversed(checkpoints):
        if (cp / "adapter_config.json").exists():
            return cp
    return root


def _load_model(
    base_model_name: str,
    adapter_path: Path,
    token: str | None,
    *,
    use_cpu: bool = False,
    use_4bit: bool = True,
) -> tuple:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "")
    logger.info("Loading base model %s...", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    mps_available = (
        not use_cuda
        and use_cpu is False
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    # CUDA: full precision; Mac: CPU-only, or MPS 4-bit via mps-bitsandbytes, or full MPS (may OOM)
    if use_cuda:
        dtype = torch.bfloat16
        load_kwargs: dict = {
            "token": token or None,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    elif use_cpu:
        model = _load_base_cpu(base_model_name, token, torch.float32)
        model = model.to("cpu")
        logger.info("CPU-only mode: model on CPU (slower but avoids MPS OOM).")
    elif mps_available and use_4bit:
        # Apple Silicon: quantize to 4-bit on MPS (~3.5 GB for 7B) to avoid OOM
        try:
            from mps_bitsandbytes import BitsAndBytesConfig, quantize_model
        except ImportError:
            logger.warning(
                "mps-bitsandbytes not installed; falling back to full precision on CPU. "
                "Install with: uv add mps-bitsandbytes"
            )
            model = _load_base_cpu(base_model_name, token, torch.float32)
            model = model.to("cpu")
        else:
            load_kwargs = {
                "token": token or None,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = quantize_model(model, quantization_config=config, device="mps")
            logger.info("Quantized base model to 4-bit on MPS (Apple Silicon).")
    elif mps_available:
        # MPS without 4-bit: likely OOM for 7B
        model = _load_base_cpu(base_model_name, token, torch.float32)
        model = model.to("mps")
        logger.info("Moved model to MPS (Apple Silicon GPU).")
    else:
        model = _load_base_cpu(base_model_name, token, torch.float32)
        model = model.to("cpu")

    adapter_dir = _find_adapter_dir(adapter_path)
    logger.info("Loading LoRA adapter from %s...", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer


def _load_base_cpu(base_model_name: str, token: str | None, dtype) -> "torch.nn.Module":
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "token": token or None,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    return AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)


class WandbAnalyzer:
    """Load a LoRA adapter from a W&B artifact and run security analysis locally.

    Use the full artifact path, e.g.:
      ratnam1510-jpdz/mistral-vibe-security/security-scan-lora:v0

    You must also pass the base model name (same as used during fine-tuning), e.g.:
      mistralai/Mistral-7B-Instruct-v0.3
    """

    def __init__(
        self,
        artifact_path: str,
        base_model_name: str,
        token: str | None = None,
        cache_root: Path | None = None,
        *,
        use_cpu: bool = False,
        use_4bit: bool = True,
    ) -> None:
        _ensure_deps()
        self._artifact_path = artifact_path.strip()
        self._base_model_name = base_model_name.strip()
        self._token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN", "")
        self._cache_root = cache_root or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "wandb-remediation"
        self._use_cpu = use_cpu
        self._use_4bit = use_4bit
        self._model = None
        self._tokenizer = None

    def _get_model_and_tokenizer(self):
        if self._model is not None:
            return self._model, self._tokenizer
        adapter_path = _download_artifact(self._artifact_path, self._cache_root)
        self._model, self._tokenizer = _load_model(
            self._base_model_name,
            adapter_path,
            self._token,
            use_cpu=self._use_cpu,
            use_4bit=self._use_4bit,
        )
        return self._model, self._tokenizer

    def analyze(self, code: str, file_path: str = "<stdin>") -> AnalyzerOutput:
        import torch

        model, tokenizer = self._get_model_and_tokenizer()
        prompt = (
            f"{ANALYZER_SYSTEM_PROMPT}\n\n"
            f"Analyze this code for security vulnerabilities:\n\n```\n{code}\n```"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.2,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the new tokens
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        raw = generated.strip()
        findings = _parse_findings(raw, file_path)
        return AnalyzerOutput(
            findings=findings,
            raw_response=raw,
            model_id=f"wandb:{self._artifact_path}",
        )


def _parse_findings(raw: str, file_path: str) -> list[Finding]:
    """Parse JSON findings from model output (same schema as MistralAnalyzer)."""
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

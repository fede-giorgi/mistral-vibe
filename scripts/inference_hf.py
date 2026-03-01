# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers>=4.46.0,<4.48",
#     "peft>=0.13.0,<0.15",
#     "datasets",
#     "bitsandbytes",
#     "accelerate",
#     "huggingface_hub",
#     "sentencepiece",
# ]
# ///
"""
Run inference with the fine-tuned LoRA model on HF Jobs.

Loads base model + LoRA adapter, runs inference on the test set,
and uploads results as a JSON artifact to the HF dataset repo.

Usage (runs on HF Jobs, launched by launch_inference.py):
    Env vars: HF_TOKEN, DATASET_REPO, OUTPUT_REPO
"""

import json
import os
import sys
import time
import traceback

# Force unbuffered stdout so HF Jobs log API picks up prints immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=== inference_hf.py starting ===", flush=True)

try:
    import torch

    print(
        f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            flush=True,
        )

    from huggingface_hub import HfApi, login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset

    HF_TOKEN = os.environ["HF_TOKEN"]
    DATASET_REPO = os.environ["DATASET_REPO"]
    OUTPUT_REPO = os.environ["OUTPUT_REPO"]
    SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "20"))

    login(token=HF_TOKEN)

    # Load test data
    print(f"Loading test data from {DATASET_REPO}...", flush=True)
    dataset = load_dataset(DATASET_REPO, split="test")
    n_samples = min(SAMPLE_SIZE, len(dataset))
    print(f"  Loaded {len(dataset)} test examples, using {n_samples}", flush=True)

    # Quick sanity check on dataset structure
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}", flush=True)
    msgs = sample["messages"]
    print(f"  Messages type: {type(msgs)}, len: {len(msgs)}", flush=True)
    if msgs:
        print(
            f"  First message type: {type(msgs[0])}, keys: {list(msgs[0].keys()) if isinstance(msgs[0], dict) else 'N/A'}",
            flush=True,
        )

    # Load tokenizer from LoRA adapter repo
    print(f"Loading tokenizer from {OUTPUT_REPO}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_REPO, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("  Tokenizer loaded.", flush=True)

    # Load base model in 4-bit
    print("Loading base model in 4-bit quantization...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410",
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )
    print(f"  Base model loaded. Device: {model.device}", flush=True)

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {OUTPUT_REPO}...", flush=True)
    model = PeftModel.from_pretrained(model, OUTPUT_REPO, token=HF_TOKEN)
    model.eval()
    print("  LoRA adapter loaded. Model ready!", flush=True)

    # Run inference
    results = []
    for idx in range(n_samples):
        example = dataset[idx]
        messages = example["messages"]

        # Extract messages — handle both list-of-dicts and other formats
        user_msg = ""
        assistant_msg = ""
        system_msg = "You are a senior security engineer. Analyze the provided codebase snippet and output a detailed vulnerability explanation."
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.get("role", "")
            content = m["content"] if isinstance(m, dict) else m.get("content", "")
            if role == "user":
                user_msg = content
            elif role == "assistant":
                assistant_msg = content
            elif role == "system":
                system_msg = content

        if not user_msg:
            print(f"[{idx + 1}/{n_samples}] SKIP — no user message found", flush=True)
            continue

        # Build prompt
        chat_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(
            f"[{idx + 1}/{n_samples}] Generating (input_len={inputs['input_ids'].shape[1]})...",
            flush=True,
        )
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = (time.monotonic() - t0) * 1000

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(
            f"  Done ({latency:.0f}ms, {len(new_tokens)} tokens): {response[:80]}...",
            flush=True,
        )

        results.append({
            "index": idx,
            "code_snippet": user_msg[:1000],
            "ground_truth": assistant_msg[:1000],
            "model_response": response[:2000],
            "latency_ms": latency,
        })

    # Save results
    output_path = "/tmp/inference_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"\nInference complete. {len(results)} results saved to {output_path}",
        flush=True,
    )

    # Upload results to HF dataset repo
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="eval/inference_results.json",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"Results uploaded to {DATASET_REPO}/eval/inference_results.json", flush=True)
    print("=== DONE ===", flush=True)

except Exception:
    traceback.print_exc()
    sys.exit(1)

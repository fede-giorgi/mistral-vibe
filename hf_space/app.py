import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
ADAPTER_REPO = "ratnam1510/mistral-small-secure-scan"
MAX_NEW_TOKENS = 1024

# ---------------------------------------------------------------------------
# Load model + LoRA once at startup (weights stay on CPU until @spaces.GPU)
# ---------------------------------------------------------------------------
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

print("Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()
print("Model ready!", flush=True)


SYSTEM_PROMPT = (
    "You are an expert security vulnerability analyst. Analyze the provided code "
    "for security vulnerabilities. For each vulnerability found, provide:\n"
    "1. Vulnerability type (CWE category)\n"
    "2. Severity assessment\n"
    "3. Clear explanation of the issue\n"
    "4. Suggested fix\n"
    "Be specific and reference the actual code."
)


@spaces.GPU(duration=120)
def analyze_code(code: str, custom_prompt: str | None = None) -> str:
    """Analyze code for security vulnerabilities."""
    if not code.strip():
        return "Please provide code to analyze."

    system = (
        custom_prompt.strip()
        if custom_prompt and custom_prompt.strip()
        else SYSTEM_PROMPT
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Analyze the following code for security vulnerabilities:\n\n```\n{code}\n```",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][inputs.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
EXAMPLES = [
    [
        """import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()""",
        "",
    ],
    [
        """from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/run')
def run_command():
    cmd = request.args.get('cmd')
    result = os.popen(cmd).read()
    return result""",
        "",
    ],
    [
        """const express = require('express');
const app = express();

app.get('/redirect', (req, res) => {
    const url = req.query.url;
    res.redirect(url);
});""",
        "",
    ],
]

with gr.Blocks(title="Security Vulnerability Scanner", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Security Vulnerability Scanner\n"
        "Powered by **Ministral-8B + QLoRA** fine-tuned for security analysis.\n"
        "Paste your code below and get a vulnerability assessment."
    )

    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Textbox(
                label="Code to Analyze",
                placeholder="Paste your code here...",
                lines=15,
                max_lines=50,
            )
            custom_prompt = gr.Textbox(
                label="Custom System Prompt (optional)",
                placeholder="Leave empty for default security analysis prompt",
                lines=3,
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Vulnerability Analysis",
                lines=20,
                max_lines=50,
                show_copy_button=True,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[code_input, custom_prompt],
        label="Example Vulnerable Code Snippets",
    )

    submit_btn.click(
        fn=analyze_code, inputs=[code_input, custom_prompt], outputs=output
    )

demo.launch()

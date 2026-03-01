# Base vs Fine-tuned Model Comparison Report

## Setup
- **Base model:** Ministral-8B-Instruct-2410 (no LoRA)
- **Fine-tuned model:** Ministral-8B-Instruct-2410 + QLoRA (security-scan-lora:v0)
- **Judge:** GLM-4.5-air via OpenRouter
- **Base samples scored:** 20/20
- **Fine-tuned samples scored:** 16/20
- **Same 20 test examples used for both models**

## Score Comparison (1-5 scale)

| Dimension | Base Avg | Fine-tuned Avg | Delta | Winner |
|-----------|----------|----------------|-------|--------|
| Overall | 1.55 | 1.81 | +0.26 | Fine-tuned |
| Vulnerability Identification | 1.9 | 2.31 | +0.41 | Fine-tuned |
| Severity Accuracy | 1.6 | 2.62 | +1.02 | Fine-tuned |
| Explanation Quality | 2.05 | 1.94 | -0.11 | Base |
| Fix Suggestion | 1.4 | 1.44 | +0.04 | Fine-tuned |
| Relevance | 1.5 | 1.25 | -0.25 | Base |

## Quality Distribution

| Metric | Base | Fine-tuned |
|--------|------|------------|
| % Good (>=4) | 0.0% | 6.2% |
| % Poor (<=2) | 90.0% | 81.2% |

## Per-Example Side-by-Side (first 16 matched pairs)

| # | Base Score | FT Score | Base Reasoning | FT Reasoning |
|---|-----------|----------|----------------|--------------|
| 1 | 1/5 | 2/5 | The candidate response analyzes PHP code for XSS vulnerabilities, completely mis | The candidate response provides general information about XSS vulnerabilities bu |
| 2 | 3/5 | 1/5 | The candidate correctly identifies XSS as a potential vulnerability but fails to | The candidate response is completely irrelevant to the code under review. Instea |
| 3 | 2/5 | 2/5 | The candidate provides a general security analysis but fails to identify the spe | The candidate correctly identifies a vulnerability related to downloading extern |
| 4 | 1/5 | 1/5 | The candidate analyzed a completely unrelated C function for NIC existence check | The candidate response completely missed the actual vulnerability (CWE-862 Missi |
| 5 | 2/5 | 2/5 | The candidate correctly identifies command injection as a vulnerability class bu | The candidate correctly identifies CWE-78 (command injection) but misattributes  |
| 6 | 1/5 | 1/5 | The candidate response completely misses the actual vulnerability (CWE-532) iden | The candidate response analyzes the wrong function entirely, failing to identify |
| 7 | 1/5 | 2/5 | The candidate response completely misses the actual vulnerability (CWE-346 relat | The candidate identifies the general vulnerability categories but fails to conne |
| 8 | 1/5 | 1/5 | The candidate response analyzes a C++ code snippet related to browser signin, co | The candidate response fails to identify any vulnerability in the code. It simpl |
| 9 | 2/5 | 1/5 | The candidate response fails to identify the actual integer overflow vulnerabili | The candidate response is completely irrelevant to the code under review. It ide |
| 10 | 3/5 | 2/5 | The candidate correctly identifies RCE through unsafe dynamic imports but doesn' | The candidate correctly identifies a real RCE vulnerability in Vite's RSC plugin |
| 11 | 2/5 | 1/5 | The candidate correctly identified SQL injection as a vulnerability class but ap | The candidate response shows a diff of the code against itself with no actual ch |
| 12 | 2/5 | 3/5 | The candidate identifies a potential security issue related to egress traffic bu | The candidate correctly identifies the vulnerability class (misconfigured toGrou |
| 13 | 1/5 | 2/5 | The candidate response completely fails to identify the actual vulnerability (CW | The candidate correctly identifies the vulnerability class (CWE-284) and provide |
| 14 | 2/5 | 1/5 | The candidate correctly identifies Stored XSS as a potential vulnerability, matc | The candidate response is a placeholder test function with no actual vulnerabili |
| 15 | 1/5 | 4/5 | The candidate completely missed the actual CWE-22 path traversal vulnerability i | The candidate correctly identified the path traversal vulnerability (CWE-22) and |
| 16 | 1/5 | 3/5 | The candidate response completely misses the actual CWE-284 vulnerability in Ope | The candidate correctly identifies the vulnerability class (CWE-284 related to t |

## Verdict

- **Base model average:** 1.55/5
- **Fine-tuned model average:** 1.81/5
- **Delta:** + 0.26

The fine-tuning improved the model by **0.26 points** on average.
However, both models still perform poorly (below 2/5), suggesting fundamental dataset issues.

## Key Observations

1. Both models score very poorly (avg ~1.5-1.8/5) — neither base nor fine-tuned can reliably analyze security vulnerabilities
2. The fine-tuned model shows marginal improvement in vulnerability identification (+0.41) and severity accuracy (+1.02)
3. Both models frequently analyze wrong/unrelated code instead of the provided snippet
4. The dataset has known issues: mismatched code/vulnerability pairs, extreme duplication
5. Fine-tuning with a broken dataset cannot fix fundamental data quality problems

## Recommended Next Steps

1. **Fix dataset quality** — ensure code snippets match their vulnerability descriptions
2. **Deduplicate** — remove the ~500 duplicate examples
3. **Standardize output format** — pick one format (structured JSON recommended)
4. **Re-train** with cleaned dataset
5. **Re-evaluate** with the same judge pipeline
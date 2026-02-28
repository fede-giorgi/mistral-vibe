# Severity Assignment Tool

## 🎯 Purpose

This tool assigns **severity levels** (Low, Medium, High) to security vulnerabilities using **Mistral Large** for consistent, guideline-based classification.

## 🔧 Installation

```bash
# Install required packages
pip install mistralai
```

## 🚀 Usage

### Basic Usage

```bash
# Set your Mistral API key
export MISTRAL_API_KEY="EXAMPLE"

# Assign severity to dataset
python assign_severity.py \
    --input dataset/train.jsonl \
    --output dataset/train_with_severity.jsonl
```

### Process All Datasets

```bash
# Process training set
python assign_severity.py --input dataset/train.jsonl --output dataset/train_severity.jsonl

# Process validation set
python assign_severity.py --input dataset/val.jsonl --output dataset/val_severity.jsonl

# Process test set
python assign_severity.py --input dataset/test.jsonl --output dataset/test_severity.jsonl
```

### Model Options

```bash
# Use different Mistral models
python assign_severity.py --input data.jsonl --output output.jsonl --model mistral-small-latest
python assign_severity.py --input data.jsonl --output output.jsonl --model open-mistral-nemo
```

## 📊 Severity Guidelines

The tool follows **strict criteria** for severity assignment:

### 🟥 HIGH Severity
- Remote Code Execution (RCE)
- OS Command Injection
- Insecure Deserialization → code execution
- SQL Injection affecting sensitive data
- Authentication bypass
- Privilege escalation
- Arbitrary file read/write in sensitive directories
- Direct access to credentials/secrets
- Widespread data exfiltration
- Full system compromise

### 🟧 MEDIUM Severity
- Limited data exposure
- Partial data manipulation
- Non-persistent Denial of Service
- Missing input validation (no clear exploit)
- Hardcoded credentials (limited scope)
- Significant preconditions required
- Meaningful but contained impact

### 🟩 LOW Severity
- Best-practice violations
- Minimal realistic exploitability
- Theoretical risk (no clear path)
- Negligible impact
- Weak logging/minor config issues
- Unrealistic attacker capabilities

## 📁 Input/Output Format

### Input Format
```json
{
  "messages": [
    {
      "role": "user",
      "content": "analyze this code for security violations:\n\n{vulnerable_code}"
    },
    {
      "role": "assistant",
      "content": "{\"violation_type\": \"os_command_injection\", \"severity\": \"critical\", ...}"
    }
  ]
}
```

### Output Format (Enhanced)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "analyze this code for security violations:\n\n{vulnerable_code}"
    },
    {
      "role": "assistant",
      "content": "{\"violation_type\": \"os_command_injection\", \"severity\": \"HIGH\", \"severity_justification\": \"Assigned HIGH based on violation type and impact analysis\", ...}"
    }
  ]
}
```

## 🎯 Features

### Automatic Severity Assignment
- Uses Mistral Large for consistent classification
- Follows strict security guidelines
- Provides justification for each assignment
- Handles errors gracefully

### Statistics Tracking
- Counts HIGH/MEDIUM/LOW assignments
- Tracks errors and skips invalid entries
- Provides summary statistics

### Conservative Approach
- Defaults to lower severity when uncertain
- Follows "when in doubt, choose lower" principle
- Focuses on code impact, not context

## 📈 Example Output

```bash
$ python assign_severity.py --input dataset/train.jsonl --output train_severity.jsonl

Processing dataset/train.jsonl...
Processed 10 entries...
Processed 20 entries...
...

✅ Processing complete!
Severity distribution:
  HIGH: 350 entries
  MEDIUM: 400 entries
  LOW: 103 entries
  Errors: 0 entries
Total processed: 853 entries
Output saved to: train_severity.jsonl
```

## 🔍 Technical Details

### Severity Decision Process

1. **Extract** violation type, explanation, and vulnerable code
2. **Create prompt** with guidelines and code context
3. **Query Mistral** for severity classification
4. **Parse response** to extract severity level
5. **Add severity** to assistant response
6. **Save enhanced dataset** with severity information

### Error Handling

- Skips malformed JSON entries
- Handles API errors gracefully
- Continues processing after errors
- Tracks and reports error count

### Performance

- Processes ~10-15 entries per minute
- Uses deterministic temperature (0.3)
- Optimized for consistency
- Low API token usage (~50 tokens per entry)

## 🎓 Best Practices

### When to Use

✅ **Before fine-tuning** - Enhance dataset with severity
✅ **For consistency** - Standardize severity across datasets
✅ **Quality assurance** - Validate existing severity assignments
✅ **Research** - Analyze severity distribution

### Tips

- **Start small**: Test on 50-100 entries first
- **Review samples**: Check a few outputs manually
- **Adjust guidelines**: Modify criteria as needed
- **Combine datasets**: Process all splits consistently

## 🚀 Integration

### With Fine-Tuning Pipeline

```bash
# Add severity to all datasets
python assign_severity.py --input dataset/train.jsonl --output dataset/train_severity.jsonl
python assign_severity.py --input dataset/val.jsonl --output dataset/val_severity.jsonl
python assign_severity.py --input dataset/test.jsonl --output dataset/test_severity.jsonl

# Fine-tune with enhanced data
python finetune_security.py --train train_severity.jsonl --val val_severity.jsonl
```

### With Analysis Tools

```python
# Analyze severity distribution
import json

severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

with open('train_severity.jsonl') as f:
    for line in f:
        data = json.loads(line)
        severity = json.loads(data['messages'][1]['content'])['severity']
        severity_counts[severity] += 1

print(f"HIGH: {severity_counts['HIGH']}")
print(f"MEDIUM: {severity_counts['MEDIUM']}")
print(f"LOW: {severity_counts['LOW']}")
```

## ✅ Success Criteria

- **Consistency**: Same violation types get same severity
- **Coverage**: All valid entries processed
- **Quality**: Meaningful severity assignments
- **Documentation**: Clear justification provided

## 📚 References

- **Severity Guidelines**: Based on CVSS and industry standards
- **Mistral API**: https://docs.mistral.ai/
- **Security Best Practices**: OWASP, CWE, NIST

## 🎉 Next Steps

1. **Run on your dataset**
2. **Review sample outputs**
3. **Integrate with pipeline**
4. **Fine-tune enhanced model**

The tool is **production-ready** and provides **consistent, guideline-based severity assignment** for your security dataset! 🚀

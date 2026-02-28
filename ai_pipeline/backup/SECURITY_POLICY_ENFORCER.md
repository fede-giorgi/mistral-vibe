# Secure Code Policy Enforcer

## Overview
A tool that reads PR diffs and enforces security policies by flagging violations, explaining risks, and suggesting compliant patches.

## Example Policies
- No hardcoded secrets
- No raw SQL without parameterization
- No unsafe deserialization
- No use of `eval()`
- Enforce logging standards
- Enforce specific architecture rules

## Why This Approach
Base models are inconsistent at policy enforcement, miss edge cases, and fail at structured policy reasoning. Fine-tuning on code snippets, violations, and correct classifications will improve accuracy.

## Implementation Plan
1. **Data Collection**: Gather code snippets with violations and correct classifications
2. **Fine-Tuning**: Train model on structured policy reasoning
3. **Integration**: Add CLI command to check PR diffs
4. **Reporting**: Generate structured violation reports with explanations and suggested fixes

## Future Development
- Implement the tool as a CLI command
- Integrate with CI/CD pipelines
- Expand policy coverage
- Improve explanation and suggestion quality
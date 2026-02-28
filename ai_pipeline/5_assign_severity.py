#!/usr/bin/env python3
"""
Assign severity levels (Low, Medium, High) to security vulnerabilities
using Mistral's analysis capabilities and structured guidelines.
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from mistralai import Mistral

# Severity guidelines for Mistral to follow
SEVERITY_GUIDELINES = """
Assign severity levels based on these strict criteria:

🟥 HIGH: Remote Code Execution, OS Command Injection, Insecure Deserialization leading to code execution, SQL Injection affecting sensitive data, Authentication bypass, Privilege escalation, Arbitrary file read/write in sensitive directories, Direct access to credentials/secrets, Widespread data exfiltration, Full system compromise

🟧 MEDIUM: Limited data exposure, Partial data manipulation without full compromise, Non-persistent Denial of Service, Missing input validation without clearly exploitable sink, Hardcoded credentials with limited scope, Significant preconditions required, Meaningful but contained impact

🟩 LOW: Best-practice violations, Minimal realistic exploitability, Theoretical risk without clear exploitation path, Negligible impact, Weak logging/minor configuration issues/non-sensitive exposure, Unrealistic attacker capabilities required

⚖️ Decision Rules:
1. If vulnerability enables code execution or full compromise → HIGH
2. If impact unclear but dangerous sink exists (e.g., command execution) → HIGH
3. If exploitation requires multiple unrealistic assumptions → MEDIUM/LOW
4. If unsure between levels → choose LOWER unless impact clearly justifies escalation
5. Do NOT use contextual factors outside provided code (assume standard production environment)

Analyze ONLY the provided code snippet and violation description. Be conservative in severity assignment.
"""

class SeverityAssigner:
    """Assign severity levels to security vulnerabilities using Mistral API."""

    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """Initialize the severity assigner."""
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "ERROR": 0}

    def assign_severity(self, vulnerable_code: str, violation_type: str, explanation: str) -> Optional[str]:
        """Assign severity level to a single vulnerability."""
        try:
            # Create prompt for Mistral
            prompt = f"""Given this security vulnerability:

Violation Type: {violation_type}
Explanation: {explanation}
Vulnerable Code:
{vulnerable_code[:2000]}  # Truncated for context

{SEVERITY_GUIDELINES}

Assign severity level (HIGH, MEDIUM, or LOW) and provide brief justification:

Severity: [HIGH/MEDIUM/LOW]
Justification: [brief reason based on guidelines above]"""

            # Get completion from Mistral
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Be deterministic
                max_tokens=50
            )

            # Parse response
            severity = self._parse_response(response.choices[0].message.content)

            if severity in ["HIGH", "MEDIUM", "LOW"]:
                self.stats[severity] += 1
                return severity
            else:
                self.stats["ERROR"] += 1
                return None

        except Exception as e:
            print(f"Error assigning severity: {e}")
            self.stats["ERROR"] += 1
            return None

    def _parse_response(self, response: str) -> str:
        """Parse severity level from Mistral response."""
        # Look for severity in response
        response = response.upper()
        if "HIGH" in response:
            return "HIGH"
        elif "MEDIUM" in response:
            return "MEDIUM"
        elif "LOW" in response:
            return "LOW"
        else:
            return "UNKNOWN"

    def process_dataset(self, input_file: str, output_file: str) -> None:
        """Process entire dataset and add severity levels."""
        print(f"Processing {input_file}...")

        processed_entries = []

        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)

                    # Extract needed information
                    messages = entry.get('messages', [])
                    if len(messages) != 2:
                        print(f"Skipping entry {i+1}: invalid format")
                        continue

                    user_msg = messages[0]['content']
                    assistant_msg = messages[1]['content']

                    # Parse assistant response
                    try:
                        assistant_data = json.loads(assistant_msg)
                        violation_type = assistant_data.get('violation_type', 'unknown')
                        explanation = assistant_data.get('explanation', '')
                        vulnerable_code = user_msg.replace('analyze this code for security violations:\n\n', '')
                    except:
                        print(f"Skipping entry {i+1}: invalid JSON format")
                        continue

                    # Assign severity
                    severity = self.assign_severity(vulnerable_code, violation_type, explanation)

                    if severity:
                        # Add severity to assistant response
                        assistant_data['severity'] = severity
                        assistant_data['severity_justification'] = f"Assigned {severity} based on violation type and impact analysis"

                        # Update assistant message
                        messages[1]['content'] = json.dumps(assistant_data)

                        processed_entries.append({
                            "messages": messages,
                            "original_file": input_file,
                            "entry_number": i + 1
                        })

                        if (i + 1) % 10 == 0:
                            print(f"Processed {i + 1} entries...")

                except Exception as e:
                    print(f"Error processing entry {i+1}: {e}")
                    continue

        # Save processed dataset
        with open(output_file, 'w') as f:
            for entry in processed_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"\n✅ Processing complete!")
        print(f"Severity distribution:")
        for level, count in self.stats.items():
            if level != "ERROR":
                print(f"  {level}: {count} entries")
        print(f"  Errors: {self.stats['ERROR']} entries")
        print(f"Total processed: {len(processed_entries)} entries")
        print(f"Output saved to: {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Assign severity levels to security vulnerabilities")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL dataset file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file with severity")
    parser.add_argument("--model", type=str, default="mistral-large-latest",
                       help="Mistral model to use (default: mistral-large-latest)")

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is required")
        print("Export it first: export MISTRAL_API_KEY='your_key'")
        return

    # Create assigner and process dataset
    assigner = SeverityAssigner(api_key=api_key, model=args.model)
    assigner.process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()

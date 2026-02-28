#!/usr/bin/env python3
"""
Convert SARD dataset to fine-tuning format.
Creates structured JSON with violation details, explanations, and fixes.
"""

import os
import json
import pandas as pd
from pathlib import Path

# CWE to violation type mapping
CWE_TO_VIOLATION = {
    "CWE-78": {
        "violation_type": "os_command_injection",
        "severity": "critical",
        "explanation": "This code allows untrusted input to be executed as operating system commands. Attackers can exploit this to execute arbitrary commands on the server with the application's privileges.",
        "risk": "Complete system compromise, arbitrary command execution, data theft, system modification"
    },
    "CWE-89": {
        "violation_type": "sql_injection",
        "severity": "critical",
        "explanation": "This code constructs SQL queries by concatenating user input directly into the query string. Attackers can manipulate input to execute malicious SQL commands.",
        "risk": "Database compromise, data theft, data manipulation, authentication bypass"
    },
    "CWE-190": {
        "violation_type": "integer_overflow",
        "severity": "high",
        "explanation": "This code performs arithmetic operations that can exceed the maximum size of the data type. This can lead to unexpected behavior, crashes, or security vulnerabilities.",
        "risk": "Memory corruption, arbitrary code execution, denial of service, security bypass"
    },
    "CWE-319": {
        "violation_type": "cleartext_transmission",
        "severity": "high",
        "explanation": "This code transmits sensitive information without encryption. Attackers can intercept and read confidential data sent over the network.",
        "risk": "Data interception, credential theft, privacy violations, compliance violations"
    },
    "CWE-416": {
        "violation_type": "use_after_free",
        "severity": "critical",
        "explanation": "This code uses memory after it has been freed. This can lead to crashes, data corruption, or arbitrary code execution if an attacker can control the freed memory.",
        "risk": "Memory corruption, arbitrary code execution, denial of service, privilege escalation"
    },
    "CWE-20": {
        "violation_type": "input_validation",
        "severity": "medium",
        "explanation": "This code does not properly validate user input before processing it. Attackers can provide malformed or unexpected input to cause errors or exploit vulnerabilities.",
        "risk": "Application crashes, unexpected behavior, potential exploitation of other vulnerabilities"
    },
    "CWE-125": {
        "violation_type": "out_of_bounds_read",
        "severity": "high",
        "explanation": "This code reads memory outside the allocated bounds of an array or buffer. This can cause crashes or expose sensitive memory contents to attackers.",
        "risk": "Memory disclosure, application crashes, information leakage, potential code execution"
    }
}

# Generic compliant patches for each violation type
COMPLIANT_PATCHES = {
    "os_command_injection": """
// FIXED: Use parameterized commands or input validation
// Instead of: system(user_input)
// Use: validated_input = validate_user_input(user_input);
//       system(escape_shell_command(validated_input));

// Or better, use library functions that handle arguments safely:
// execvp() with proper argument array
// or use higher-level APIs that avoid shell entirely
""",
    "sql_injection": """
// FIXED: Use prepared statements with parameterized queries
// Instead of: query = "SELECT * FROM users WHERE name = '" + user_input + "'"
// Use: query = "SELECT * FROM users WHERE name = ?"
//       stmt = connection.prepareStatement(query);
//       stmt.setString(1, user_input);
//       result = stmt.executeQuery();
""",
    "integer_overflow": """
// FIXED: Add bounds checking before arithmetic operations
// Instead of: result = a + b;
// Use: if (b > INT_MAX - a) { handle_error(); }
//       result = a + b;

// Or use safe arithmetic functions:
// if (!safe_add(a, b, &result)) { handle_error(); }
""",
    "cleartext_transmission": """
// FIXED: Use encryption for sensitive data transmission
// Instead of: send_data_in_cleartext(data);
// Use: encrypted_data = encrypt_data(data, encryption_key);
//       send_encrypted_data(encrypted_data);

// Use TLS/SSL for network communication
// Ensure all connections use HTTPS/SSL/TLS
""",
    "use_after_free": """
// FIXED: Nullify pointers after free and check before use
// Instead of: free(ptr); ... use(ptr);
// Use: free(ptr); ptr = NULL; ... if (ptr) use(ptr);

// Better: restructure code to avoid dangling pointers
// Use smart pointers or automatic memory management
""",
    "input_validation": """
// FIXED: Validate all user input before processing
// Instead of: process_user_input(raw_input);
// Use: if (is_valid_input(raw_input)) {
//         sanitized_input = sanitize_input(raw_input);
//         process_user_input(sanitized_input);
//       } else {
//         reject_invalid_input();
//       }
""",
    "out_of_bounds_read": """
// FIXED: Add bounds checking before array access
// Instead of: value = array[index];
// Use: if (index >= 0 && index < array_size) {
//         value = array[index];
//       } else {
//         handle_error();
//       }

// Or use safe library functions that check bounds
"""
}

def read_file_content(file_path):
    """Read file content with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def create_finetune_entry(filename, cwe, testcase_id):
    """Create a single fine-tuning entry."""
    file_path = os.path.join("sard_c", "code", filename)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    # Get violation details from CWE mapping
    if cwe not in CWE_TO_VIOLATION:
        print(f"Warning: Unknown CWE: {cwe}")
        return None
    
    violation_info = CWE_TO_VIOLATION[cwe]
    
    # Read vulnerable code
    vulnerable_code = read_file_content(file_path)
    if not vulnerable_code:
        return None
    
    # Get compliant patch for this violation type
    compliant_patch = COMPLIANT_PATCHES.get(violation_info["violation_type"], "// TODO: Implement secure alternative")
    
    return {
        "violation_type": violation_info["violation_type"],
        "severity": violation_info["severity"],
        "vulnerable_code": vulnerable_code,
        "explanation": violation_info["explanation"],
        "risk": violation_info["risk"],
        "compliant_patch": compliant_patch,
        "source": f"SARD-{testcase_id}",
        "cwe": cwe
    }

def main():
    """Main conversion function."""
    print("Converting SARD dataset to fine-tuning format...")
    
    # Read labels CSV
    labels_file = "sard_c/labels.csv"
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found: {labels_file}")
        return
    
    try:
        df = pd.read_csv(labels_file)
        print(f"Loaded {len(df)} entries from labels.csv")
    except Exception as e:
        print(f"Error reading labels CSV: {e}")
        return
    
    # Convert each entry
    finetune_data = []
    
    for index, row in df.iterrows():
        entry = create_finetune_entry(row['filename'], row['cwe'], row['testcase_id'])
        if entry:
            finetune_data.append(entry)
            print(f"✅ Converted: {row['filename']} ({row['cwe']}) -> {entry['violation_type']}")
        else:
            print(f"❌ Skipped: {row['filename']} ({row['cwe']})")
    
    # Save as JSON
    output_file = "ai_pipeline/finetune_dataset.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(finetune_data, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Success! Saved {len(finetune_data)} entries to {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
    
    # Save as JSONL (one entry per line)
    output_file_jsonl = "ai_pipeline/finetune_dataset.jsonl"
    try:
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for entry in finetune_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"🎉 Success! Saved {len(finetune_data)} entries to {output_file_jsonl}")
    except Exception as e:
        print(f"Error saving JSONL file: {e}")
    
    # Generate statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total entries: {len(finetune_data)}")
    
    # Count by violation type
    violation_counts = {}
    for entry in finetune_data:
        vtype = entry['violation_type']
        violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
    
    print(f"\nViolation Type Distribution:")
    for vtype, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vtype}: {count} samples")
    
    print(f"\n📁 Files Created:")
    print(f"  {output_file} - Full JSON dataset")
    print(f"  {output_file_jsonl} - JSONL format (one per line)")
    print(f"\n✅ Dataset is ready for fine-tuning!")

if __name__ == "__main__":
    main()
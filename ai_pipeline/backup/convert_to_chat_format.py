#!/usr/bin/env python3
"""
Convert fine-tuning dataset to chat format.
Creates structured conversations with user/assistant turns.
"""

import os
import json
import pandas as pd

def create_chat_entry(entry):
    """Convert a single entry to chat format."""
    vulnerable_code = entry['vulnerable_code']
    
    # Create structured JSON output for assistant
    structured_output = {
        "violation_type": entry['violation_type'],
        "severity": entry['severity'],
        "explanation": entry['explanation'],
        "risk": entry['risk'],
        "compliant_patch": entry['compliant_patch'],
        "source": entry['source'],
        "cwe": entry['cwe']
    }
    
    # Format as pretty JSON for better readability
    structured_json = json.dumps(structured_output, indent=2)
    
    return {
        "messages": [
            {
                "role": "user",
                "content": f"analyze this code for security violations:\n\n{vulnerable_code}"
            },
            {
                "role": "assistant",
                "content": structured_json
            }
        ]
    }

def main():
    """Main conversion function."""
    print("Converting dataset to chat format...")
    
    # Load the existing fine-tune dataset
    input_file = "ai_pipeline/finetune_dataset.json"
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} entries from {input_file}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Convert each entry to chat format
    chat_dataset = []
    
    for i, entry in enumerate(dataset):
        chat_entry = create_chat_entry(entry)
        chat_dataset.append(chat_entry)
        
        if (i + 1) % 10 == 0:
            print(f"✅ Processed {i + 1}/{len(dataset)} entries")
    
    # Save as JSON
    output_file = "ai_pipeline/chat_dataset.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chat_dataset, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Success! Saved {len(chat_dataset)} chat entries to {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
    
    # Save as JSONL (one conversation per line)
    output_file_jsonl = "ai_pipeline/chat_dataset.jsonl"
    try:
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for entry in chat_dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"🎉 Success! Saved {len(chat_dataset)} chat entries to {output_file_jsonl}")
    except Exception as e:
        print(f"Error saving JSONL file: {e}")
    
    # Show a sample
    print(f"\n📝 Sample Entry:")
    sample = chat_dataset[0]
    print(f"User: {sample['messages'][0]['content'][:100]}...")
    print(f"Assistant: {sample['messages'][1]['content'][:100]}...")
    
    # Generate statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total conversations: {len(chat_dataset)}")
    print(f"Format: User asks for analysis → Assistant provides structured JSON response")
    
    print(f"\n📁 Files Created:")
    print(f"  {output_file} - Full JSON dataset")
    print(f"  {output_file_jsonl} - JSONL format (one conversation per line)")
    print(f"\n✅ Dataset is ready for chat fine-tuning!")

if __name__ == "__main__":
    main()
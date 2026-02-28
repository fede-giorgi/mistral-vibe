#!/usr/bin/env python3
"""
Download SARD C test cases with balanced CWE distribution.
Targets approximately 100 total observations with even distribution.
"""

import requests
import os
import pandas as pd
import zipfile
import io
import re
from collections import defaultdict

# SARD API configuration - Expanded CWE list
API_URL = "https://samate.nist.gov/SARD/api/test-cases/search?language%5B%5D=c"
API_URL += "&flaw%5B%5D=CWE-798"  # Hard-coded credentials
API_URL += "&flaw%5B%5D=CWE-89"    # SQL injection
API_URL += "&flaw%5B%5D=CWE-94"    # Code injection
API_URL += "&flaw%5B%5D=CWE-502"   # Deserialization of untrusted data
API_URL += "&flaw%5B%5D=CWE-20"    # Input validation
API_URL += "&flaw%5B%5D=CWE-78"    # OS command injection
API_URL += "&flaw%5B%5D=CWE-319"   # Cleartext transmission
API_URL += "&flaw%5B%5D=CWE-125"   # Out-of-bounds read
API_URL += "&flaw%5B%5D=CWE-416"   # Use after free
API_URL += "&flaw%5B%5D=CWE-190"   # Integer overflow
API_URL += "&page=1&limit=25"

# Configuration for balanced download
TARGET_TOTAL = 100  # Target total observations
CWE_TARGETS = {
    "CWE-798": 10,  # Hard-coded credentials
    "CWE-89": 10,    # SQL injection
    "CWE-94": 10,    # Code injection
    "CWE-502": 10,   # Deserialization
    "CWE-20": 10,    # Input validation
    "CWE-78": 10,    # OS command injection
    "CWE-319": 10,   # Cleartext transmission
    "CWE-125": 10,   # Out-of-bounds read
    "CWE-416": 10,   # Use after free
    "CWE-190": 10,   # Integer overflow
}

# Create output directory
os.makedirs("sard_c/code", exist_ok=True)

def extract_c_files_from_zip(zip_content, tc_id, cwe):
    """Extract .c files from ZIP content and save them."""
    extracted_files = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.c'):
                    # Extract the file
                    with zip_ref.open(file_info) as source_file:
                        content = source_file.read()
                        
                        # Create a safe filename
                        filename = f"{tc_id}_{os.path.basename(file_info.filename)}"
                        file_path = os.path.join("sard_c/code", filename)
                        
                        with open(file_path, "wb") as f:
                            f.write(content)
                        
                        extracted_files.append({
                            "filename": filename,
                            "cwe": cwe,
                            "testcase_id": tc_id
                        })
    except Exception as e:
        print(f"Error extracting ZIP for test case {tc_id}: {e}")
    
    return extracted_files

def get_cwe_from_test_case(tc):
    """Extract CWE information from test case data."""
    # Try to get CWE from SARIF data first
    sarif_data = tc.get("sarif", {})
    runs = sarif_data.get("runs", [])
    
    for run in runs:
        results = run.get("results", [])
        for result in results:
            taxa = result.get("taxa", [])
            for taxon in taxa:
                if taxon.get("id"):
                    return f"CWE-{taxon['id']}"
    
    # Try to get CWE from properties/description
    if runs:
        properties = runs[0].get("properties", {})
        description = properties.get("description", "")
        if "CWE:" in description:
            match = re.search(r'CWE:\s*(\d+)', description)
            if match:
                return f"CWE-{match.group(1)}"
    
    return None

def main():
    """Main download function with balanced CWE distribution."""
    print("Starting balanced SARD C test case download...")
    print(f"Target: {TARGET_TOTAL} total observations (~{TARGET_TOTAL//len(CWE_TARGETS)} per CWE)")
    
    page = 1
    rows = []
    cwe_counts = defaultdict(int)
    total_downloaded = 0
    
    # Track which CWEs have reached their targets
    cwe_targets_met = set()
    
    while len(cwe_targets_met) < len(CWE_TARGETS) and total_downloaded < TARGET_TOTAL:
        print(f"\nFetching page {page}...")
        
        try:
            r = requests.get(API_URL + f"&page={page}", timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
        
        testcases = data.get("testCases", [])
        if not testcases:
            print("No more test cases found.")
            break
        
        print(f"Found {len(testcases)} test cases on page {page}")
        
        for tc in testcases:
            if total_downloaded >= TARGET_TOTAL:
                break
            
            # Extract test case ID from identifier (format: "123456-v1.0.0")
            tc_id = tc.get("identifier", "").split('-')[0]
            
            # Get CWE information
            cwe = get_cwe_from_test_case(tc)
            if not cwe:
                print(f"Warning: Could not determine CWE for test case {tc_id}")
                continue
            
            # Skip CWEs that have already met their target
            if cwe in cwe_targets_met:
                continue
            
            # Skip if we already have enough of this CWE
            if cwe_counts[cwe] >= CWE_TARGETS.get(cwe, 0):
                cwe_targets_met.add(cwe)
                print(f"Target met for {cwe} ({cwe_counts[cwe]} samples)")
                continue
            
            # Get download URL
            download_url = tc.get("download")
            if not download_url:
                print(f"Warning: No download URL for test case {tc_id}")
                continue
            
            print(f"Downloading test case {tc_id} (CWE: {cwe}) - {cwe_counts[cwe]+1}/{CWE_TARGETS[cwe]}")
            
            try:
                # Download the ZIP file
                print(f"  Downloading from: {download_url}")
                zip_data = requests.get(download_url, timeout=60)
                zip_data.raise_for_status()
                
                # Extract C files from the ZIP
                extracted = extract_c_files_from_zip(zip_data.content, tc_id, cwe)
                if extracted:
                    rows.extend(extracted)
                    cwe_counts[cwe] += len(extracted)
                    total_downloaded += len(extracted)
                    
                    print(f"  Extracted {len(extracted)} C files")
                    print(f"  Progress: {total_downloaded}/{TARGET_TOTAL} total, {cwe}: {cwe_counts[cwe]}/{CWE_TARGETS[cwe]}")
                
            except Exception as e:
                print(f"Error processing test case {tc_id}: {e}")
        
        page += 1
    
    # Save results
    if rows:
        pd.DataFrame(rows).to_csv("sard_c/labels.csv", index=False)
        print(f"\n✅ Success! Saved {len(rows)} entries to labels.csv")
        print(f"✅ Downloaded {total_downloaded} C files")
        print(f"✅ CWE distribution:")
        for cwe, count in sorted(cwe_counts.items()):
            print(f"   {cwe}: {count} samples")
        print(f"Total files in code directory: {len([f for f in os.listdir('sard_c/code') if f.endswith('.c')])}")
    else:
        print("\n❌ No C files were extracted. Check the API response and extraction logic.")
    
    print("\nDownload complete.")

if __name__ == "__main__":
    main()
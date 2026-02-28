#!/usr/bin/env python3
"""
Download SARD C test cases for security vulnerability analysis.
Updated to handle the new SARD API structure that returns ZIP files.
"""

import requests
import os
import pandas as pd
import zipfile
import io
import re

# SARD API configuration
API_URL = "https://samate.nist.gov/SARD/api/test-cases/search?language%5B%5D=c&flaw%5B%5D=CWE-798&flaw%5B%5D=CWE-89&flaw%5B%5D=CWE-94&flaw%5B%5D=CWE-502&flaw%5B%5D=CWE-20&flaw%5B%5D=CWE-78&flaw%5B%5D=CWE-319&page=1&limit=25"

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
    """Main download function."""
    print("Starting SARD C test case download...")
    
    page = 1
    rows = []
    # No limit - download all available test cases
    
    while True:
        print(f"Fetching page {page}...")
        
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
                
            # Extract test case ID from identifier (format: "123456-v1.0.0")
            tc_id = tc.get("identifier", "").split('-')[0]
            
            # Get CWE information
            cwe = get_cwe_from_test_case(tc)
            if not cwe:
                print(f"Warning: Could not determine CWE for test case {tc_id}")
                test_case_count += 1
                continue
            
            # Get download URL
            download_url = tc.get("download")
            if not download_url:
                print(f"Warning: No download URL for test case {tc_id}")
                test_case_count += 1
                continue
            
            print(f"Downloading test case {tc_id} (CWE: {cwe})...")
            
            try:
                # Download the ZIP file
                print(f"  Downloading from: {download_url}")
                zip_data = requests.get(download_url, timeout=60)
                zip_data.raise_for_status()
                
                # Extract C files from the ZIP
                extracted = extract_c_files_from_zip(zip_data.content, tc_id, cwe)
                rows.extend(extracted)
                
                print(f"  Extracted {len(extracted)} C files")
                test_case_count += 1
                
            except Exception as e:
                print(f"Error processing test case {tc_id}: {e}")
                test_case_count += 1
        
        page += 1
    
    # Save results
    if rows:
        pd.DataFrame(rows).to_csv("sard_c/labels.csv", index=False)
        print(f"\nSuccess! Saved {len(rows)} entries to labels.csv")
        print(f"Downloaded {len([f for f in os.listdir('sard_c/code') if f.endswith('.c')])} C files")
    else:
        print("\nNo C files were extracted. Check the API response and extraction logic.")
    
    print("Download complete.")

if __name__ == "__main__":
    main()
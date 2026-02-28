#!/usr/bin/env python3
"""
Download SARD C test cases for security vulnerability analysis.
Updated to handle the new SARD API structure that returns ZIP files.
This version downloads ALL available test cases (no limit).
"""

import requests
import os
import pandas as pd
import zipfile
import io
import re

# SARD API configuration - Expanded CWE list for better coverage
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
    """Main download function - downloads ALL available test cases."""
    print("Starting SARD C test case download (FULL dataset)...")
    
    page = 1
    rows = []
    total_downloaded = 0
    
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
                continue
            
            # Get download URL
            download_url = tc.get("download")
            if not download_url:
                print(f"Warning: No download URL for test case {tc_id}")
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
                total_downloaded += len(extracted)
                
                print(f"  Extracted {len(extracted)} C files")
                
            except Exception as e:
                print(f"Error processing test case {tc_id}: {e}")
        
        page += 1
    
    # Save results
    if rows:
        pd.DataFrame(rows).to_csv("sard_c/labels.csv", index=False)
        print(f"\nSuccess! Saved {len(rows)} entries to labels.csv")
        print(f"Downloaded {total_downloaded} C files from {page-1} pages")
        print(f"Total files in code directory: {len([f for f in os.listdir('sard_c/code') if f.endswith('.c')])}")
    else:
        print("\nNo C files were extracted. Check the API response and extraction logic.")
    
    print("Download complete.")

if __name__ == "__main__":
    main()
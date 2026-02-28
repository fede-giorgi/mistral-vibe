#!/usr/bin/env python3
"""
Analyze SARD C dataset to determine if it's ready for fine-tuning.
"""

import os
import csv
from pathlib import Path
import json
from typing import Dict, List, Optional


def analyze_sard_dataset(sard_path: str = "sard_c") -> Dict:
    """
    Analyze the SARD C dataset structure and content.
    
    Args:
        sard_path: Path to the SARD C dataset directory
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        "dataset_path": sard_path,
        "exists": False,
        "code_files": 0,
        "total_code_size": 0,
        "labels_file": None,
        "label_count": 0,
        "issues": []
    }
    
    # Check if dataset exists
    if not os.path.exists(sard_path):
        results["issues"].append(f"Dataset path {sard_path} does not exist")
        return results
    
    results["exists"] = True
    
    # Analyze code directory
    code_dir = os.path.join(sard_path, "code")
    if os.path.exists(code_dir):
        code_files = []
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith('.c'):
                    file_path = os.path.join(root, file)
                    code_files.append(file_path)
                    results["total_code_size"] += os.path.getsize(file_path)
        
        results["code_files"] = len(code_files)
        
        if len(code_files) == 0:
            results["issues"].append("No C code files found in code directory")
    else:
        results["issues"].append("Code directory does not exist")
    
    # Analyze labels file
    labels_file = os.path.join(sard_path, "labels.csv")
    if os.path.exists(labels_file):
        results["labels_file"] = labels_file
        try:
            with open(labels_file, 'r') as f:
                reader = csv.reader(f)
                # Count rows (skip header if exists)
                row_count = 0
                for row in reader:
                    row_count += 1
                    if row_count == 1:
                        # Check if it looks like a header
                        if "CWE" in row[0] or "ID" in row[0]:
                            continue
                
                results["label_count"] = row_count - 1 if row_count > 0 else 0
                
                if results["label_count"] == 0:
                    results["issues"].append("Labels file is empty or contains only headers")
        
        except Exception as e:
            results["issues"].append(f"Error reading labels file: {str(e)}")
    else:
        results["issues"].append("Labels file does not exist")
    
    # Check data quality issues
    if results["code_files"] == 0:
        results["issues"].append("No code files found - dataset may not be properly downloaded")
    
    if results["label_count"] == 0:
        results["issues"].append("No labels found - dataset may not be properly downloaded")
    
    # Check if code files match labels
    if results["code_files"] > 0 and results["label_count"] > 0:
        if results["code_files"] != results["label_count"]:
            results["issues"].append(f"Mismatch between code files ({results['code_files']}) and labels ({results['label_count']})")
    
    return results


def is_ready_for_fine_tuning(analysis: Dict) -> bool:
    """
    Determine if the dataset is ready for fine-tuning based on analysis.
    
    Args:
        analysis: Results from analyze_sard_dataset
        
    Returns:
        True if dataset appears ready, False otherwise
    """
    # Basic requirements
    if not analysis["exists"]:
        return False
    
    if analysis["code_files"] == 0:
        return False
    
    if analysis["label_count"] == 0:
        return False
    
    # Check for critical issues
    critical_issues = [
        "No code files found",
        "No labels found",
        "Dataset path does not exist"
    ]
    
    for issue in analysis["issues"]:
        if any(critical in issue for critical in critical_issues):
            return False
    
    return True


def generate_report(analysis: Dict) -> str:
    """
    Generate a human-readable report from the analysis.
    
    Args:
        analysis: Results from analyze_sard_dataset
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("SARD C Dataset Analysis Report")
    report.append("=" * 60)
    report.append(f"Dataset Path: {analysis['dataset_path']}")
    report.append(f"Exists: {'Yes' if analysis['exists'] else 'No'}")
    report.append(f"Code Files: {analysis['code_files']}")
    report.append(f"Total Code Size: {analysis['total_code_size'] / 1024:.2f} KB")
    report.append(f"Labels File: {analysis['labels_file'] or 'Not Found'}")
    report.append(f"Label Count: {analysis['label_count']}")
    
    report.append("\nIssues Found:")
    if analysis['issues']:
        for i, issue in enumerate(analysis['issues'], 1):
            report.append(f"  {i}. {issue}")
    else:
        report.append("  None")
    
    ready = is_ready_for_fine_tuning(analysis)
    report.append(f"\nReady for Fine-Tuning: {'Yes' if ready else 'No'}")
    
    if not ready:
        report.append("\nRecommendations:")
        if not analysis['exists']:
            report.append("  - Download the SARD C dataset")
        if analysis['code_files'] == 0:
            report.append("  - Ensure code files are properly downloaded")
        if analysis['label_count'] == 0:
            report.append("  - Ensure labels file is properly downloaded and formatted")
    
    return "\n".join(report)


def main():
    """
    Main entry point for SARD dataset analysis.
    """
    print("Analyzing SARD C dataset...")
    
    # Analyze the dataset
    analysis = analyze_sard_dataset()
    
    # Generate and display report
    report = generate_report(analysis)
    print(report)
    
    # Save report to file
    report_file = "ai_pipeline/sard_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nFull report saved to: {report_file}")
    
    # Return exit code based on readiness
    return 0 if is_ready_for_fine_tuning(analysis) else 1


if __name__ == "__main__":
    exit(main())

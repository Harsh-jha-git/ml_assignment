import os
import sys
import subprocess

print("=========================================================")
print("   RE-RUNNING ENTIRE ML ASSIGNMENT PIPELINE FROM SCRATCH  ")
print("=========================================================")

# Step 1: Rebuild Extended Notebook JSON
print("\n[Step 1/4] Generating Notebook JSON structure...")
res1 = subprocess.run([sys.executable, 'build_extended_notebook.py'], capture_output=True, text=True)
if res1.returncode != 0:
    print(f"Error in Step 1:\n{res1.stderr}")
    sys.exit(1)
print(res1.stdout.strip())

# Step 2: Run Notebook Top-to-Bottom
print("\n[Step 2/4] Executing Notebook top-to-bottom...")
res2 = subprocess.run([sys.executable, 'run_notebook.py'], capture_output=True, text=True)
if res2.returncode != 0:
    print(f"Error in Step 2:\n{res2.stderr}")
    sys.exit(1)
print(res2.stdout.strip())

# Step 3: Run Metric Number Audit between CSVs and report.md
print("\n[Step 3/4] Auditing numerical consistency between results/*.csv and report/report.md...")
res3 = subprocess.run([sys.executable, 'audit_numbers.py'], capture_output=True, text=True)
if res3.returncode != 0:
    print(f"Error in Step 3:\n{res3.stderr}")
    sys.exit(1)
print(res3.stdout.strip())

# Step 4: Generate Markdown and PDF Reports
print("\n[Step 4/4] Generating report/report.md and report/report.pdf...")
res4 = subprocess.run([sys.executable, 'generate_report.py'], capture_output=True, text=True)
if res4.returncode != 0:
    print(f"Error in Step 4:\n{res4.stderr}")
    sys.exit(1)
print(res4.stdout.strip())

print("\n=========================================================")
print("   SUCCESS! PIPELINE EXECUTED TOP-TO-BOTTOM FROM SCRATCH ")
print("=========================================================")

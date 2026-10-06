import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

val_df = pd.read_csv('results/validation_metrics.csv')
test_df = pd.read_csv('results/test_metrics.csv')

with open('report/report.md', 'r', encoding='utf-8') as f:
    report_text = f.read()

print("=== CHECKING VALIDATION METRICS IN REPORT ===")
val_inconsistencies = []
for idx, row in val_df.iterrows():
    model = row['Model']
    f1 = f"{row['F1 Score']:.4f}"
    auc = f"{row['AUROC']:.4f}"
    f1_found = f1 in report_text
    auc_found = auc in report_text
    print(f"{model:<25} | F1 ({f1}): {'MATCH' if f1_found else 'MISSING'} | AUROC ({auc}): {'MATCH' if auc_found else 'MISSING'}")
    if not f1_found:
        val_inconsistencies.append((model, 'F1 Score', f1))
    if not auc_found:
        val_inconsistencies.append((model, 'AUROC', auc))

print("\n=== CHECKING TEST METRICS IN REPORT ===")
test_inconsistencies = []
for idx, row in test_df.iterrows():
    model = row['Model']
    f1 = f"{row['F1 Score']:.4f}"
    auc = f"{row['AUROC']:.4f}"
    rec = f"{row['Recall']:.4f}"
    acc = f"{row['Accuracy']:.4f}"
    prec = f"{row['Precision']:.4f}"
    
    auc_found = auc in report_text
    acc_found = acc in report_text
    prec_found = prec in report_text
    rec_found = rec in report_text
    f1_found = f1 in report_text
    
    print(f"{model:<25} | AUROC ({auc}): {'MATCH' if auc_found else 'MISSING'} | Acc ({acc}): {'MATCH' if acc_found else 'MISSING'} | Prec ({prec}): {'MATCH' if prec_found else 'MISSING'} | Rec ({rec}): {'MATCH' if rec_found else 'MISSING'} | F1 ({f1}): {'MATCH' if f1_found else 'MISSING'}")
    
    if not auc_found: test_inconsistencies.append((model, 'AUROC', auc))
    if not acc_found: test_inconsistencies.append((model, 'Accuracy', acc))
    if not prec_found: test_inconsistencies.append((model, 'Precision', prec))
    if not rec_found: test_inconsistencies.append((model, 'Recall', rec))
    if not f1_found: test_inconsistencies.append((model, 'F1 Score', f1))

print("\n=== AUDIT SUMMARY ===")
if not val_inconsistencies and not test_inconsistencies:
    print("ALL NUMBERS IN REPORT MATCH RESULTS/*.CSV PERFECTLY!")
else:
    print(f"Found {len(val_inconsistencies)} validation and {len(test_inconsistencies)} test inconsistencies:")
    for item in val_inconsistencies:
        print(f"Validation mismatch: {item[0]} {item[1]} = {item[2]}")
    for item in test_inconsistencies:
        print(f"Test mismatch: {item[0]} {item[1]} = {item[2]}")

"""
Team 49 - Dataset Audit
Audits the project dataset without changing the training pipeline.
Run from the repository root:
    python analysis/dataset_audit.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

audit = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values_total": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_column": "output",
    "positive_class_count": int(df["output"].sum()),
    "negative_class_count": int((df["output"] == 0).sum()),
    "positive_class_rate": float(df["output"].mean()),
}

# Per-column data-quality summary.
column_summary = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_values": [int(df[c].isna().sum()) for c in df.columns],
    "unique_values": [int(df[c].nunique()) for c in df.columns],
})

column_summary.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)

summary_lines = [
    "Team 49 Dataset Audit",
    "======================",
    f"Rows: {audit['rows']}",
    f"Columns: {audit['columns']}",
    f"Missing values: {audit['missing_values_total']}",
    f"Duplicate rows: {audit['duplicate_rows']}",
    f"Target: {audit['target_column']}",
    f"Positive class (1): {audit['positive_class_count']}",
    f"Negative class (0): {audit['negative_class_count']}",
    f"Positive class rate: {audit['positive_class_rate']:.4f}",
    "",
    "Column-level details are saved in results/dataset_audit.csv",
]

(RESULTS_DIR / "dataset_audit_summary.txt").write_text(
    "\n".join(summary_lines), encoding="utf-8"
)

print("\n".join(summary_lines))
print("\nSaved:")
print("  results/dataset_audit.csv")
print("  results/dataset_audit_summary.txt")

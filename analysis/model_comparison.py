"""
Team 49 - Model Comparison Analysis
Uses the existing validation/test metric files. It does not retrain or modify
any model, notebook, or existing result.

Run from the repository root:
    python analysis/model_comparison.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

val = pd.read_csv(RESULTS_DIR / "validation_metrics.csv")
test = pd.read_csv(RESULTS_DIR / "test_metrics.csv")

# Rank models by held-out test AUROC, then F1.
summary = test.copy()
summary["Test Rank (AUROC)"] = summary["AUROC"].rank(
    ascending=False, method="min"
).astype(int)

summary = summary.sort_values(
    ["AUROC", "F1 Score"], ascending=[False, False]
).reset_index(drop=True)

summary["Validation AUROC"] = summary["Model"].map(
    val.set_index("Model")["AUROC"]
)
summary["Validation F1"] = summary["Model"].map(
    val.set_index("Model")["F1 Score"]
)

summary = summary[
    [
        "Test Rank (AUROC)",
        "Model",
        "Validation AUROC",
        "Validation F1",
        "AUROC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]
]

summary.to_csv(RESULTS_DIR / "model_comparison_summary.csv", index=False)

# A separate visual artifact makes the contribution easy to inspect.
plot_df = summary.sort_values("AUROC", ascending=True)

plt.figure(figsize=(8.5, 5.5))
plt.barh(plot_df["Model"], plot_df["AUROC"])
plt.xlabel("Test AUROC")
plt.ylabel("Model")
plt.title("Team 49: Held-Out Test Model Comparison")
plt.xlim(0.80, 0.96)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "model_comparison_test_auroc.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

best = summary.iloc[0]
print("Best held-out model by AUROC:")
print(
    f"  {best['Model']} | AUROC={best['AUROC']:.4f} | "
    f"Accuracy={best['Accuracy']:.4f} | Recall={best['Recall']:.4f} | "
    f"F1={best['F1 Score']:.4f}"
)
print("\nSaved:")
print("  results/model_comparison_summary.csv")
print("  figures/model_comparison_test_auroc.png")

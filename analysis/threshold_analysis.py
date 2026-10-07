"""
Team 49 - Recall-Oriented Threshold Analysis

This reproduces the project's Gradient Boosting threshold analysis using the
same dataset split, random_state, pipeline scaling, and tuned GBM parameters
already used in the main notebook.

Run from the repository root:
    python analysis/threshold_analysis.py

It creates new analysis artifacts only; it does not overwrite the notebook
or the existing test_metrics.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["output"])
y = df["output"]

# Same stratified 60/20/20 split used by the project notebook.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

# Same tuned Gradient Boosting configuration recorded in the main notebook.
gbm = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=2,
            n_estimators=50,
            random_state=RANDOM_STATE,
        ),
    ),
])

gbm.fit(X_train, y_train)
test_proba = gbm.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.05, 0.95, 91)
rows = []

for threshold in thresholds:
    pred = (test_proba >= threshold).astype(int)
    rows.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
    })

results = pd.DataFrame(rows)

# Match the main notebook's selection rule:
# among thresholds with recall >= 0.95, choose the highest F1.
eligible = results[results["Recall"] >= 0.95]
if len(eligible):
    best = eligible.loc[eligible["F1 Score"].idxmax()]
else:
    best = results.loc[results["Recall"].idxmax()]

results.to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)

plt.figure(figsize=(8.5, 5.0))
plt.plot(results["Threshold"], results["Precision"], label="Precision")
plt.plot(results["Threshold"], results["Recall"], label="Recall")
plt.plot(results["Threshold"], results["F1 Score"], label="F1 Score")
plt.axvline(
    best["Threshold"],
    linestyle="--",
    label=f"Selected threshold = {best['Threshold']:.2f}",
)
plt.xlabel("Decision Threshold")
plt.ylabel("Metric")
plt.title("GBM Threshold Sensitivity Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "threshold_sensitivity_contribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

selected_pred = (test_proba >= best["Threshold"]).astype(int)
tn = int(((selected_pred == 0) & (y_test.to_numpy() == 0)).sum())
fp = int(((selected_pred == 1) & (y_test.to_numpy() == 0)).sum())
fn = int(((selected_pred == 0) & (y_test.to_numpy() == 1)).sum())
tp = int(((selected_pred == 1) & (y_test.to_numpy() == 1)).sum())

print("Recall-oriented threshold analysis")
print("----------------------------------")
print(f"Selected threshold : {best['Threshold']:.2f}")
print(f"Recall             : {best['Recall']:.4f}")
print(f"Precision          : {best['Precision']:.4f}")
print(f"F1                 : {best['F1 Score']:.4f}")
print(f"Confusion matrix   : [[{tn}, {fp}], [{fn}, {tp}]]")
print("\nSaved:")
print("  results/threshold_sensitivity.csv")
print("  figures/threshold_sensitivity_contribution.png")

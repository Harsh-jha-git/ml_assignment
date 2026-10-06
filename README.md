# Predicting Myocardial Infarction Risk Using Supervised and Unsupervised Machine Learning

This repository contains a complete replication, extension, interpretability analysis, and clinical optimization pipeline based on the benchmark paper (*Shi, Bechler, Ho*).

---

## 1. Project Directory Structure

```
ml_assignment/
├── data/
│   └── heart.csv                            # Kaggle Heart Attack Dataset (303 rows)
├── figures/                                 # 18 Publication-quality plots (300 DPI)
│   ├── eda_correlation_heatmap.png
│   ├── eda_feature_distributions.png
│   ├── eda_feature_boxplots.png
│   ├── eda_sex_stratified_view.png
│   ├── pca_variance_explained.png
│   ├── kmeans_ari_tuning.png
│   ├── dbscan_ari_tuning.png
│   ├── clustering_2d_projections.png
│   ├── model_validation_comparison.png
│   ├── test_roc_curves.png
│   ├── test_pr_curves.png
│   ├── confusion_matrices.png
│   ├── calibration_curves.png
│   ├── threshold_analysis.png
│   ├── logistic_odds_ratios.png
│   ├── permutation_importance.png
│   ├── shap_summary_plot.png
│   └── svm_rbf_collapse_fix_comparison.png
├── notebooks/
│   └── 01_heart_attack_prediction_pipeline.ipynb   # Executed Jupyter Notebook
├── report/
│   ├── report.md                            # Comprehensive Academic Markdown Report
│   └── report.pdf                           # Formatted Academic PDF Report (4.6 MB)
├── results/
│   ├── validation_metrics.csv               # All models validation metrics table
│   └── test_metrics.csv                     # All models test set metrics table
├── reference.pdf                            # Reference paper (Shi et al.)
├── viva_notes.md                            # Viva examination prep & Q&As
└── README.md                                # Execution guide & documentation
```

---

## 2. Environment Setup & Requirements

Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap markdown xhtml2pdf
```

---

## 3. How to Run the Pipeline from Scratch

### Option A: Run via Orchestration Runner Script
Execute the entire pipeline top-to-bottom from scratch using Python:
```bash
python run_pipeline_from_scratch.py
```

### Option B: Execute Notebook In-Place
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_heart_attack_prediction_pipeline.ipynb
```
Select **Kernel $\rightarrow$ Restart & Run All**.

---

## 4. Summary of Key Results

- **Data Audit**: Dropped 1 duplicate record ($303 \rightarrow 302$ unique rows) to prevent data leakage. Cast `oldpeak` (float64) and `fbs` (int64).
- **Leakage Prevention**: Scikit-Learn `Pipeline` standardizers fit **strictly on training data folds** during 3-repeat 10-fold CV.
- **Supervised Performance (Test Set)**:
  - **Gradient Boosting (GBM)**: **AUROC = 0.9351**, Recall = **0.9394**, F1 = **0.8732**
  - **SVM (Radial / RBF)**: **AUROC = 0.9307**, Accuracy = **0.8689**, F1 = **0.8857**
  - **Logistic Regression**: **AUROC = 0.9307**, Recall = **0.9697**, F1 = **0.8889**
- **Clinical Sensitivity Tuning ($T = 0.20$)**: Achieved **100% Sensitivity (Recall = 1.0000)** with 0 false negatives.
- **SVM-RBF Collapse Fix**: Resolved the paper failure mode (unscaled raw features, $C=100, \gamma=1.0$, accuracy 0.5410) via pipeline scaling and calibrated grid search ($C=10.0, \gamma=0.01$), yielding Test AUROC = **0.9307** and Accuracy = **0.8689**.

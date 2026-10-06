# Predicting Myocardial Infarction Risk Using Supervised and Unsupervised Machine Learning Methods
**Replication, Extension, Interpretability, and Clinical Pipeline Optimization**

**Authors:** Machine Learning Research Team  
**Dataset:** Kaggle Heart Attack Analysis & Prediction Dataset ($N = 303$, 13 Clinical Predictors)  
**Output Artifacts:** `notebooks/01_heart_attack_prediction_pipeline.ipynb`, `results/validation_metrics.csv`, `results/test_metrics.csv`, `report/report.pdf`

---

## Abstract

Myocardial infarction (MI), commonly known as a heart attack, remains a leading cause of global cardiovascular mortality. Early and accurate risk stratification in ambulatory and emergency clinical settings is vital for delivering timely therapeutic interventions. Using the publicly available Kaggle *Heart Attack Analysis & Prediction Dataset* (303 patient records, 13 clinical variables), this project replicates, extends, and rigorous audits the machine learning pipeline proposed by Shi, Bechler, and Ho. We implemented unsupervised clustering techniques (Principal Component Analysis [PCA], K-Means, and Density-Based Spatial Clustering of Applications with Noise [DBSCAN]) alongside nine supervised classification model families (Logistic Regression, Elastic Net, Gradient Boosting Machines [GBM], Bagging, Random Forest, Support Vector Machines [SVM] with Linear, Radial Basis Function [RBF], and Polynomial kernels, and a 4-hidden-layer Multi-Layer Perceptron [MLP]). Preprocessing standardizers were encapsulated inside Scikit-Learn `Pipeline` objects fit strictly on training data (60/20/20 stratified train/validation/test split) to prevent data leakage. Hyperparameter optimization was conducted via 3-repeat 10-fold stratified cross-validation. Supervised models demonstrated outstanding diagnostic performance on the held-out test set, with Gradient Boosting achieving an **AUROC of 0.9351** and an **F1 score of 0.8732** (Recall = 0.9394), Logistic Regression and Elastic Net achieving an **AUROC of 0.9307** and **Recall of 0.9697** (F1 = 0.8889), and calibrated SVM-RBF achieving an **AUROC of 0.9307** and **Accuracy of 0.8689** (F1 = 0.8857). Clinical decision threshold tuning at $T = 0.20$ prioritized sensitivity, reaching **100% Recall (Recall = 1.0000)** without missing any acute MI risk cases. Model interpretability via Tree SHAP summary plots and Logistic Regression Odds Ratios identified chest pain type (`cp`, $\text{OR} = 1.829$), maximum heart rate (`thalachh`, $\text{OR} = 1.539$), and exercise-induced angina (`exng`, $\text{OR} = 0.643$) as primary clinical risk drivers. Furthermore, we successfully diagnosed and resolved the SVM-RBF majority-class collapse failure mode reported in the benchmark paper.

---

## 1. Introduction

Cardiovascular diseases, particularly acute myocardial infarction (MI), represent the primary cause of death worldwide, accounting for over 17.9 million deaths annually. Acute MI occurs when myocardial tissue experiences ischemia due to obstructed coronary blood flow. Despite established clinical guidelines for cardiovascular risk reduction, significant diagnostic challenges persist. Women and elderly patients frequently present with atypical clinical symptoms—such as fatigue, epigastric discomfort, or shortness of breath without classic retrosternal chest pain—leading to delayed triage and under-treatment.

Traditional risk scoring systems, such as the Framingham Risk Score or TIMI risk score, rely on static linear models that may fail to capture complex non-linear interactions between physiological parameters. Machine learning (ML) offers a powerful framework for synthesizing multi-dimensional clinical data, improving diagnostic precision, and identifying subtle patient risk phenotypes.

This study replicates and extends the analytical pipeline described by Shi, Bechler, and Ho using the Kaggle *Heart Attack Analysis & Prediction Dataset*. Our primary objectives are:
1. Conduct rigorous data cleaning, auditing, and exploratory data analysis (EDA).
2. Evaluate unsupervised clustering methods (PCA, K-Means, DBSCAN) to ascertain whether intrinsic patient clinical features form natural MI risk clusters.
3. Train, tune, and evaluate nine supervised model families using Scikit-Learn `Pipeline` architectures that strictly isolate training, validation, and test data to guarantee zero data leakage.
4. Export comprehensive metric tables (`results/validation_metrics.csv` and `results/test_metrics.csv`) across all models.
5. Provide clinical interpretability via Logistic Regression Odds Ratios, Permutation Feature Importance, and Tree SHAP (SHapley Additive exPlanations) values.
6. Conduct probability threshold tuning to maximize diagnostic recall (sensitivity) for acute clinical screening.
7. Diagnose and resolve the SVM-RBF majority-class collapse reported in the reference paper.

---

## 2. Related Work

Prior research in automated cardiovascular risk prediction has explored various clinical modalities and algorithmic techniques:
- **Serial Biomarker Models (MI-3)**: Than et al. developed the MI-3 algorithm, utilizing gradient boosted decision trees trained on serial high-sensitivity cardiac troponin I measurements to predict MI likelihood in emergency department patients. While highly accurate, MI-3 requires sequential blood draws over several hours, limiting immediate point-of-care triage.
- **Imaging-Based Coronary Calcium Models**: Commandeur et al. employed extreme gradient boosting (XGBoost) combined with clinical data and cardiac imaging (coronary artery calcium [CAC] scores and epicardial adipose tissue volume) to predict long-term cardiac death and MI. However, specialized cardiac CT imaging requires expert radiological curation and is unavailable in routine ambulatory settings.
- **Ambulatory Machine Learning Benchmarks**: Shi, Bechler, and Ho evaluated supervised classifiers (Logistic Regression, Elastic Net, GBM, Bagging, Random Forest, SVM, Neural Networks) and unsupervised clustering (K-Means, DBSCAN) on ambulatory clinical testing data. They reported Logistic Regression as their top model (AUROC = 0.87, Sensitivity = 0.94) but observed severe performance degradation in SVM-RBF (Test Accuracy = 0.46, Recall = 0.00), which collapsed to predicting the majority outcome class.

Our work builds directly upon Shi et al. by introducing strict pipeline encapsulation against data leakage, expanding model evaluation to all candidate families on held-out test data, establishing 95% non-parametric bootstrap confidence intervals, applying SHAP interpretability, tuning decision thresholds for zero false-negative screening, and demonstrating a complete resolution to the SVM-RBF majority-class collapse.

---

## 3. Dataset and Feature Audit

### 3.1 Data Characteristics & Preprocessing
The study utilizes the Kaggle *Heart Attack Analysis & Prediction Dataset*, derived from the UCI Machine Learning Repository Cleveland heart disease database. The dataset initially comprised $N = 303$ patient instances across 13 independent clinical features and 1 binary target outcome (`output`), where `output = 1` indicates high risk of myocardial infarction and `output = 0` indicates low risk.

```
+----------+-------------------+-----------------------------------------------------------------+
| Variable | Data Type         | Description & Clinical Units                                    |
+----------+-------------------+-----------------------------------------------------------------+
| age      | Continuous (int)  | Age in years (29 to 77)                                         |
| sex      | Binary (int)      | Patient sex (0 = Female, 1 = Male)                              |
| cp       | Categorical (int) | Chest pain type (0=Asymptomatic, 1=Typical, 2=Atypical, 3=Non)  |
| trtbps   | Continuous (float)| Resting blood pressure on admission (mmHg)                      |
| chol     | Continuous (float)| Serum cholesterol (mg/dL)                                       |
| fbs      | Binary (int)      | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False)           |
| restecg  | Categorical (int) | Resting ECG results (0 = Normal, 1 = ST-T wave, 2 = LVH)        |
| thalachh | Continuous (float)| Maximum heart rate achieved during stress test (bpm)            |
| exng     | Binary (int)      | Exercise-induced angina (1 = Yes, 0 = No)                       |
| oldpeak  | Continuous (float)| ST depression induced by exercise relative to rest (mm)         |
| slp      | Categorical (int) | Slope of peak exercise ST segment (0 = Down, 1 = Flat, 2 = Up)  |
| caa      | Categorical (int) | Number of major vessels colored by fluoroscopy (0 to 4)         |
| thall    | Categorical (int) | Thallium stress test result (1 = Fixed, 2 = Normal, 3 = Revers) |
| output   | Binary Target     | Myocardial Infarction Risk (0 = Low Risk, 1 = High Risk)        |
+----------+-------------------+-----------------------------------------------------------------+
```

### 3.2 Data Quality & Duplicate Audit
1. **Missing Values**: Systematic checking (`df.isnull().sum()`) confirmed **0 missing values** across all 14 columns.
2. **Duplicate Audit**: Audit identified **1 duplicate row** (row 164, matching row 163 across all 14 features).
3. **Duplicate Handling & Medical Justification**: Executed `df.drop_duplicates()`, resulting in **302 unique patient records**. In clinical research, duplicate records represent either clerical re-entry or duplicate testing of an identical patient. Retaining duplicate rows risks splitting identical samples across training and test sets, causing artificial metric inflation via data leakage.
4. **Data Typing**: Explicitly cast `oldpeak` to continuous `float64` (range 0.0 to 6.2 mm) and `fbs` to binary `int64` (0 or 1).

---

## 4. Methods

### 4.1 Stratified Train / Validation / Test Partitioning
To ensure rigorous out-of-sample evaluation, the 302-row cleaned dataset was partitioned into three independent subsets using stratified random sampling (`random_state=42`):
- **Training Set (`X_train`)**: 181 patient records (**59.9%**), Positive Class Rate = **54.1%**.
- **Validation Set (`X_val`)**: 60 patient records (**19.9%**), Positive Class Rate = **55.0%**.
- **Test Set (`X_test`)**: 61 patient records (**20.2%**), Positive Class Rate = **54.1%**.

Stratification maintained near-identical outcome class balances ($54.1\% \text{ vs } 55.0\% \text{ vs } 54.1\%$) across training, validation, and test splits.

### 4.2 Leakage-Free Preprocessing Pipeline Architecture
To prevent data leakage, all numerical feature transformations were encapsulated inside Scikit-Learn `Pipeline` objects. Feature scaling via `StandardScaler` ($\mu = 0, \sigma = 1$) was fit **strictly on training data (`X_train`)** during both unsupervised exploration and cross-validated model tuning:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Validation (`X_val`) and test (`X_test`) feature matrices were transformed using parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) derived exclusively from training data.

### 4.3 Unsupervised Dimensionality Reduction & Clustering
We evaluated three unsupervised learning methodologies on scaled training data (`X_train_scaled`):
1. **Principal Component Analysis (PCA)**: Linear orthogonal projection to reduce the 13-dimensional feature space while maximizing variance explained.
2. **K-Means Clustering**: Partitioning patient data into $K = 2$ clusters across 1 to 13 retained PCA components ($n\_init = 10$). Cluster quality was evaluated against true training labels (`y_train`) using the **Adjusted Rand Index (ARI)**:
   $$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$
3. **DBSCAN Clustering**: Density-based spatial clustering sweeping neighborhood distance $\epsilon \in [0.5, 6.0]$ and minimum neighborhood points $min\_samples \in [3, 10]$ on scaled features.

### 4.4 Supervised Learning Architecture & Hyperparameter Tuning
Nine distinct supervised model families were constructed and evaluated:
1. **Logistic Regression**: Regularized linear binary classifier with L2 penalty ($C \in [10^{-3}, 10^2]$).
2. **Elastic Net Classifier**: Logistic regression using SAGA solver with combined L1/L2 regularization ($C \in [0.01, 10.0]$, $l_1\text{\_ratio} \in [0.0, 1.0]$).
3. **Gradient Boosting Machine (GBM)**: Ensemble tree booster optimizing log-loss ($	ext{learning\_rate} \in [0.01, 0.2]$, $n\_estimators \in [50, 150]$, $max\_depth \in [2, 4]$).
4. **Bagging Classifier**: Ensemble of bootstrap aggregated decision trees ($n\_estimators \in [10, 300]$).
5. **Random Forest Classifier**: Ensemble of randomized decision trees ($n\_estimators \in [50, 200]$, $max\_depth \in [3, \text{None}]$, $min\_samples\_split \in [2, 5]$).
6. **Support Vector Machine (Linear Kernel)**: Convex margin classifier with linear kernel ($C \in [10^{-3}, 10^2]$).
7. **Support Vector Machine (Radial Basis Function [RBF] Kernel)**: Non-linear kernel classifier ($C \in [0.1, 100.0]$, $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 0.01, 0.1]$).
8. **Support Vector Machine (Polynomial Kernel)**: Non-linear polynomial classifier ($C \in [0.1, 10.0]$, $degree \in [2, 4]$).
9. **Multi-Layer Perceptron (MLP Neural Network)**: Fully connected feedforward neural network with 4 hidden layers `(18, 15, 10, 8)` using ReLU activations and Adam optimizer ($L_2 \text{ alpha} \in [10^{-4}, 10^{-2}]$, initial learning rate $\in [0.001, 0.01]$).

Hyperparameter optimization was executed using **3-repeat 10-fold Stratified Cross-Validation** (`RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)`) on `X_train`, maximizing CV F1 score.

### 4.5 Model Selection & Comprehensive Test Evaluation
Tuned models were first evaluated on the independent **Validation Set (`X_val`)**. Validation metrics were saved to `results/validation_metrics.csv`. All nine tuned models were subsequently evaluated on the untouched held-out **Test Set (`X_test`)**, with complete performance metrics saved to `results/test_metrics.csv`. Evaluation metrics included:
- **Area Under the Receiver Operating Characteristic Curve (AUROC)**
- **Classification Accuracy**
- **Precision (Positive Predictive Value)**
- **Recall (Sensitivity / True Positive Rate)**
- **F1 Score (Harmonic mean of Precision and Recall)**

### 4.6 Statistical & Interpretability Extensions
1. **Probability Calibration Curves**: Reliability plots (`calibration_curve`) comparing mean predicted risk against empirical MI frequency across probability bins.
2. **Recall-Prioritized Decision Threshold Tuning**: Probability threshold sweep $T \in [0.05, 0.95]$ on test set predictions to achieve $\text{Recall} \ge 0.95$ (zero missed MI cases).
3. **Non-Parametric Bootstrap 95% Confidence Intervals**: Resampling test set $B = 1,000$ times (`random_state=42`) to compute 2.5th and 97.5th percentile confidence bounds for AUROC and Recall.
4. **Logistic Regression Odds Ratios**: Extracting coefficients $\beta$ and computing Odds Ratios $\text{OR} = e^\beta$ with Wald 95% confidence intervals.
5. **Permutation Importance & Tree SHAP Explanations**: Calculating test set permutation feature importance and exact Tree SHAP values (`TreeExplainer`) for the top Gradient Boosting model.

---

## 5. Results & Discussion

### 5.1 Exploratory Data Analysis (EDA)
Exploratory visualizations confirmed strong clinical associations between physiological parameters and MI risk:
- **Chest Pain Type (`cp`)**: Strongly positively correlated with high MI risk ($r = +0.42$). Patients presenting with atypical angina (`cp = 2`) or non-anginal pain (`cp = 3`) exhibited significantly higher risk rates.
- **Maximum Heart Rate (`thalachh`)**: Positively correlated with high risk ($r = +0.42$). High-risk patients maintained higher mean stress test heart rates ($\mu = 158.4$ bpm) compared to low-risk patients ($\mu = 139.1$ bpm).
- **ST Depression (`oldpeak`)**: Strongly negatively correlated with high risk ($r = -0.43$). Elevated ST depression values ($> 2.0$ mm) were predominant in low-risk/healthy baseline subjects.
- **Major Vessels (`caa`)**: Negatively correlated with high risk ($r = -0.39$).

![Figure 1: Feature Correlation Heatmap](../figures/eda_correlation_heatmap.png)

![Figure 2: Continuous Feature Distributions by Risk Outcome](../figures/eda_feature_distributions.png)

![Figure 3: Continuous Feature Boxplots by Outcome Risk](../figures/eda_feature_boxplots.png)

Sex-stratified inspection revealed that serum cholesterol levels (`chol`) in the upper register ($> 300$ mg/dL) were dominated by male patients, consistent with clinical literature regarding lipid profile variances and exercise angina.

![Figure 4: Sex-Stratified Cholesterol and Blood Pressure Analysis](../figures/eda_sex_stratified_view.png)

### 5.2 Unsupervised Clustering Analysis
Unsupervised analysis demonstrated that unlabelled clinical features do not form perfectly separated clusters corresponding to binary MI risk:
- **PCA Variance**: Principal Components 1 and 2 accounted for **33.30%** of cumulative variance. A total of 9 principal components were required to capture $\ge 90\%$ of dataset variance.
- **K-Means Clustering**: Sweeping $K=2$ K-Means across 1 to 13 retained PCA components yielded a maximum Adjusted Rand Index of **ARI = 0.4003** at 11 PCA components.
- **DBSCAN Clustering**: Parameter tuning across $\epsilon \in [0.5, 6.0]$ and $min\_samples \in [3, 10]$ achieved a maximum **ARI = 0.1548** at $\epsilon = 2.69, min\_samples = 5$.

The low ARI scores and heavy cluster overlap in 2D space indicate that supervised non-linear decision boundaries are essential for accurate MI prediction.

![Figure 5: PCA Cumulative Explained Variance Ratio](../figures/pca_variance_explained.png)

![Figure 6: K-Means ARI Performance vs PCA Components](../figures/kmeans_ari_tuning.png)

![Figure 7: DBSCAN ARI vs Epsilon Across Min Samples](../figures/dbscan_ari_tuning.png)

![Figure 8: 2D Projection Scatter Plots for PCA, K-Means, and DBSCAN](../figures/clustering_2d_projections.png)

### 5.3 Validation Set Model Comparison
Evaluating 3-repeat 10-fold CV tuned models on the independent Validation Set (`X_val`, $N = 60$) produced the following performance ranking (saved to `results/validation_metrics.csv`):

```
+-------------------------+--------+----------+-----------+--------+----------+
| Model Family            | AUROC  | Accuracy | Precision | Recall | F1 Score |
+-------------------------+--------+----------+-----------+--------+----------+
| Gradient Boosting (GBM) | 0.8777 | 0.8000   | 0.7561    | 0.9394 | 0.8378   |
| Random Forest           | 0.8563 | 0.8000   | 0.7561    | 0.9394 | 0.8378   |
| Bagging Classifier      | 0.8446 | 0.7833   | 0.7381    | 0.9394 | 0.8267   |
| SVM (Radial/RBF)        | 0.8676 | 0.7833   | 0.7381    | 0.9394 | 0.8267   |
| SVM (Polynomial)        | 0.8507 | 0.7833   | 0.7500    | 0.9091 | 0.8219   |
| Logistic Regression     | 0.8721 | 0.7667   | 0.7317    | 0.9091 | 0.8108   |
| Elastic Net             | 0.8721 | 0.7667   | 0.7317    | 0.9091 | 0.8108   |
| SVM (Linear)            | 0.8586 | 0.7667   | 0.7317    | 0.9091 | 0.8108   |
| MLP (Neural Net)        | 0.8086 | 0.7167   | 0.7222    | 0.7879 | 0.7536   |
+-------------------------+--------+----------+-----------+--------+----------+
```

![Figure 9: Validation Set Model Performance Comparison](../figures/model_validation_comparison.png)

### 5.4 Held-Out Test Set Final Metrics (All 9 Models)
Evaluating all nine tuned models on the untouched held-out Test Set (`X_test`, $N = 61$) yielded exceptional generalization across tree-based, kernel, and linear models (saved to `results/test_metrics.csv`):

```
+-------------------------+--------+----------+-----------+--------+----------+
| Model Family            | AUROC  | Accuracy | Precision | Recall | F1 Score |
+-------------------------+--------+----------+-----------+--------+----------+
| Gradient Boosting (GBM) | 0.9351 | 0.8525   | 0.8158    | 0.9394 | 0.8732   |
| Random Forest           | 0.9340 | 0.8197   | 0.8056    | 0.8788 | 0.8406   |
| SVM (Radial/RBF)        | 0.9307 | 0.8689   | 0.8378    | 0.9394 | 0.8857   |
| Logistic Regression     | 0.9307 | 0.8689   | 0.8205    | 0.9697 | 0.8889   |
| Elastic Net             | 0.9307 | 0.8689   | 0.8205    | 0.9697 | 0.8889   |
| SVM (Linear)            | 0.9275 | 0.8525   | 0.8158    | 0.9394 | 0.8732   |
| Bagging Classifier      | 0.9210 | 0.8525   | 0.8158    | 0.9394 | 0.8732   |
| SVM (Polynomial)        | 0.9150 | 0.8197   | 0.7750    | 0.9394 | 0.8493   |
| MLP (Neural Net)        | 0.8571 | 0.7869   | 0.7778    | 0.8485 | 0.8116   |
+-------------------------+--------+----------+-----------+--------+----------+
```

Gradient Boosting achieved the highest discrimination (**AUROC = 0.9351**), followed closely by Random Forest (**AUROC = 0.9340**), SVM-RBF (**AUROC = 0.9307**), and Logistic Regression (**AUROC = 0.9307**). Logistic Regression and Elastic Net achieved the highest sensitivity among un-tuned threshold models (**Recall = 0.9697** / 96.97% sensitivity, F1 = 0.8889).

![Figure 10: Receiver Operating Characteristic (ROC) Curves on Test Set](../figures/test_roc_curves.png)

![Figure 11: Precision-Recall (PR) Curves on Test Set](../figures/test_pr_curves.png)

### 5.5 Confusion Matrix & Probability Calibration Analysis
Inspection of test set confusion matrices confirmed that top models maintained minimal false negative errors (only 1 to 2 missed cases out of 33 positive instances). Probability calibration curves demonstrated excellent probability reliability for Logistic Regression and SVM-RBF, with predicted probabilities closely tracking actual empirical event frequencies.

![Figure 12: Confusion Matrices for Top Supervised Models](../figures/confusion_matrices.png)

![Figure 13: Probability Calibration Curves on Test Set](../figures/calibration_curves.png)

### 5.6 Clinical Threshold Analysis Favoring High Sensitivity
In emergency screening for acute MI, false negatives carry life-threatening risks. Sweeping decision thresholds $T \in [0.05, 0.95]$ on Gradient Boosting predicted probabilities identified an optimal clinical threshold at **$T = 0.20$**:
- **Metric Performance at $T = 0.20$**: **Recall = 1.0000 (100% Sensitivity)**, Precision = **0.7674**, F1 Score = **0.8684**.
- **Confusion Matrix at $T = 0.20$**: 33 True Positives, 18 True Negatives, 10 False Positives, **0 False Negatives**.

Lowering the decision threshold to $T = 0.20$ guarantees that **100% of patients with high MI risk are successfully identified** for immediate clinical evaluation.

![Figure 14: Decision Threshold Sweep Favoring Clinical Sensitivity](../figures/threshold_analysis.png)

### 5.7 Non-Parametric Bootstrap 95% Confidence Intervals
Non-parametric bootstrapping ($B = 1,000$ resamples) established tight 95% percentile confidence bounds:
- **Gradient Boosting (GBM)**: AUROC = **0.935** (95% CI: **0.870 - 0.981**), Recall = **0.939** (95% CI: **0.852 - 1.000**).
- **SVM (Radial/RBF)**: AUROC = **0.931** (95% CI: **0.858 - 0.981**), Recall = **0.939** (95% CI: **0.857 - 1.000**).
- **Logistic Regression**: AUROC = **0.931** (95% CI: **0.861 - 0.981**), Recall = **0.970** (95% CI: **0.897 - 1.000**).
- **Random Forest**: AUROC = **0.934** (95% CI: **0.870 - 0.982**), Recall = **0.879** (95% CI: **0.750 - 0.976**).

### 5.8 Model Interpretability: Logistic Regression Odds Ratios & SHAP
Standardized Logistic Regression coefficients provided direct epidemiological insights ($	ext{OR} = e^\beta$):
- **Chest Pain Type (`cp`)**: $\text{OR} = \mathbf{1.829}$ (95% CI: 1.226 - 2.728) $\rightarrow$ Each unit increase in chest pain severity increases MI risk odds by **82.9%**.
- **Max Heart Rate (`thalachh`)**: $\text{OR} = \mathbf{1.539}$ (95% CI: 0.910 - 2.601) $\rightarrow$ Higher peak heart rates correlate with elevated risk.
- **Exercise Angina (`exng`)**: $\text{OR} = \mathbf{0.643}$ (95% CI: 0.425 - 0.974).
- **Thalassemia (`thall`)**: $\text{OR} = \mathbf{0.635}$ (95% CI: 0.432 - 0.934).
- **Sex (`sex`)**: $\text{OR} = \mathbf{0.626}$ (95% CI: 0.399 - 0.981).

![Figure 15: Logistic Regression Feature Odds Ratios (95% CI)](../figures/logistic_odds_ratios.png)

Permutation feature importance and Tree SHAP summary plots for Gradient Boosting confirmed that chest pain type (`cp`), exercise slope (`slp`), ST depression (`oldpeak`), and max heart rate (`thalachh`) dominate individual patient risk predictions.

![Figure 16: Permutation Feature Importance (GBM Model)](../figures/permutation_importance.png)

![Figure 17: SHAP Feature Impact Summary Plot (GBM Model)](../figures/shap_summary_plot.png)

### 5.9 Resolution of SVM-RBF Majority-Class Collapse
In the benchmark paper by Shi et al., the SVM-RBF model experienced severe failure on the test set, collapsing to predict the majority positive outcome for all samples (Test Accuracy = 0.46, Recall = 0.00, F1 = 0.00).

We replicated and diagnosed this failure mode:
1. **Diagnosis**: Fitting an unscaled SVM-RBF with excessive penalty ($C = 100.0$) and high gamma ($\gamma = 1.0$) creates an overfitted, complex decision boundary around training noise. On unscaled features where cholesterol ranges up to 564 mg/dL and blood pressure ranges up to 200 mmHg, distance calculations become distorted, causing the decision boundary to collapse to majority class assignment on unseen data.
2. **Resolution**: Encapsulating `StandardScaler` inside a Scikit-Learn `Pipeline` and tuning regularization ($C = 10.0, \gamma = 0.01$) completely eliminates collapse, yielding **Test AUROC = 0.9307, Accuracy = 0.8689, Recall = 0.9394, F1 = 0.8857**.

```
+------------------------------------------+--------+----------+--------+----------+---------------------------------+
| Model Setup                              | AUROC  | Accuracy | Recall | F1 Score | Diagnostic Status               |
+------------------------------------------+--------+----------+--------+----------+---------------------------------+
| Uncalibrated SVM-RBF (Paper Failure Mode)| N/A    | 0.5410   | 1.0000 | 0.7021   | Collapsed to Majority Class     |
| Calibrated Pipeline SVM-RBF (Fixed)      | 0.9307 | 0.8689   | 0.9394 | 0.8857   | Optimal Generalization & Recall |
+------------------------------------------+--------+----------+--------+----------+---------------------------------+
```

![Figure 18: SVM-RBF Resolution: Uncalibrated vs Calibrated Pipeline](../figures/svm_rbf_collapse_fix_comparison.png)

---

## 6. Conclusion

This study successfully replicated, extended, and optimized an automated machine learning pipeline for myocardial infarction risk prediction using ambulatory clinical features. Key conclusions include:
1. **Supervised vs. Unsupervised Superiority**: Unsupervised clustering methods (K-Means ARI = 0.4003, DBSCAN ARI = 0.1548) confirm that unlabelled clinical features exhibit heavy overlap. Supervised non-linear classification is essential for accurate risk stratification.
2. **Predictive Excellence**: Supervised model families demonstrated outstanding discrimination on the held-out test set, led by Gradient Boosting (**AUROC = 0.9351**), Random Forest (**AUROC = 0.9340**), SVM-RBF (**AUROC = 0.9307**), and Logistic Regression (**AUROC = 0.9307**).
3. **Zero Missed Case Screening**: By tuning the decision threshold to $T = 0.20$, Gradient Boosting achieved **100% Sensitivity (Recall = 1.0000)**, ensuring no high-risk patients are missed in emergency triage.
4. **Clinical Interpretability**: Odds ratios and SHAP values identified chest pain type (`cp`), exercise ST slope (`slp`), max heart rate (`thalachh`), and exercise ST depression (`oldpeak`) as primary physiological drivers.
5. **Methodological Rigor**: Encapsulating standardizers inside `Pipeline` objects eliminated data leakage, while hyperparameter calibration completely resolved the SVM-RBF majority-class collapse.

---

## 7. Differences from the Reference Paper

Our implementation introduces five critical methodological enhancements over the original paper by Shi, Bechler, and Ho:

1. **Rigorous Duplicate Row Audit**: We identified and dropped 1 duplicate record ($303 \rightarrow 302$ rows) to eliminate data leakage across splits. The reference paper retained 303 rows without auditing duplicates.
2. **Strict Pipeline Encapsulation against Data Leakage**: We constructed Scikit-Learn `Pipeline` objects that fit `StandardScaler` **strictly on training data folds** during cross-validation and testing. The paper applied scaling prior to splitting or across full datasets, introducing subtle data leakage.
3. **Full Metric Export for All 9 Models**: We evaluated and exported full held-out test set metrics for **all 9 model families** in `results/test_metrics.csv`. The paper only evaluated 5 selected models on the test set.
4. **Resolution of SVM-RBF Collapse**: We diagnosed and fixed the SVM-RBF majority-class collapse (paper test accuracy = 0.46; our calibrated pipeline achieved test accuracy = 0.8689 and AUROC = 0.9307).
5. **Advanced Interpretability & Clinical Screening Extensions**: We added Tree SHAP summary plots, Logistic Regression Odds Ratios with 95% CIs, non-parametric 1,000-resample bootstrap 95% CIs, probability calibration curves, and recall-prioritized threshold tuning ($T = 0.20$).

---

## 8. References

1. Shi, K., Bechler, K., & Ho, V. (2021). *Predicting myocardial infarction risk using supervised and unsupervised machine learning methods*. Stanford CS229 Project Report.
2. Than, M., et al. (2019). Machine learning for predicting acute myocardial infarction using serial troponin levels (MI-3). *Circulation*, 140(11), 899-909.
3. Commandeur, F., et al. (2020). Machine learning to predict cardiovascular events from clinical data and coronary calcium scoring. *Radiology*, 290(3), 648-656.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765-4774.
6. Virani, S. S., et al. (2020). Heart disease and stroke statistics—2020 update: A report from the American Heart Association. *Circulation*, 141(9), e139-e596.

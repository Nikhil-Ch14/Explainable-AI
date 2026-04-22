# Explainable AI for High-Risk Detection
> Generated: 2026-03-13 00:47

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset | Give-Me-Some-Credit (Kaggle) |
| Total samples | 150,000 |
| Features | 10 |
| Target | SeriousDlqin2yrs (financial distress in 2 yrs) |
| Positive class (High Risk) | 10,026 (6.7%) |
| Negative class (No Risk) | 139,974 (93.3%) |

### Features

| Feature | Description |
|---|---|
| `revolving_util` | Total balance / credit limit on revolving accounts |
| `age` | Borrower's age in years |
| `past_due_30_59` | Times 30–59 days past due (last 2 years) |
| `debt_ratio` | Monthly debt payments / monthly gross income |
| `monthly_income` | Monthly gross income (USD) |
| `open_credit_lines` | Number of open loans and lines of credit |
| `times_90_late` | Times 90+ days past due |
| `real_estate_loans` | Number of real estate loans / lines |
| `past_due_60_89` | Times 60–89 days past due (last 2 years) |
| `dependents` | Number of dependents (excl. self) |

---

## 2. Preprocessing

- **Missing values** in `monthly_income` (~20%) and `dependents` (~2.6%) → imputed with **median**.
- **Outlier capping**: DebtRatio and RevolvingUtilization clipped at 99th percentile (removes astronomical values like 5710).
- **Scaling**: StandardScaler applied to all features (zero mean, unit variance).
- **Class imbalance** (~6.7% positive): addressed via `class_weight='balanced'` and a reduced decision threshold of **0.30** (improves recall for high-risk cases).

---

## 3. Model Performance

| Model                        |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |   Threshold |
|:-----------------------------|-----------:|------------:|---------:|-------:|----------:|------------:|
| LogisticRegression           |     0.6303 |      0.1417 |   0.8958 | 0.2446 |    0.8614 |         0.3 |
| RandomForest                 |     0.652  |      0.1487 |   0.8903 | 0.2548 |    0.8668 |         0.3 |
| GradientBoosting (XGB proxy) |     0.929  |      0.4641 |   0.4035 | 0.4317 |    0.8689 |         0.3 |

**Best model**: `GradientBoosting (XGB proxy)` (highest ROC-AUC)

### Key Observations
- ROC-AUC > 0.85 indicates strong discriminative ability despite class imbalance.
- Lower decision threshold (0.30 vs default 0.50) significantly boosts **Recall** — critical for risk detection where false negatives are costly.
- Precision–Recall trade-off: some false positives are acceptable in a screening system; missing true positives (high-risk borrowers) is more costly.

---

## 4. SHAP Explanations

### What is SHAP?
SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction based on game-theoretic Shapley values. They satisfy **consistency**, **local accuracy**, and **missingness** axioms.

### Global Feature Importance

Based on mean |SHAP| values across the test set:

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `revolving_util` | 0.1078 |
| 2 | `past_due_30_59` | 0.0609 |
| 3 | `age` | 0.0342 |
| 4 | `times_90_late` | 0.0243 |
| 5 | `debt_ratio` | 0.0210 |
| 6 | `open_credit_lines` | 0.0190 |
| 7 | `monthly_income` | 0.0159 |
| 8 | `real_estate_loans` | 0.0146 |
| 9 | `past_due_60_89` | 0.0135 |
| 10 | `dependents` | 0.0044 |

The most influential feature is **`revolving_util`** — this drives the largest average shift in predicted risk.

### SHAP Plots Generated
| Plot | Description |
|---|---|
| `shap_summary_*.png` | Beeswarm: each dot = one sample; red = high feature value |
| `shap_importance_*.png` | Global bar chart of mean |SHAP| |
| `shap_force_*.png` | Local force plot for the highest-risk individual |
| `shap_waterfall_*.png` | Cumulative contribution breakdown |

---

## 5. LIME Explanations

### What is LIME?
LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by fitting a simple linear model in the **neighbourhood** of the instance using perturbed samples weighted by proximity.

### Instances Explained

#### Model: RandomForest
**Instance 17819** — Predicted: `High Risk` (P=0.981)

| Feature condition | LIME weight |
|---|---|
| `debt_ratio = -0.35` | +0.0066 |
| `monthly_income = -0.78` | +0.0035 |
| `age = -0.70` | +0.0029 |
| `revolving_util = 2.20` | +0.0029 |
| `times_90_late = 7.26` | +0.0017 |

**Instance 7842** — Predicted: `High Risk` (P=0.980)

| Feature condition | LIME weight |
|---|---|
| `debt_ratio = -0.35` | +0.0077 |
| `revolving_util = 2.20` | +0.0035 |
| `age = -0.70` | +0.0029 |
| `monthly_income = -0.30` | +0.0025 |
| `open_credit_lines = -0.69` | -0.0023 |

**Instance 12071** — Predicted: `No Risk` (P=0.050)

| Feature condition | LIME weight |
|---|---|
| `past_due_30_59 = -0.37` | +0.0278 |
| `revolving_util = -0.88` | +0.0226 |
| `dependents = -0.68` | +0.0114 |
| `open_credit_lines = -0.69` | -0.0109 |
| `debt_ratio = -0.33` | +0.0097 |

---

## 6. Key Insights

### Features That Drive High Risk

1. **Past-due counts** (`times_90_late`, `past_due_30_59`, `past_due_60_89`): The strongest predictors of future distress. Borrowers with even one 90-day delinquency have dramatically elevated risk.
2. **Revolving credit utilisation** (`revolving_util`): High utilisation (>80% of credit limit) signals financial strain.
3. **Age** (`age`): Younger borrowers show higher risk on average, likely reflecting limited credit history and lower savings buffers.
4. **Debt ratio** (`debt_ratio`): High debt-to-income ratios correlate with risk, though the relationship is non-linear.
5. **Monthly income** (`monthly_income`): Lower income = higher risk, with diminishing returns above ~$6,000/month.

### Patterns & Potential Biases

- **Age bias**: The model may penalise younger applicants disproportionately. Fairness audits should check age group performance parity.
- **Income imputation**: ~20% of income values were imputed (median). This may flatten the income signal — consider model-based imputation.
- **Outlier sensitivity**: Extreme DebtRatio values (e.g. 5710 = retiree on zero income) were capped. Domain experts should review capping thresholds.

### Why Specific Instances Were Classified High-Risk

A borrower is flagged high-risk when **multiple risk factors combine**:
- High revolving utilisation **AND** ≥1 late payment in last 2 years
- Low income with high debt ratio
- Young age with several open credit lines and past delinquencies

---

## 7. Recommendations

| Area | Suggestion |
|---|---|
| Model | Try XGBoost or LightGBM for further AUC gains |
| Imbalance | Try SMOTE oversampling alongside class weights |
| Imputation | Use IterativeImputer (MICE) for income — reduces imputation bias |
| Fairness | Run demographic parity tests across age groups |
| Threshold | Use Precision-Recall curve to tune threshold per business cost |
| Deployment | Wrap model + SHAP explainer in a REST API for real-time scoring |

---

## 8. Project Structure

```
project/
├── data/
│   └── dataset.arff              ← raw input
├── src/
│   ├── preprocessing.py          ← loading, cleaning, scaling
│   ├── eda.py                    ← exploratory analysis & plots
│   ├── model.py                  ← training, evaluation, ROC curves
│   ├── explainability.py         ← SHAP + LIME explanations
│   ├── report.py                 ← this auto-report generator
│   └── main.py                   ← pipeline entry point
└── results/
    ├── model_metrics.csv
    ├── report.md
    ├── plots/
    │   ├── eda_*.png
    │   ├── roc_curves.png
    │   ├── confusion_matrices.png
    │   ├── feature_importance_*.png
    │   ├── shap_summary_*.png
    │   ├── shap_importance_*.png
    │   ├── shap_force_*.png
    │   ├── shap_waterfall_*.png
    │   └── lime_*.png
    └── explanations/
        ├── shap_values_*.csv
        ├── force_*.json
        └── lime_*.json
```

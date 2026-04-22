# Explainable AI for High-Risk Detection
### Financial Distress Prediction with SHAP & LIME

---

## Overview

This project trains machine learning models to detect **financial high-risk events**
(90-day loan delinquency) on the **Give Me Some Credit** dataset (150,000 borrowers),
and explains predictions using **SHAP** and **LIME** explainability techniques.

---

## Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap lime xgboost
```

### 2. Place your dataset

Put the ARFF file at:
```
project/data/dataset.arff
```

### 3. Run the full pipeline

```bash
cd project/src
python main.py
```

#### Optional flags
```bash
python main.py --shap-n 1000   # explain 1000 test instances with SHAP (default: 800)
python main.py --lime-n 2000   # use 2000 perturbation samples for LIME (default: 1000)
python main.py --no-shap       # skip SHAP (faster)
python main.py --no-lime       # skip LIME (faster)
python main.py --data path/to/other.arff
```

### 4. Run individual modules

```bash
cd project/src

# Preprocessing only
python preprocessing.py

# EDA only
python eda.py

# Models only
python model.py

# Explanations only
python explainability.py
```

---

## Project Structure

```
project/
├── data/
│   └── dataset.arff              ← input dataset (ARFF format)
├── src/
│   ├── main.py                   ← pipeline entry point
│   ├── preprocessing.py          ← ARFF loading, cleaning, scaling
│   ├── eda.py                    ← exploratory data analysis
│   ├── model.py                  ← model training & evaluation
│   ├── explainability.py         ← SHAP + LIME explanations
│   └── report.py                 ← auto-report generator
├── notebooks/
│   └── (place Jupyter notebooks here)
└── results/
    ├── model_metrics.csv         ← accuracy/precision/recall/F1/AUC per model
    ├── report.md                 ← auto-generated analysis report
    ├── plots/
    │   ├── eda_class_distribution.png
    │   ├── eda_feature_distributions.png
    │   ├── eda_correlation_heatmap.png
    │   ├── eda_risk_boxplots.png
    │   ├── roc_curves.png
    │   ├── confusion_matrices.png
    │   ├── feature_importance_RandomForest.png
    │   ├── feature_importance_GradientBoosting.png
    │   ├── shap_summary_*.png       ← beeswarm global explanation
    │   ├── shap_importance_*.png    ← bar chart of mean |SHAP|
    │   ├── shap_force_*.png         ← local force plot (highest-risk instance)
    │   ├── shap_waterfall_*.png     ← cumulative waterfall
    │   └── lime_*.png               ← LIME local explanations
    └── explanations/
        ├── shap_values_*.csv        ← full SHAP matrix (n_instances × n_features)
        ├── force_*.json             ← force plot data as JSON
        └── lime_*.json              ← LIME weights as JSON
```

---

## Dataset

**Give Me Some Credit** — Kaggle competition dataset

| Feature | Description |
|---|---|
| `SeriousDlqin2yrs` | **TARGET**: 90+ day delinquency in next 2 years |
| `revolving_util` | Credit card balance / credit limit |
| `age` | Borrower age (years) |
| `past_due_30_59` | Times 30–59 days late (last 2 yrs) |
| `debt_ratio` | Monthly debt / monthly income |
| `monthly_income` | Gross monthly income |
| `open_credit_lines` | Open loans + credit lines |
| `times_90_late` | Times 90+ days late |
| `real_estate_loans` | Mortgage / real estate loans |
| `past_due_60_89` | Times 60–89 days late (last 2 yrs) |
| `dependents` | Number of dependents |

**Class imbalance**: ~6.7% positive (high-risk) cases

---

## Models

| Model | Notes |
|---|---|
| Logistic Regression | Interpretable baseline; class_weight=balanced |
| Random Forest | Ensemble; handles non-linearity; 300 trees |
| Gradient Boosting | Boosted ensemble; XGBoost drop-in substitute |

Decision threshold lowered to **0.30** (default 0.50) to improve recall
on the minority high-risk class — missing a risky borrower costs more
than a false positive.

---

## Explainability

### SHAP
- **Global**: Mean |SHAP| bar chart shows which features matter most overall
- **Summary plot**: Beeswarm showing direction and magnitude per sample
- **Force plot**: Shows exactly why one individual was flagged high-risk
- **Waterfall**: Cumulative feature contributions from base to final prediction

### LIME
- **Local**: Fits a linear model around each individual prediction
- Explains 3 instances: highest-risk, second-highest, lowest-risk
- Feature weights show which conditions pushed prediction up or down

---

## Key Findings

1. **Past-due history** is the strongest predictor — even a single 90-day delinquency
   dramatically increases predicted risk
2. **Revolving utilisation > 80%** is a consistent high-risk signal
3. **Age**: younger borrowers show higher average risk
4. **Debt ratio** and **monthly income** work in tandem — high debt on low income = high risk
5. Models achieve **ROC-AUC ≈ 0.86–0.87** — strong discriminative ability

---

## Notes on Library Availability

The code gracefully handles cases where `shap` or `lime` are not installed:

- **SHAP fallback**: Permutation-importance approximation (slightly less precise)
- **LIME fallback**: Ridge regression on perturbed neighbourhood (built-in)

Install the full libraries for exact results:
```bash
pip install shap lime xgboost
```

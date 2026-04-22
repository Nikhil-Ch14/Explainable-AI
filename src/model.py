"""
model.py
========
Trains and evaluates three classifiers for high-risk financial distress
detection:

  1. Logistic Regression      – interpretable baseline
  2. Random Forest            – ensemble, handles non-linearity
  3. Gradient Boosting (GBM)  – high-accuracy boosted ensemble
                                (drop-in substitute for XGBoost when that
                                 library is unavailable; replace class with
                                 `xgboost.XGBClassifier` if desired)

Class imbalance (~93% / 7%) is handled via `class_weight='balanced'`
and threshold tuning using the ROC curve.

Outputs
-------
  • Trained model objects
  • Per-model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
  • Saved metric summary  →  results/model_metrics.csv
  • ROC curves plot        →  results/plots/roc_curves.png
  • Confusion matrices     →  results/plots/confusion_matrices.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

# Try importing XGBoost; fall back to GradientBoostingClassifier
try:
    from xgboost import XGBClassifier as _XGB
    def build_xgb():
        return _XGB(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=13,   # ~93/7 class ratio
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    XGB_LABEL = "XGBoost"
except ImportError:
    def build_xgb():
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
    XGB_LABEL = "GradientBoosting (XGB proxy)"


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")


# ---------------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------------

def build_models():
    """Return a dict of {label: unfitted model}."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        XGB_LABEL: build_xgb(),
    }
    return models


# ---------------------------------------------------------------------------
# TRAIN & EVALUATE
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, y_train, y_test,
                   threshold: float = 0.30):
    """
    Train model, predict, compute metrics.
    Uses a lower decision threshold (0.30) to improve recall on the
    minority positive class.
    """
    print(f"\n  Training {name} …", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("done.")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "Threshold": threshold,
    }

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    for k, v in metrics.items():
        if k not in ("Model", "Threshold"):
            print(f"    {k:<12} {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Risk','High Risk'])}")

    return metrics, y_proba


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models, collect metrics, generate plots.

    Returns
    -------
    trained_models : dict  {name: fitted model}
    metrics_df     : DataFrame of all metrics
    proba_dict     : dict  {name: y_proba array}
    best_model_name: str
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    models      = build_models()
    all_metrics = []
    proba_dict  = {}
    trained     = {}

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        metrics, y_proba = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        all_metrics.append(metrics)
        proba_dict[name]  = y_proba
        trained[name]     = model

    metrics_df = pd.DataFrame(all_metrics).set_index("Model")
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"))
    print("\n  Metrics saved → results/model_metrics.csv")

    # Best model by ROC-AUC
    best_name = metrics_df["ROC-AUC"].idxmax()
    print(f"\n  ★ Best model: {best_name}  "
          f"(ROC-AUC = {metrics_df.loc[best_name,'ROC-AUC']:.4f})")

    # --- Plots ---
    _plot_roc_curves(trained, proba_dict, y_test, metrics_df)
    _plot_confusion_matrices(trained, proba_dict, y_test)

    return trained, metrics_df, proba_dict, best_name


# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------

def _plot_roc_curves(models, proba_dict, y_test, metrics_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for (name, _), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba_dict[name])
        auc = metrics_df.loc[name, "ROC-AUC"]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – High-Risk Detection Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def _plot_confusion_matrices(models, proba_dict, y_test, threshold=0.30):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, _) in zip(axes, models.items()):
        y_pred = (proba_dict[name] >= threshold).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Risk", "High Risk"],
                    yticklabels=["No Risk", "High Risk"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices (threshold=0.30)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def plot_feature_importance_sklearn(model, feature_names, model_name="RandomForest"):
    """Bar chart of built-in feature importances for tree models."""
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} has no built-in feature_importances_; skipping.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importances)))
    ax.bar(range(len(importances)), importances[idx], color=colors[idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} – Feature Importances", fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{model_name.replace(' ','_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess

    X_tr, X_te, y_tr, y_te, feats, sc, df = preprocess("../data/dataset.arff")
    trained, metrics_df, proba_dict, best = train_and_evaluate(
        X_tr, X_te, y_tr, y_te, feats
    )
    print("\nMetrics:\n", metrics_df.round(4))

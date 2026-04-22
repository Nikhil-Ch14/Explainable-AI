"""
main.py
=======
Orchestrates the full Explainable AI for High-Risk Detection pipeline:

  1. Preprocessing      → data/ → preprocessed arrays
  2. EDA                → results/plots/eda_*.png
  3. Model training     → three classifiers, evaluation metrics
  4. Feature importance → built-in tree importances
  5. SHAP explanations  → global + local plots + JSON
  6. LIME explanations  → local plots + JSON
  7. Report generation  → results/report.md

Run:
  python main.py                    # uses default data path
  python main.py --data path/to/file
"""

import os
import sys
import argparse
import time

# Ensure src/ is importable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from preprocessing    import preprocess
from eda              import run_eda
from model            import train_and_evaluate, plot_feature_importance_sklearn
from explainability   import run_shap, run_lime
from report           import generate_report


DEFAULT_DATA = os.path.join(SRC_DIR, "..", "data", "dataset.arff")


def parse_args():
    p = argparse.ArgumentParser(description="XAI High-Risk Detection Pipeline")
    p.add_argument("--data",       default=DEFAULT_DATA,
                   help="Path to ARFF dataset")
    p.add_argument("--shap-n",     type=int, default=800,
                   help="Rows of X_test to explain with SHAP")
    p.add_argument("--lime-n",     type=int, default=1000,
                   help="LIME perturbation samples")
    p.add_argument("--no-shap",    action="store_true")
    p.add_argument("--no-lime",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "█" * 60)
    print("  EXPLAINABLE AI FOR HIGH-RISK DETECTION")
    print("  Give-Me-Some-Credit – Financial Distress Dataset")
    print("█" * 60)

    # ------------------------------------------------------------------ #
    # 1. PREPROCESSING                                                     #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, feature_names, scaler, df_clean = \
        preprocess(args.data, scale=True)

    # ------------------------------------------------------------------ #
    # 2. EDA                                                               #
    # ------------------------------------------------------------------ #
    run_eda(df_clean, feature_names)

    # ------------------------------------------------------------------ #
    # 3. MODEL TRAINING & EVALUATION                                       #
    # ------------------------------------------------------------------ #
    trained_models, metrics_df, proba_dict, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # ------------------------------------------------------------------ #
    # 4. BUILT-IN FEATURE IMPORTANCE (tree models)                        #
    # ------------------------------------------------------------------ #
    print("\n  Plotting built-in feature importances …")
    for name, model in trained_models.items():
        plot_feature_importance_sklearn(model, feature_names, name)

    # ------------------------------------------------------------------ #
    # 5. SHAP EXPLANATIONS  (best model + logistic regression)            #
    # ------------------------------------------------------------------ #
    shap_results = {}
    if not args.no_shap:
        for name in [best_name, "LogisticRegression"]:
            if name in trained_models:
                sv = run_shap(
                    trained_models[name],
                    X_train, X_test,
                    feature_names, name,
                    n_explain=args.shap_n,
                )
                shap_results[name] = sv

    # ------------------------------------------------------------------ #
    # 6. LIME EXPLANATIONS                                                 #
    # ------------------------------------------------------------------ #
    lime_results = {}
    if not args.no_lime:
        for name in [best_name]:
            if name in trained_models:
                exps = run_lime(
                    trained_models[name],
                    X_train, X_test,
                    feature_names, name,
                    n_samples=args.lime_n,
                )
                lime_results[name] = exps

    # ------------------------------------------------------------------ #
    # 7. REPORT                                                            #
    # ------------------------------------------------------------------ #
    generate_report(
        metrics_df     = metrics_df,
        feature_names  = feature_names,
        best_name      = best_name,
        shap_results   = shap_results,
        lime_results   = lime_results,
        df_clean       = df_clean,
    )

    elapsed = time.time() - t0
    print(f"\n{'█'*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results in:  results/")
    print(f"{'█'*60}\n")


if __name__ == "__main__":
    main()

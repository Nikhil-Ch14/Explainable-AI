"""
explainability.py
=================
Provides SHAP and LIME explanations for the trained models.

SHAP (SHapley Additive exPlanations)
  • Uses shap.TreeExplainer for tree models (RF, XGB/GBM)
  • Uses shap.LinearExplainer for Logistic Regression
  • Generates:
      - Summary plot      (global feature importance)
      - Bar importance    (mean |SHAP|)
      - Force plot        (individual prediction – saved as HTML & PNG)
      - Waterfall plot    (single-instance breakdown)

LIME (Local Interpretable Model-agnostic Explanations)
  • Works with any black-box model via predict_proba
  • Generates tabular explanations for selected instances
  • Saves bar charts for each explained instance

All outputs → results/plots/ and results/explanations/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
EXPL_DIR    = os.path.join(RESULTS_DIR, "explanations")


# ---------------------------------------------------------------------------
# SHAP EXPLANATIONS
# ---------------------------------------------------------------------------

def run_shap(model, X_train, X_test, feature_names, model_name,
             n_background: int = 500, n_explain: int = 1000):
    """
    Compute SHAP values and generate all SHAP plots.

    Parameters
    ----------
    model        : fitted sklearn / xgboost model
    X_train      : training data (numpy array) for background distribution
    X_test       : test data to explain
    feature_names: list of str
    model_name   : str  (used in filenames / titles)
    n_background : rows sampled from X_train as background (KernelExplainer)
    n_explain    : rows of X_test to explain

    Returns
    -------
    shap_values  : numpy array  (n_explain × n_features)
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(EXPL_DIR,  exist_ok=True)

    try:
        import shap
    except ImportError:
        print("\n  [SHAP] shap not installed — running built-in permutation "
              "importance fallback.\n  Install with:  pip install shap")
        shap_values = _permutation_shap_fallback(
            model, X_test[:n_explain], feature_names, model_name
        )
        return shap_values

    print(f"\n  [SHAP] Computing SHAP values for {model_name} …", flush=True)

    X_bg   = X_train[:n_background]
    X_expl = X_test[:n_explain]

    # Choose fastest explainer based on model type
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                      "XGBClassifier", "ExtraTreesClassifier"):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_expl)
        # RF returns list [class0, class1]; take class-1 values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = (explainer.expected_value[1]
                      if hasattr(explainer.expected_value, "__len__")
                      else explainer.expected_value)
    elif model_type == "LogisticRegression":
        explainer   = shap.LinearExplainer(model, X_bg)
        shap_values = explainer.shap_values(X_expl)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = explainer.expected_value
        if hasattr(base_value, "__len__"):
            base_value = base_value[1]
    else:
        # Kernel explainer – slow but universal
        print("    (using KernelExplainer – may take a few minutes)")
        predict_fn  = lambda x: model.predict_proba(x)[:, 1]
        explainer   = shap.KernelExplainer(predict_fn, X_bg[:100])
        shap_values = explainer.shap_values(X_expl[:200])
        base_value  = explainer.expected_value

    print(f"    SHAP values shape: {shap_values.shape}")

    # ---- Plot 1: Summary (beeswarm) ----
    _shap_summary_plot(shap_values, X_expl, feature_names, model_name)

    # ---- Plot 2: Global bar importance ----
    _shap_bar_importance(shap_values, feature_names, model_name)

    # ---- Plot 3: Force plot for top high-risk instance ----
    # Find instance with highest predicted risk
    proba = model.predict_proba(X_expl)[:, 1]
    top_idx = int(np.argmax(proba))
    _shap_force_plot(
        shap_values[top_idx], X_expl[top_idx],
        feature_names, base_value, model_name,
        instance_label=f"Highest-Risk Instance (idx={top_idx})"
    )

    # ---- Plot 4: Waterfall for a low-risk instance ----
    low_idx = int(np.argmin(proba))
    _shap_waterfall_plot(
        shap_values[low_idx], X_expl[low_idx],
        feature_names, base_value, model_name,
        instance_label=f"Lowest-Risk Instance (idx={low_idx})"
    )

    # ---- Save SHAP values as CSV ----
    sv_df = pd.DataFrame(shap_values, columns=feature_names)
    sv_df.to_csv(os.path.join(EXPL_DIR, f"shap_values_{model_name}.csv"),
                 index=False)
    print(f"    SHAP values saved → results/explanations/shap_values_{model_name}.csv")

    return shap_values


def _shap_summary_plot(shap_values, X, feature_names, model_name):
    """Beeswarm-style summary: each dot = one sample, colour = feature value."""
    n_features = len(feature_names)
    mean_abs   = np.abs(shap_values).mean(axis=0)
    order      = np.argsort(mean_abs)          # ascending → bottom to top

    fig, ax = plt.subplots(figsize=(10, 7))

    # Normalize feature values to [0,1] for colouring
    X_norm = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9)
    cmap   = plt.cm.RdBu_r

    for rank, feat_idx in enumerate(order):
        sv   = shap_values[:, feat_idx]
        fval = X_norm[:, feat_idx]
        # Jitter on y-axis for beeswarm effect
        jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
        sc = ax.scatter(sv, rank + jitter, c=fval, cmap=cmap,
                        alpha=0.4, s=4, vmin=0, vmax=1, rasterized=True)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=10)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=11)
    ax.set_title(f"SHAP Summary Plot – {model_name}", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Feature value\n(low → high)", fontsize=9)
    low_patch  = mpatches.Patch(color=cmap(0.0), label="Low value")
    high_patch = mpatches.Patch(color=cmap(1.0), label="High value")
    ax.legend(handles=[low_patch, high_patch], loc="lower right", fontsize=9)

    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved → {path}")


def _shap_bar_importance(shap_values, feature_names, model_name):
    """Global bar chart: mean |SHAP| per feature."""
    mean_abs  = np.abs(shap_values).mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("Reds_r", len(feature_names))
    ax.barh(range(len(feature_names)),
            mean_abs[order][::-1],
            color=palette[:len(feature_names)])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance – {model_name}",
                 fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"shap_importance_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot saved → {path}")


def _shap_force_plot(sv_row, x_row, feature_names, base_value,
                     model_name, instance_label="Instance"):
    """
    Horizontal waterfall (force-plot style) for one prediction.
    Red bars = push prediction UP (toward high risk).
    Blue bars = push prediction DOWN (toward low risk).
    """
    final_value = base_value + sv_row.sum()

    # Sort by absolute contribution
    order = np.argsort(np.abs(sv_row))[::-1][:10]   # top-10 features

    labels = [f"{feature_names[i]}\n= {x_row[i]:.2f}" for i in order]
    values = sv_row[order]
    colors = ["#EF5350" if v > 0 else "#42A5F5" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(values)), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP contribution", fontsize=11)
    ax.set_title(
        f"SHAP Force Plot – {model_name}\n"
        f"{instance_label}  |  Base={base_value:.3f}  →  Prediction={final_value:.3f}",
        fontsize=11, fontweight="bold"
    )
    red_p  = mpatches.Patch(color="#EF5350", label="Increases risk (↑)")
    blue_p = mpatches.Patch(color="#42A5F5", label="Decreases risk (↓)")
    ax.legend(handles=[red_p, blue_p], fontsize=9, loc="lower right")
    sns.despine()
    fig.tight_layout()
    safe_label = instance_label.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(PLOTS_DIR, f"shap_force_{model_name}_{safe_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot saved → {path}")

    # Save as JSON for programmatic use
    expl = {
        "model":       model_name,
        "instance":    instance_label,
        "base_value":  float(base_value),
        "final_value": float(final_value),
        "features": {
            feature_names[i]: {"shap": float(sv_row[i]), "value": float(x_row[i])}
            for i in order
        }
    }
    jpath = os.path.join(EXPL_DIR,
                         f"force_{model_name}_{safe_label}.json")
    with open(jpath, "w") as f:
        json.dump(expl, f, indent=2)
    print(f"    JSON saved  → {jpath}")


def _shap_waterfall_plot(sv_row, x_row, feature_names, base_value,
                         model_name, instance_label="Instance"):
    """Cumulative waterfall from base value to final prediction."""
    # Show top-8 features, collapse rest into "other"
    order     = np.argsort(np.abs(sv_row))[::-1]
    top_n     = 8
    top_idx   = order[:top_n]
    other_sum = sv_row[order[top_n:]].sum()

    labels = [feature_names[i] for i in top_idx] + ["(other)"]
    values = list(sv_row[top_idx]) + [other_sum]

    # Cumulative running total for waterfall
    running = base_value
    lefts, widths, colors = [], [], []
    for v in values:
        lefts.append(running if v > 0 else running + v)
        widths.append(abs(v))
        colors.append("#EF5350" if v > 0 else "#42A5F5")
        running += v

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(values)), widths, left=lefts, color=colors,
            edgecolor="white", height=0.65)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(
        [f"{l}\n= {x_row[top_idx[i]]:.2f}" if i < top_n else l
         for i, l in enumerate(labels)],
        fontsize=9
    )
    ax.axvline(base_value, color="gray", lw=1.2, ls=":", label=f"Base={base_value:.3f}")
    ax.axvline(running,    color="black", lw=1.5, ls="-", label=f"Pred ={running:.3f}")
    ax.set_xlabel("Model output (log-odds)", fontsize=11)
    ax.set_title(f"SHAP Waterfall – {model_name}\n{instance_label}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine()
    fig.tight_layout()
    safe = instance_label.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(PLOTS_DIR, f"shap_waterfall_{model_name}_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot saved → {path}")


# ---------------------------------------------------------------------------
# PERMUTATION-BASED SHAP FALLBACK  (when shap library not installed)
# ---------------------------------------------------------------------------

def _permutation_shap_fallback(model, X, feature_names, model_name,
                                n_repeats: int = 10):
    """
    Approximate SHAP values via permutation importance.
    For each feature, randomly shuffle that column and measure the drop in
    predicted probability → proxy for SHAP contribution.

    This is less precise than true SHAP but works without the shap library.
    """
    print("    Using permutation-importance fallback for SHAP …")
    rng      = np.random.default_rng(42)
    base_proba = model.predict_proba(X)[:, 1]
    n_samples, n_features = X.shape
    shap_approx = np.zeros((n_samples, n_features))

    for fi in range(n_features):
        diffs = np.zeros(n_samples)
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            perm_proba = model.predict_proba(X_perm)[:, 1]
            diffs += base_proba - perm_proba
        shap_approx[:, fi] = diffs / n_repeats

    print(f"    Permutation SHAP shape: {shap_approx.shape}")

    # Still generate plots using the same helpers
    _shap_summary_plot(shap_approx, X, feature_names,
                       model_name + "_PermSHAP")
    _shap_bar_importance(shap_approx, feature_names,
                         model_name + "_PermSHAP")

    proba    = model.predict_proba(X)[:, 1]
    top_idx  = int(np.argmax(proba))
    base_val = float(base_proba.mean())
    _shap_force_plot(shap_approx[top_idx], X[top_idx], feature_names,
                     base_val, model_name + "_PermSHAP",
                     "Highest-Risk_Instance")

    sv_df = pd.DataFrame(shap_approx, columns=feature_names)
    sv_df.to_csv(
        os.path.join(EXPL_DIR,
                     f"shap_values_{model_name}_PermSHAP.csv"), index=False
    )
    return shap_approx


# ---------------------------------------------------------------------------
# LIME EXPLANATIONS
# ---------------------------------------------------------------------------

def run_lime(model, X_train, X_test, feature_names, model_name,
             instance_indices=None, n_features: int = 10,
             n_samples: int = 1000):
    """
    Generate LIME local explanations for selected instances.

    Uses lime.lime_tabular.LimeTabularExplainer when available,
    otherwise falls back to a built-in local linear approximation.

    Parameters
    ----------
    instance_indices : list of int indices into X_test to explain.
                       Defaults to [highest-risk, second-highest, lowest-risk].
    n_features       : number of top features to show in explanation
    n_samples        : number of perturbation samples for LIME

    Returns
    -------
    explanations : list of explanation dicts
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(EXPL_DIR,  exist_ok=True)

    proba = model.predict_proba(X_test)[:, 1]
    if instance_indices is None:
        sorted_idx     = np.argsort(proba)[::-1]
        instance_indices = [sorted_idx[0], sorted_idx[1],
                            int(np.argmin(proba))]

    print(f"\n  [LIME] Explaining {len(instance_indices)} instances for {model_name} …")

    try:
        from lime import lime_tabular
        explainer = lime_tabular.LimeTabularExplainer(
            training_data  = X_train,
            feature_names  = feature_names,
            class_names    = ["No Risk", "High Risk"],
            mode           = "classification",
            discretize_continuous = True,
            random_state   = 42,
        )
        use_lime_lib = True
    except ImportError:
        print("    lime not installed — using built-in linear fallback.")
        use_lime_lib = False

    explanations = []
    for idx in instance_indices:
        instance = X_test[idx]
        pred_p   = proba[idx]
        pred_cls = "High Risk" if pred_p >= 0.30 else "No Risk"
        label    = f"Instance_{idx}_pred={pred_cls}_{pred_p:.2f}"

        print(f"    Explaining {label} …")

        if use_lime_lib:
            exp = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features = n_features,
                num_samples  = n_samples,
                labels       = (1,),
            )
            feature_weights = exp.as_list(label=1)
        else:
            feature_weights = _lime_fallback(
                model, X_train, instance, feature_names,
                n_features, n_samples
            )

        # Plot
        _plot_lime_explanation(
            feature_weights, label, model_name, pred_p, pred_cls
        )

        # Save JSON
        expl_dict = {
            "model":       model_name,
            "instance_idx": int(idx),
            "predicted_proba": float(pred_p),
            "predicted_class": pred_cls,
            "explanation": [
                {"feature": f, "weight": w} for f, w in feature_weights
            ]
        }
        jpath = os.path.join(EXPL_DIR,
                             f"lime_{model_name}_{idx}.json")
        with open(jpath, "w") as f:
            json.dump(expl_dict, f, indent=2)
        print(f"    JSON saved → {jpath}")
        explanations.append(expl_dict)

    return explanations


def _lime_fallback(model, X_train, instance, feature_names,
                   n_features, n_samples):
    """
    Built-in LIME-like local explanation via perturbed neighbourhood.
    1. Sample perturbations around the instance (Gaussian noise).
    2. Get model predictions on perturbed samples.
    3. Fit weighted linear regression in the neighbourhood.
    4. Return coefficient magnitude as feature weights.
    """
    from sklearn.linear_model import Ridge

    rng   = np.random.default_rng(42)
    sigma = X_train.std(axis=0) + 1e-9
    noise = rng.normal(0, 1, size=(n_samples, len(instance))) * sigma
    perturbed = instance + noise

    # Distance-based kernel weights (closer = higher weight)
    dists   = np.sqrt(((perturbed - instance) ** 2).sum(axis=1))
    kernel  = np.exp(-(dists ** 2) / (2 * (dists.std() + 1e-9) ** 2))

    preds = model.predict_proba(perturbed)[:, 1]

    # Fit local linear model
    lr = Ridge(alpha=1.0)
    lr.fit(perturbed, preds, sample_weight=kernel)

    # Return top n_features
    coefs     = lr.coef_
    order     = np.argsort(np.abs(coefs))[::-1][:n_features]
    fw = [(f"{feature_names[i]} = {instance[i]:.2f}", float(coefs[i]))
          for i in order]
    return fw


def _plot_lime_explanation(feature_weights, label, model_name,
                            pred_proba, pred_class):
    """Horizontal bar chart of LIME feature weights."""
    feats   = [fw[0] for fw in feature_weights]
    weights = [fw[1] for fw in feature_weights]
    colors  = ["#EF5350" if w > 0 else "#42A5F5" for w in weights]

    # Sort by absolute weight
    order   = np.argsort(np.abs(weights))
    feats   = [feats[i] for i in order]
    weights = [weights[i] for i in order]
    colors  = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feats) * 0.55)))
    ax.barh(range(len(feats)), weights, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME weight (contribution to High-Risk class)", fontsize=10)
    ax.set_title(
        f"LIME Local Explanation – {model_name}\n"
        f"Prediction: {pred_class}  (P={pred_proba:.3f})",
        fontsize=11, fontweight="bold"
    )
    red_p  = mpatches.Patch(color="#EF5350", label="Increases risk")
    blue_p = mpatches.Patch(color="#42A5F5", label="Decreases risk")
    ax.legend(handles=[red_p, blue_p], fontsize=9)
    sns.despine()
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    path = os.path.join(PLOTS_DIR, f"lime_{model_name}_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Plot saved → {path}")


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import preprocess
    from model import train_and_evaluate

    X_tr, X_te, y_tr, y_te, feats, sc, df = preprocess("../data/dataset.arff")
    trained, metrics_df, proba_dict, best = train_and_evaluate(
        X_tr, X_te, y_tr, y_te, feats
    )
    rf = trained["RandomForest"]
    sv = run_shap(rf, X_tr, X_te, feats, "RandomForest", n_explain=500)
    ex = run_lime(rf, X_tr, X_te, feats, "RandomForest")

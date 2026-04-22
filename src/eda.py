"""
eda.py
======
Exploratory Data Analysis for the Give-Me-Some-Credit dataset.

Generates:
  • Class distribution bar chart
  • Feature distribution histograms (with risk overlay)
  • Correlation heatmap
  • Box plots comparing risk vs no-risk groups
  • Missing-value bar chart

All saved → results/plots/eda_*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")


def run_eda(df: pd.DataFrame, feature_cols: list):
    """Run full EDA suite and save all plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    _print_summary(df, feature_cols)
    _plot_class_distribution(df)
    _plot_missing_values(df, feature_cols)
    _plot_feature_distributions(df, feature_cols)
    _plot_correlation_heatmap(df, feature_cols)
    _plot_risk_boxplots(df, feature_cols)
    print("\n  EDA complete. Plots saved → results/plots/eda_*\n")


def _print_summary(df, feature_cols):
    print(f"\n  Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"  Class balance:")
    vc = df["target"].value_counts()
    for cls, cnt in vc.items():
        print(f"    {cls}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")
    print(f"\n  Descriptive statistics:")
    print(df[feature_cols].describe().round(2).to_string())


def _plot_class_distribution(df):
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["No Risk (0)", "High Risk (1)"], counts.values,
                  color=["#42A5F5", "#EF5350"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Class Distribution – Financial Distress", fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "eda_class_distribution.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Plot → {path}")


def _plot_missing_values(df, feature_cols):
    missing = df[feature_cols].isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    missing.plot.barh(ax=ax, color="#FF7043")
    ax.set_xlabel("Missing rows", fontsize=10)
    ax.set_title("Missing Values per Feature", fontsize=12, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "eda_missing_values.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Plot → {path}")


def _plot_feature_distributions(df, feature_cols):
    n   = len(feature_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for cls, color in [(0, "#42A5F5"), (1, "#EF5350")]:
            data = df[df["target"] == cls][col].dropna()
            # Clip to 99th percentile for readability
            cap = data.quantile(0.99)
            data = data.clip(upper=cap)
            ax.hist(data, bins=40, alpha=0.6, color=color,
                    label="No Risk" if cls == 0 else "High Risk",
                    density=True)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Risk Class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "eda_feature_distributions.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Plot → {path}")


def _plot_correlation_heatmap(df, feature_cols):
    corr_cols = feature_cols + ["target"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "eda_correlation_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Plot → {path}")


def _plot_risk_boxplots(df, feature_cols):
    # Top features by correlation with target
    corr = df[feature_cols + ["target"]].corr()["target"].drop("target")
    top_feats = corr.abs().nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, col in enumerate(top_feats):
        ax = axes[i]
        data_0 = df[df["target"] == 0][col].dropna().clip(upper=df[col].quantile(0.99))
        data_1 = df[df["target"] == 1][col].dropna().clip(upper=df[col].quantile(0.99))
        ax.boxplot([data_0, data_1],
                   labels=["No Risk", "High Risk"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#42A5F5", alpha=0.6),
                   medianprops=dict(color="black", lw=2))
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Value", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top-6 Features: Risk vs No-Risk Distribution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "eda_risk_boxplots.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Plot → {path}")

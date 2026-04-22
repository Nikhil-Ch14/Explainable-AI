"""
preprocessing.py
================
Handles all data loading, cleaning, and feature engineering for the
"Give Me Some Credit" financial-distress detection dataset (ARFF format).

Steps:
  1. Parse ARFF file → pandas DataFrame
  2. Replace '?' with NaN
  3. Impute missing values (median strategy)
  4. Clip extreme outliers (99th-percentile cap)
  5. Scale features (StandardScaler)
  6. Return train/test splits ready for modelling
"""

import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1.  ARFF PARSER
# ---------------------------------------------------------------------------

def load_arff(filepath: str) -> pd.DataFrame:
    """
    Minimal ARFF parser that extracts column names and data rows.
    Returns a raw DataFrame with string values (missing shown as '?').
    """
    columns = []
    data_lines = []
    in_data_section = False

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            # Skip comments and blank lines
            if line.startswith("%") or line == "":
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                # Extract column name (second token)
                parts = line.split()
                columns.append(parts[1])
            elif line.upper().startswith("@DATA"):
                in_data_section = True
            elif in_data_section:
                data_lines.append(line)

    # Parse CSV rows
    records = [row.split(",") for row in data_lines if row]
    df = pd.DataFrame(records, columns=columns)
    return df


# ---------------------------------------------------------------------------
# 2.  CLEAN + ENCODE
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Replace '?' with NaN
    • Cast all columns to numeric
    • Rename columns for readability
    """
    df = df.replace("?", np.nan)

    # Rename long column names
    rename_map = {
        "SeriousDlqin2yrs": "target",
        "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
        "age": "age",
        "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
        "DebtRatio": "debt_ratio",
        "MonthlyIncome": "monthly_income",
        "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
        "NumberOfTimes90DaysLate": "times_90_late",
        "NumberRealEstateLoansOrLines": "real_estate_loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
        "NumberOfDependents": "dependents",
    }
    df = df.rename(columns=rename_map)

    # Convert everything to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 3.  OUTLIER CAPPING
# ---------------------------------------------------------------------------

def cap_outliers(df: pd.DataFrame, feature_cols: list, upper_percentile: float = 99) -> pd.DataFrame:
    """
    Cap each feature at its upper_percentile value to reduce the effect
    of extreme outliers (e.g. DebtRatio = 5710).
    """
    df = df.copy()
    for col in feature_cols:
        cap = np.nanpercentile(df[col], upper_percentile)
        df[col] = df[col].clip(upper=cap)
    return df


# ---------------------------------------------------------------------------
# 4.  FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------

def preprocess(filepath: str,
               test_size: float = 0.20,
               random_state: int = 42,
               scale: bool = True):
    """
    End-to-end preprocessing.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names                     : list of feature column names
    scaler                            : fitted StandardScaler (or None)
    df_clean                          : cleaned DataFrame (for EDA)
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # --- Load ---
    print("\n[1/5] Loading ARFF file …")
    df = load_arff(filepath)
    print(f"      Raw shape: {df.shape}")

    # --- Clean ---
    print("[2/5] Cleaning and type-casting …")
    df = clean_dataframe(df)

    # Drop rows where target is NaN (shouldn't happen, but defensive)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    feature_cols = [c for c in df.columns if c != "target"]

    # --- Missing value report ---
    print("\n      Missing values per column:")
    missing = df[feature_cols].isnull().sum()
    for col, cnt in missing[missing > 0].items():
        pct = cnt / len(df) * 100
        print(f"        {col:<30} {cnt:>6} rows  ({pct:.1f}%)")

    # --- Outlier capping ---
    print("\n[3/5] Capping outliers at 99th percentile …")
    df = cap_outliers(df, feature_cols)

    # --- Imputation ---
    print("[4/5] Imputing missing values (median) …")
    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # --- Train / test split ---
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Scaling ---
    scaler = None
    if scale:
        print("[5/5] Scaling features (StandardScaler) …")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"\n  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")
    class_counts = np.bincount(y_train)
    print(f"  Class balance : 0={class_counts[0]:,}  1={class_counts[1]:,}  "
          f"(positive rate {class_counts[1]/len(y_train)*100:.1f}%)")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, feature_cols, scaler, df


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, sc, df = preprocess(
        "../data/dataset.arff"
    )
    print("\nFeatures:", feats)
    print("X_train sample:\n", X_tr[:3])

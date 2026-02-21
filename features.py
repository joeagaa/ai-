"""
features.py

Simple + robust feature builder:
- expects target column: 'y'
- uses all other columns as features
- converts everything to numeric
- fills missing values
"""

import pandas as pd

TARGET_COL = "y"

def make_features(df: pd.DataFrame):
    # Basic validation
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COL}'. "
            f"Your CSV columns are: {list(df.columns)}"
        )

    # Split X/y
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    if X.shape[1] == 0:
        raise ValueError("No feature columns found. Add columns besides 'y'.")

    # Convert all features to numeric
    # (non-numeric values become NaN, then we fill with 0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Convert target to int if it looks like 0/1
    # (keeps it simple for binary classification)
    try:
        y = pd.to_numeric(y, errors="raise").astype(int)
    except Exception:
        # If y isn't numeric, keep it as-is (but your model might fail later)
        pass

    return X, y

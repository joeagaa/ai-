import pandas as pd

TARGET_COL = "y"

def make_features(df: pd.DataFrame):
    """
    Expects a target column named 'y'.
    Everything else is treated as numeric features.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}' in CSV.")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # force numeric (simple starter)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y

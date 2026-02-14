"""
Feature engineering helpers.

Replace this with your real feature logic later.
"""

import pandas as pd

def make_features(df: pd.DataFrame):
    """
    Expects a target column named 'y' in your dataset.
    Returns X (features) and y (target).
    """
    if "y" not in df.columns:
        raise ValueError("Expected a target column named 'y' in your dataset.")

    y = df["y"]
    X = df.drop(columns=["y"])

    return X, y

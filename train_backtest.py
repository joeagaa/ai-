"""
Train + backtest script (starter).

Expects: data/train.csv with a target column named 'y'
- Saves model to out/models/
- Writes metrics to out/logs/
"""

import os
import json
from datetime import datetime

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from config import PATHS
from features import make_features

def ensure_dirs():
    os.makedirs(PATHS.data_dir, exist_ok=True)
    os.makedirs(PATHS.models_dir, exist_ok=True)
    os.makedirs(PATHS.logs_dir, exist_ok=True)

def main():
    ensure_dirs()

    train_path = os.path.join(PATHS.data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing {train_path}. Put your dataset there (must include target column 'y')."
        )

    df = pd.read_csv(train_path)
    X, y = make_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    try:
        proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else None
    except Exception:
        roc = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc) if roc is not None else None,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(PATHS.models_dir, f"model_{stamp}.joblib")
    dump(model, model_path)

    metrics_path = os.path.join(PATHS.logs_dir, f"backtest_metrics_{stamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()

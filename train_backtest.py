import os
import json
from datetime import datetime

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import PATHS, SETTINGS
from features import make_features

def ensure_dirs():
    os.makedirs(PATHS.data_dir, exist_ok=True)
    os.makedirs(PATHS.models_dir, exist_ok=True)
    os.makedirs(PATHS.logs_dir, exist_ok=True)

def read_train_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV load:
    - strips weird BOM chars
    - uses python engine (more forgiving)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Put your dataset there.")

    # if your CSV is broken, this will still fail, but the error is clearer
    return pd.read_csv(path, engine="python", encoding="utf-8-sig")

def main():
    ensure_dirs()

    df = read_train_csv(PATHS.train_csv)
    X, y = make_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SETTINGS.test_size,
        random_state=SETTINGS.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(PATHS.models_dir, f"model_{stamp}.joblib")
    dump(model, model_path)

    metrics = {
        "accuracy": acc,
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "saved_model": model_path,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }

    metrics_path = os.path.join(PATHS.logs_dir, f"metrics_{stamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete")
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()

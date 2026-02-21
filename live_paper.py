import os
import glob
import pandas as pd
from joblib import load

from config import PATHS
from safety import should_bet, bet_size

def latest_model_path() -> str:
    paths = sorted(glob.glob(os.path.join(PATHS.models_dir, "*.joblib")))
    if not paths:
        raise FileNotFoundError("No model found. Run: python train_backtest.py")
    return paths[-1]

def main():
    os.makedirs(PATHS.models_dir, exist_ok=True)

    model_path = latest_model_path()
    model = load(model_path)
    print("Loaded model:", model_path)

    # Example inputs — replace with your real features later
    X_new = pd.DataFrame([
        {"feature1": 0.2, "feature2": 1.1},
        {"feature1": -0.3, "feature2": 0.4},
        {"feature1": 1.7, "feature2": -0.2},
    ])

    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    probs = model.predict_proba(X_new)[:, 1]

    bankroll = 1000.0
    threshold = 0.55
    max_pct = 0.02

    for i, p in enumerate(probs):
        conf = float(p)
        if should_bet(conf, threshold):
            size = bet_size(bankroll, conf, max_pct)
            print(f"BET  #{i}  confidence={conf:.3f}  size=${size:.2f}")
        else:
            print(f"SKIP #{i}  confidence={conf:.3f}  (< {threshold})")

if __name__ == "__main__":
    main()

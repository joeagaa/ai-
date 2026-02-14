"""
Paper trading / live simulation (starter).

Loads the latest model from out/models/ and shows how you *would* place bets.
You will adapt this to your real odds feed later.
"""

import os
import glob
import pandas as pd
from joblib import load

from config import PATHS, RISK
from safety import bet_size, should_bet

def latest_model_path():
    paths = sorted(glob.glob(os.path.join(PATHS.models_dir, "*.joblib")))
    if not paths:
        raise FileNotFoundError("No saved model found in out/models/. Run train_backtest.py first.")
    return paths[-1]

def main():
    os.makedirs(PATHS.models_dir, exist_ok=True)
    os.makedirs(PATHS.logs_dir, exist_ok=True)

    model_path = latest_model_path()
    model = load(model_path)
    print("Loaded model:", model_path)

    # Example: pretend we have upcoming bets with features already computed
    upcoming = pd.DataFrame([
        {"feature1": 0.2, "feature2": 1.1},
        {"feature1": -0.3, "feature2": 0.4},
        {"feature1": 1.7, "feature2": -0.2},
    ])

    probs = model.predict_proba(upcoming)[:, 1]

    bankroll = 1000.0  # example bankroll
    for i, p in enumerate(probs):
        conf = float(p)
        if should_bet(conf, RISK.min_confidence_to_bet):
            size = bet_size(bankroll, conf, RISK.max_bet_pct_bankroll)
            print(f"Bet #{i}: confidence={conf:.3f} -> bet_size=${size:.2f}")
        else:
            print(f"Skip #{i}: confidence={conf:.3f} < threshold={RISK.min_confidence_to_bet:.2f}")

if __name__ == "__main__":
    main()

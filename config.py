"""
Project configuration
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    data_dir: str = "data"
    out_dir: str = "out"
    models_dir: str = "out/models"
    logs_dir: str = "out/logs"

@dataclass(frozen=True)
class Risk:
    # Tweak these later
    max_bet_pct_bankroll: float = 0.02   # 2% of bankroll per bet
    max_daily_loss_pct: float = 0.05     # stop after 5% daily loss
    min_confidence_to_bet: float = 0.55  # model confidence threshold

PATHS = Paths()
RISK = Risk()

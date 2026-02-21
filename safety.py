def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def should_bet(confidence: float, threshold: float = 0.55) -> bool:
    return confidence >= threshold

def bet_size(bankroll: float, confidence: float, max_pct: float = 0.02) -> float:
    """
    Simple bet sizing: scales bet with confidence, capped at max_pct of bankroll.
    """
    if bankroll <= 0:
        return 0.0

    scaled = clamp((confidence - 0.5) / 0.5, 0.0, 1.0)  # confidence 0.5->0, 1.0->1
    return bankroll * max_pct * scaled

"""
Risk management helpers.
"""

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def bet_size(bankroll: float, confidence: float, max_pct: float) -> float:
    """
    Simple bet sizing:
    scales with confidence and caps at max_pct of bankroll.
    """
    if bankroll <= 0:
        return 0.0

    # Map confidence [0.5..1.0] -> [0..1]
    scaled = clamp((confidence - 0.5) / 0.5, 0.0, 1.0)
    return bankroll * max_pct * scaled

def should_bet(confidence: float, threshold: float) -> bool:
    return confidence >= threshold

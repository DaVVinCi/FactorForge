from __future__ import annotations

import random

from .models import FactorProposal


CLASSIC_FACTORS: dict[str, str] = {
    "momentum_20": "Rank(close / Delay(close, 20) - 1)",
    "reversal_5": "-Rank(close / Delay(close, 5) - 1)",
    "low_volatility_20": "-Rank(Std(close / Delay(close, 1) - 1, 20))",
    "volume_price_corr_20": "-Rank(Corr(close, volume, 20))",
    "volume_expansion": "Rank(Mean(volume, 5) / Mean(volume, 20) - 1)",
    "amihud_liquidity_20": (
        "-Rank(Mean(Abs(close / Delay(close, 1) - 1) / (amount + 1), 20))"
    ),
    "overnight_gap": "Rank(open / Delay(close, 1) - 1)",
}


RANDOM_TEMPLATES = [
    "Rank(Delta(volume, {short})) - Rank(Std(close, {long}))",
    "Rank(Corr(vwap, volume, {long})) + Rank(Delta(close, {short}))",
    "Rank(Mean(high - low, {short}) / Mean(close, {long}))",
    "Rank(Delta(amount, {short}) / (Mean(amount, {long}) + 1))",
]


def random_factor(seed: int = 7) -> FactorProposal:
    """Generate a reproducible random baseline, not a financially informed factor."""
    rng = random.Random(seed)
    template = rng.choice(RANDOM_TEMPLATES)
    short = rng.choice([2, 3, 5, 7])
    long = rng.choice([10, 15, 20, 30])
    expression = template.format(short=short, long=long)
    return FactorProposal(
        name=f"random_baseline_s{seed}",
        hypothesis=(
            "A seeded random combination of allowed operators is used only as a "
            "control group; it does not encode an economic hypothesis."
        ),
        economic_rationale=(
            "None. This baseline estimates how easily a mechanically generated "
            "expression can appear interesting on the same sample."
        ),
        expression=expression,
        expected_direction="positive",
        data_requirements=["OHLCV", "amount"],
        risk_notes=[
            "No economic rationale",
            "In-sample chance discovery",
            "Seed-sensitive expression",
        ],
        iteration_note="Seeded random control; not produced by an LLM.",
    )


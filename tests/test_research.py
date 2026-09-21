import numpy as np

from factorforge.data import generate_demo_data
from factorforge.research import FactorResearcher, ResearchSettings


def test_research_metrics_and_cost_accounting():
    data = generate_demo_data(periods=90, symbols=12, seed=3)
    researcher = FactorResearcher(
        data,
        ResearchSettings(
            quantiles=5, min_cross_section=8, top_k=3, fee_bps=15
        ),
    )
    result = researcher.evaluate(
        "test_momentum",
        "test",
        "Rank(close / Delay(close, 10) - 1)",
    )
    assert {"ic_mean", "rank_ic_mean", "icir", "rank_icir"}.issubset(
        result.metrics
    )
    assert {"net_value", "gross_value", "drawdown", "turnover"}.issubset(
        result.equity_curve.columns
    )
    assert result.metrics["total_cost"] > 0
    assert result.equity_curve["cost"].ge(0).all()
    assert np.isfinite(result.metrics["max_drawdown"])
    assert "long_short" in result.quantile_returns


def test_signal_at_t_uses_forward_return():
    data = generate_demo_data(periods=60, symbols=10, seed=4)
    researcher = FactorResearcher(data)
    expected = (
        data["close"].groupby(level="symbol").shift(-1) / data["close"] - 1
    )
    assert researcher.forward_returns.equals(expected.rename("forward_return"))


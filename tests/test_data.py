import pandas as pd

from factorforge.data import generate_demo_data


def test_demo_data_is_deterministic_and_explicitly_simulated():
    first = generate_demo_data(periods=60, symbols=8, seed=9)
    second = generate_demo_data(periods=60, symbols=8, seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 480
    assert first["is_simulated"].all()
    assert all(symbol.endswith(".SIM") for symbol in first.index.levels[1])


def test_demo_ohlc_invariants():
    data = generate_demo_data(periods=50, symbols=5)
    assert (data["high"] >= data[["open", "close"]].max(axis=1)).all()
    assert (data["low"] <= data[["open", "close"]].min(axis=1)).all()
    assert (data[["close", "volume", "amount"]] > 0).all().all()


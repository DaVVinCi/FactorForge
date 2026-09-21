import numpy as np
import pytest

from factorforge.data import generate_demo_data
from factorforge.dsl import ExpressionEngine, ExpressionValidationError


@pytest.fixture()
def engine():
    return ExpressionEngine()


@pytest.fixture()
def data():
    return generate_demo_data(periods=70, symbols=8)


def test_required_operators_evaluate(engine, data):
    expressions = [
        "Rank(close)",
        "Delay(close, 2)",
        "Delta(volume, 3)",
        "Mean(close, 5)",
        "Std(close, 5)",
        "Corr(close, volume, 10)",
    ]
    for expression in expressions:
        result = engine.evaluate(expression, data)
        assert len(result) == len(data)
        assert not np.isinf(result.dropna()).any()


@pytest.mark.parametrize(
    "expression",
    [
        "Delay(close, -1)",
        "Delay(close, 0)",
        "Mean(close, 253)",
        "future_return",
        "__import__('os')",
        "close ** 2",
        "close[0]",
        "Mean(close)",
    ],
)
def test_invalid_or_lookahead_expressions_are_rejected(engine, expression):
    with pytest.raises(ExpressionValidationError):
        engine.validate(expression)


def test_division_by_zero_becomes_nan_not_infinity(engine, data):
    result = engine.evaluate("close / (volume - volume)", data)
    assert result.isna().all()


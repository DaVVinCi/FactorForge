from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


class ExpressionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class FunctionSpec:
    arity: int
    window_positions: tuple[int, ...] = ()


FUNCTIONS: dict[str, FunctionSpec] = {
    "Rank": FunctionSpec(1),
    "Delay": FunctionSpec(2, (1,)),
    "Delta": FunctionSpec(2, (1,)),
    "Mean": FunctionSpec(2, (1,)),
    "Std": FunctionSpec(2, (1,)),
    "Sum": FunctionSpec(2, (1,)),
    "Min": FunctionSpec(2, (1,)),
    "Max": FunctionSpec(2, (1,)),
    "Corr": FunctionSpec(3, (2,)),
    "Abs": FunctionSpec(1),
    "Sign": FunctionSpec(1),
}
FIELDS = {"open", "high", "low", "close", "volume", "amount", "vwap"}
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class _Validator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, ALLOWED_BINOPS):
            raise ExpressionValidationError(
                f"operator {type(node.op).__name__} is not allowed"
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            raise ExpressionValidationError("only unary +/- are allowed")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise ExpressionValidationError("function is not on the DSL whitelist")
        if node.keywords:
            raise ExpressionValidationError("keyword arguments are not allowed")
        spec = FUNCTIONS[node.func.id]
        if len(node.args) != spec.arity:
            raise ExpressionValidationError(
                f"{node.func.id} expects {spec.arity} arguments"
            )
        for position in spec.window_positions:
            arg = node.args[position]
            if (
                not isinstance(arg, ast.Constant)
                or isinstance(arg.value, bool)
                or not isinstance(arg.value, int)
            ):
                raise ExpressionValidationError(
                    f"{node.func.id} window/lag must be an integer literal"
                )
            if not 1 <= arg.value <= 252:
                raise ExpressionValidationError(
                    f"{node.func.id} window/lag must be between 1 and 252"
                )
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in FIELDS:
            raise ExpressionValidationError(f"field {node.id!r} is not allowed")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("only numeric constants are allowed")
        if not np.isfinite(float(node.value)) or abs(float(node.value)) > 1e12:
            raise ExpressionValidationError("numeric constant is invalid or too large")

    def generic_visit(self, node: ast.AST) -> None:
        raise ExpressionValidationError(
            f"syntax element {type(node).__name__} is not allowed"
        )


class ExpressionEngine:
    """Validate and evaluate a compact, point-in-time factor DSL."""

    def __init__(self, zero_epsilon: float = 1e-12):
        self.zero_epsilon = zero_epsilon

    def validate(self, expression: str) -> ast.Expression:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ExpressionValidationError(f"invalid syntax: {exc.msg}") from exc
        _Validator().visit(tree)
        return tree

    def evaluate(self, expression: str, data: pd.DataFrame) -> pd.Series:
        tree = self.validate(expression)
        missing = FIELDS.intersection(_field_names(tree)).difference(data.columns)
        if missing:
            raise ExpressionValidationError(
                f"data does not provide fields: {sorted(missing)}"
            )
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("data must use a (date, symbol) MultiIndex")
        result = self._eval(tree.body, data)
        if np.isscalar(result):
            result = pd.Series(float(result), index=data.index)
        return (
            pd.Series(result, index=data.index, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .rename("factor")
        )

    def _eval(self, node: ast.AST, data: pd.DataFrame):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return data[node.id].astype(float)
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, data)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left, data), self._eval(node.right, data)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            denominator = right
            if np.isscalar(denominator):
                if abs(float(denominator)) < self.zero_epsilon:
                    return pd.Series(np.nan, index=data.index)
            else:
                denominator = denominator.mask(
                    denominator.abs() < self.zero_epsilon
                )
            return left / denominator
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [self._eval(arg, data) for arg in node.args]
            return self._call(name, args, data.index)
        raise ExpressionValidationError(f"cannot evaluate {type(node).__name__}")

    def _call(self, name: str, args: list, index: pd.MultiIndex):
        series = _as_series(args[0], index)
        if name == "Rank":
            return series.groupby(level="date").rank(pct=True, method="average")
        if name == "Delay":
            return series.groupby(level="symbol").shift(int(args[1]))
        if name == "Delta":
            return series - series.groupby(level="symbol").shift(int(args[1]))
        if name in {"Mean", "Std", "Sum", "Min", "Max"}:
            window = int(args[1])
            method: Callable = {
                "Mean": lambda x: x.mean(),
                "Std": lambda x: x.std(ddof=0),
                "Sum": lambda x: x.sum(),
                "Min": lambda x: x.min(),
                "Max": lambda x: x.max(),
            }[name]
            return series.groupby(level="symbol").transform(
                lambda values: values.rolling(
                    window, min_periods=window
                ).apply(method, raw=False)
            )
        if name == "Corr":
            other = _as_series(args[1], index)
            window = int(args[2])
            return _rolling_corr(series, other, window)
        if name == "Abs":
            return series.abs()
        if name == "Sign":
            return np.sign(series)
        raise ExpressionValidationError(f"unsupported function {name}")


def _rolling_corr(
    left: pd.Series, right: pd.Series, window: int
) -> pd.Series:
    output = pd.Series(np.nan, index=left.index, dtype=float)
    symbols = left.index.get_level_values("symbol").unique()
    for symbol in symbols:
        left_group = left.xs(symbol, level="symbol")
        right_group = right.xs(symbol, level="symbol")
        corr = left_group.rolling(window, min_periods=window).corr(right_group)
        locations = left.index.get_level_values("symbol") == symbol
        output.loc[locations] = corr.to_numpy()
    return output


def _as_series(value, index: pd.MultiIndex) -> pd.Series:
    if np.isscalar(value):
        return pd.Series(float(value), index=index)
    return pd.Series(value, index=index, dtype=float)


def _field_names(tree: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in FIELDS
    }

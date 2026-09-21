from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dsl import ExpressionEngine
from .models import EvaluationResult, FactorProposal


@dataclass
class ResearchSettings:
    forward_periods: int = 1
    quantiles: int = 5
    min_cross_section: int = 8
    top_k: int = 5
    fee_bps: float = 10.0
    annualization: int = 252


class FactorResearcher:
    def __init__(
        self,
        data: pd.DataFrame,
        settings: ResearchSettings | None = None,
        engine: ExpressionEngine | None = None,
    ):
        self.data = data.sort_index()
        self.settings = settings or ResearchSettings()
        self.engine = engine or ExpressionEngine()
        self.forward_returns = self._forward_returns()

    def evaluate(
        self,
        name: str,
        category: str,
        expression: str,
        proposal: FactorProposal | None = None,
    ) -> EvaluationResult:
        factor = self.engine.evaluate(expression, self.data)
        daily_ic = self._daily_ic(factor)
        quantile_returns = self._quantile_returns(factor)
        equity_curve, backtest_metrics = self._backtest(factor)

        pearson = daily_ic["ic"].dropna()
        spearman = daily_ic["rank_ic"].dropna()
        metrics = {
            "coverage": float(factor.notna().mean()),
            "ic_mean": float(pearson.mean()) if not pearson.empty else np.nan,
            "rank_ic_mean": (
                float(spearman.mean()) if not spearman.empty else np.nan
            ),
            "icir": _icir(pearson, self.settings.annualization),
            "rank_icir": _icir(spearman, self.settings.annualization),
            "ic_positive_rate": (
                float((pearson > 0).mean()) if not pearson.empty else np.nan
            ),
            **backtest_metrics,
        }
        warnings: list[str] = []
        if metrics["coverage"] < 0.7:
            warnings.append("Factor coverage is below 70%.")
        if len(spearman) < 30:
            warnings.append("Fewer than 30 valid cross-sectional IC observations.")
        return EvaluationResult(
            name=name,
            category=category,
            expression=expression,
            metrics=metrics,
            daily_ic=daily_ic,
            quantile_returns=quantile_returns,
            equity_curve=equity_curve,
            proposal=proposal,
            warnings=warnings,
        )

    def _forward_returns(self) -> pd.Series:
        close = self.data["close"].astype(float)
        horizon = self.settings.forward_periods
        future_close = close.groupby(level="symbol").shift(-horizon)
        return (future_close / close - 1.0).rename("forward_return")

    def _daily_ic(self, factor: pd.Series) -> pd.DataFrame:
        joined = pd.concat([factor, self.forward_returns], axis=1).dropna()
        rows: list[dict] = []
        for date, group in joined.groupby(level="date"):
            if (
                len(group) < self.settings.min_cross_section
                or group["factor"].nunique() < 2
                or group["forward_return"].nunique() < 2
            ):
                continue
            rows.append(
                {
                    "date": date,
                    "ic": group["factor"].corr(
                        group["forward_return"], method="pearson"
                    ),
                    "rank_ic": group["factor"].corr(
                        group["forward_return"], method="spearman"
                    ),
                    "n": len(group),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["ic", "rank_ic", "n"]).rename_axis("date")
        return pd.DataFrame(rows).set_index("date").sort_index()

    def _quantile_returns(self, factor: pd.Series) -> pd.DataFrame:
        joined = pd.concat([factor, self.forward_returns], axis=1).dropna()
        records: list[dict] = []
        quantiles = self.settings.quantiles
        for date, group in joined.groupby(level="date"):
            if len(group) < max(self.settings.min_cross_section, quantiles):
                continue
            ranked = group["factor"].rank(method="first")
            try:
                buckets = pd.qcut(ranked, quantiles, labels=False) + 1
            except ValueError:
                continue
            means = group.assign(quantile=buckets).groupby("quantile")[
                "forward_return"
            ].mean()
            row = {"date": date}
            row.update({f"Q{int(q)}": value for q, value in means.items()})
            records.append(row)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records).set_index("date").sort_index()
        low, high = "Q1", f"Q{quantiles}"
        if low in frame and high in frame:
            frame["long_short"] = frame[high] - frame[low]
        return frame

    def _backtest(
        self, factor: pd.Series
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        signals = factor.unstack("symbol")
        returns = self.forward_returns.unstack("symbol").reindex(signals.index)
        previous = pd.Series(0.0, index=signals.columns)
        records: list[dict] = []
        for position, date in enumerate(signals.index):
            available = signals.loc[date].dropna()
            valid_returns = returns.loc[date].dropna()
            candidates = available.index.intersection(valid_returns.index)
            weights = pd.Series(0.0, index=signals.columns)
            if len(candidates):
                selected = available.loc[candidates].nlargest(
                    min(self.settings.top_k, len(candidates))
                ).index
                weights.loc[selected] = 1.0 / len(selected)
            if position == 0:
                turnover = float(weights.abs().sum())
            else:
                turnover = float((weights - previous).abs().sum() / 2.0)
            gross = float(
                (weights * returns.loc[date].fillna(0.0)).sum()
            )
            cost = turnover * self.settings.fee_bps / 10_000.0
            benchmark = (
                float(valid_returns.mean()) if len(valid_returns) else 0.0
            )
            records.append(
                {
                    "date": date,
                    "gross_return": gross,
                    "cost": cost,
                    "net_return": gross - cost,
                    "benchmark_return": benchmark,
                    "turnover": turnover,
                    "holdings": int((weights > 0).sum()),
                }
            )
            previous = weights
        curve = pd.DataFrame(records).set_index("date")
        curve["net_value"] = (1.0 + curve["net_return"]).cumprod()
        curve["gross_value"] = (1.0 + curve["gross_return"]).cumprod()
        curve["benchmark_value"] = (1.0 + curve["benchmark_return"]).cumprod()
        running_max = curve["net_value"].cummax()
        curve["drawdown"] = curve["net_value"] / running_max - 1.0
        return curve, {
            "annual_return": _annual_return(
                curve["net_value"], self.settings.annualization
            ),
            "annual_volatility": float(
                curve["net_return"].std(ddof=0)
                * np.sqrt(self.settings.annualization)
            ),
            "sharpe": _sharpe(
                curve["net_return"], self.settings.annualization
            ),
            "max_drawdown": float(curve["drawdown"].min()),
            "mean_turnover": float(curve["turnover"].mean()),
            "total_cost": float(curve["cost"].sum()),
            "final_net_value": float(curve["net_value"].iloc[-1]),
        }


def build_feedback_advice(metrics: dict[str, float]) -> list[str]:
    advice: list[str] = []
    rank_ic = metrics.get("rank_ic_mean", np.nan)
    turnover = metrics.get("mean_turnover", np.nan)
    coverage = metrics.get("coverage", np.nan)
    drawdown = metrics.get("max_drawdown", np.nan)
    if pd.isna(rank_ic) or abs(rank_ic) < 0.02:
        advice.append(
            "Rank IC is weak on this sample; simplify the thesis or lengthen the horizon."
        )
    if not pd.isna(turnover) and turnover > 0.35:
        advice.append(
            "Turnover is high; use longer rolling windows or a slower-changing input."
        )
    if not pd.isna(coverage) and coverage < 0.8:
        advice.append(
            "Coverage is limited; reduce nested lookbacks or missing-data dependence."
        )
    if not pd.isna(drawdown) and drawdown < -0.2:
        advice.append(
            "Drawdown is material; strengthen the risk penalty and test regime stability."
        )
    if not advice:
        advice.append(
            "No simple threshold failed; prioritize walk-forward and neutralization tests."
        )
    advice.append(
        "Any revision evaluated on the same sample remains in-sample and may overfit."
    )
    return advice


def _icir(values: pd.Series, annualization: int) -> float:
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(values.mean() / std * np.sqrt(annualization))


def _annual_return(net_value: pd.Series, annualization: int) -> float:
    if net_value.empty or net_value.iloc[-1] <= 0:
        return np.nan
    return float(net_value.iloc[-1] ** (annualization / len(net_value)) - 1.0)


def _sharpe(returns: pd.Series, annualization: int) -> float:
    std = returns.std(ddof=0)
    if std == 0 or pd.isna(std):
        return np.nan
    return float(returns.mean() / std * np.sqrt(annualization))


from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import pandas as pd

from .config import load_config
from .data import generate_demo_data
from .dsl import ExpressionEngine
from .factors import CLASSIC_FACTORS, random_factor
from .llm import build_llm
from .models import EvaluationResult
from .reporting import write_report
from .research import FactorResearcher, ResearchSettings, build_feedback_advice


class DemoPipeline:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

    def run(
        self,
        output_dir: str | Path = "artifacts/demo",
        llm_mode: str | None = None,
    ) -> tuple[list[EvaluationResult], dict]:
        data_config = self.config["data"]
        if data_config.get("source") != "simulated":
            raise ValueError(
                "DemoPipeline is deliberately offline-only; use a data adapter "
                "explicitly for real data."
            )
        data = generate_demo_data(
            start_date=data_config["start_date"],
            periods=int(data_config["periods"]),
            symbols=int(data_config["symbols"]),
            seed=int(data_config["seed"]),
        )
        return self.run_data(
            data,
            output_dir=output_dir,
            llm_mode=llm_mode,
            data_source="SIMULATED_A_SHARE_LIKE",
            is_simulated=True,
        )

    def run_data(
        self,
        data: pd.DataFrame,
        output_dir: str | Path,
        llm_mode: str | None = None,
        data_source: str = "USER_SUPPLIED_PANEL",
        is_simulated: bool = False,
    ) -> tuple[list[EvaluationResult], dict]:
        """Run the same benchmark on a validated simulated or real data panel."""
        data_config = self.config["data"]
        research_config = self.config["research"]
        backtest_config = self.config["backtest"]
        researcher = FactorResearcher(
            data,
            settings=ResearchSettings(
                forward_periods=int(research_config["forward_periods"]),
                quantiles=int(research_config["quantiles"]),
                min_cross_section=int(research_config["min_cross_section"]),
                top_k=int(backtest_config["top_k"]),
                fee_bps=float(backtest_config["fee_bps"]),
                annualization=int(backtest_config["annualization"]),
            ),
            engine=ExpressionEngine(),
        )

        results: list[EvaluationResult] = []
        for name, expression in CLASSIC_FACTORS.items():
            results.append(
                researcher.evaluate(name, "classic", expression)
            )

        random_proposal = random_factor(seed=int(data_config["seed"]) + 101)
        results.append(
            researcher.evaluate(
                random_proposal.name,
                "random",
                random_proposal.expression,
                random_proposal,
            )
        )

        llm_config = self.config["llm"]
        active_mode = llm_mode or llm_config.get("mode", "mock")
        configured_provider = llm_config.get("provider", "deepseek")
        llm = build_llm(
            active_mode,
            model=llm_config.get("model"),
            base_url=(
                llm_config.get("base_url")
                if active_mode == configured_provider
                else None
            ),
        )
        single = llm.generate()
        researcher.engine.validate(single.expression)
        single_expression = _orient(single.expression, single.expected_direction)
        single_result = researcher.evaluate(
            single.name, "llm_single", single_expression, single
        )
        results.append(single_result)

        advice = build_feedback_advice(single_result.metrics)
        revised = llm.improve(single, single_result.summary(), advice)
        researcher.engine.validate(revised.expression)
        revised_expression = _orient(
            revised.expression, revised.expected_direction
        )
        revised_result = researcher.evaluate(
            revised.name,
            "llm_feedback",
            revised_expression,
            revised,
        )
        results.append(revised_result)

        metadata = {
            "project": "FactorForge",
            "data_source": data_source,
            "is_simulated": is_simulated,
            "rows": len(data),
            "symbols": int(data.index.get_level_values("symbol").nunique()),
            "dates": int(data.index.get_level_values("date").nunique()),
            "seed": int(data_config["seed"]) if is_simulated else None,
            "llm_mode": active_mode,
            "fee_bps": float(backtest_config["fee_bps"]),
            "forward_periods": int(research_config["forward_periods"]),
            "feedback_advice": advice,
            "feedback_vs_single": {
                "rank_ic_mean_delta": _metric_delta(
                    revised_result, single_result, "rank_ic_mean"
                ),
                "mean_turnover_delta": _metric_delta(
                    revised_result, single_result, "mean_turnover"
                ),
                "max_drawdown_delta": _metric_delta(
                    revised_result, single_result, "max_drawdown"
                ),
                "note": (
                    "Computed on the same sample; a positive delta is not "
                    "out-of-sample evidence of improvement."
                ),
            },
        }
        paths = write_report(results, output_dir, metadata)
        return results, {"metadata": metadata, "paths": paths}


def _orient(expression: str, expected_direction: str) -> str:
    """Normalize all strategies so a larger evaluated value is preferred."""
    return expression if expected_direction == "positive" else f"-({expression})"


def _metric_delta(
    revised: EvaluationResult, original: EvaluationResult, key: str
) -> float | None:
    left, right = revised.metrics.get(key), original.metrics.get(key)
    if left is None or right is None:
        return None
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return None
    return float(left - right)

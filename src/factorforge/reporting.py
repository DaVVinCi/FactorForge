from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .models import EvaluationResult


def write_report(
    results: Iterable[EvaluationResult],
    output_dir: str | Path,
    metadata: dict,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    items = list(results)

    comparison = pd.DataFrame([item.summary() for item in items])
    comparison.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    proposals = [
        {
            "category": item.category,
            **item.proposal.model_dump(),
        }
        for item in items
        if item.proposal is not None
    ]
    (output / "factor_proposals.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    disclaimer = (
        "All observations and resulting metrics are simulated. They are pipeline "
        "validation outputs, not investment evidence."
        if metadata.get("is_simulated")
        else
        "Metrics use a user-supplied market-data panel that FactorForge does not "
        "independently verify. They are research outputs, not investment evidence."
    )
    payload = {
        **metadata,
        "disclaimer": disclaimer,
        "results": [item.summary() for item in items],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "feedback_advice.json").write_text(
        json.dumps(
            {
                "advice": metadata.get("feedback_advice", []),
                "feedback_vs_single": metadata.get("feedback_vs_single", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    equity = pd.concat(
        {item.name: item.equity_curve["net_value"] for item in items}, axis=1
    )
    equity.to_csv(output / "equity_curves.csv", encoding="utf-8-sig")
    sample_label = "simulated" if metadata.get("is_simulated") else "user-supplied"
    _plot_equity(equity, output / "equity_curves.png", sample_label)
    _plot_ic(comparison, output / "ic_comparison.png", sample_label)
    _plot_quantiles(items, output / "quantile_returns.png", sample_label)
    return {
        "comparison": output / "comparison.csv",
        "summary": output / "summary.json",
        "feedback": output / "feedback_advice.json",
        "equity_plot": output / "equity_curves.png",
        "ic_plot": output / "ic_comparison.png",
        "quantile_plot": output / "quantile_returns.png",
    }


def _plot_equity(equity: pd.DataFrame, path: Path, sample_label: str) -> None:
    selected = _display_columns(equity.columns)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    equity[selected].plot(ax=ax, linewidth=1.4)
    ax.set_title(f"FactorForge {sample_label} Top-K net value comparison")
    ax.set_ylabel("Net value")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ic(comparison: pd.DataFrame, path: Path, sample_label: str) -> None:
    frame = comparison.set_index("name")[["ic_mean", "rank_ic_mean"]].fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    frame.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{sample_label.title()}-sample IC comparison")
    ax.set_ylabel("Mean daily cross-sectional correlation")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_quantiles(
    items: list[EvaluationResult], path: Path, sample_label: str
) -> None:
    chosen = next(
        (item for item in items if item.category == "llm_feedback"), items[-1]
    )
    means = chosen.quantile_returns.filter(regex=r"^Q\d+$").mean()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    means.plot(kind="bar", ax=ax, color="#3676a3")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Mean next-period return by quantile: {chosen.name}")
    ax.set_ylabel(f"Mean return ({sample_label} sample)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _display_columns(columns: Iterable[str]) -> list[str]:
    values = list(columns)
    classic = [value for value in values if value == "momentum_20"]
    special = [
        value
        for value in values
        if value.startswith("random_")
        or "mock_llm" in value
        or "deepseek" in value.lower()
    ]
    chosen = classic + special
    return chosen or values[:5]

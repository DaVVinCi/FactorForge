from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactorProposal(BaseModel):
    """Structured contract shared by mock and live LLM providers."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=80)
    hypothesis: str = Field(min_length=10, max_length=1200)
    economic_rationale: str = Field(min_length=10, max_length=1600)
    expression: str = Field(min_length=3, max_length=1000)
    expected_direction: Literal["positive", "negative"]
    data_requirements: list[str] = Field(min_length=1)
    risk_notes: list[str] = Field(min_length=1)
    iteration_note: str = ""

    @field_validator("data_requirements", "risk_notes", mode="before")
    @classmethod
    def normalize_string_lists(cls, value):
        """Tolerate providers returning one string instead of a JSON array."""
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        return value


@dataclass
class EvaluationResult:
    name: str
    category: str
    expression: str
    metrics: dict[str, float]
    daily_ic: pd.DataFrame
    quantile_returns: pd.DataFrame
    equity_curve: pd.DataFrame
    proposal: FactorProposal | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "expression": self.expression,
            **{key: _json_number(value) for key, value in self.metrics.items()},
            "warnings": list(self.warnings),
        }


def _json_number(value: float) -> float | None:
    return None if pd.isna(value) else float(value)

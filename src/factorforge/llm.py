from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from .models import FactorProposal


SYSTEM_PROMPT = """You are a cautious A-share quantitative researcher.
Return exactly one JSON object matching these keys:
name, hypothesis, economic_rationale, expression, expected_direction,
data_requirements, risk_notes, iteration_note.
data_requirements and risk_notes MUST be JSON arrays of strings, even when
there is only one item. Example:
"data_requirements": ["daily adjusted close", "daily volume"],
"risk_notes": ["high turnover", "regime sensitivity"].
The expression may only use fields open, high, low, close, volume, amount, vwap;
functions Rank, Delay, Delta, Mean, Std, Sum, Min, Max, Corr, Abs, Sign;
and arithmetic +, -, *, /. Windows and lags must be integer literals from 1 to
252. Never use future data, negative lags, fundamentals, news or unavailable
fields. expected_direction must be "positive" or "negative". Do not claim that a
factor is profitable or validated."""

CURRENT_DEEPSEEK_FLASH_MODEL = "deepseek-flash"
DASHSCOPE_DEFAULT_MODEL = "deepseek-v4-flash"


class FactorLLM(ABC):
    @abstractmethod
    def generate(self) -> FactorProposal:
        raise NotImplementedError

    @abstractmethod
    def improve(
        self, previous: FactorProposal, metrics: dict[str, Any], advice: list[str]
    ) -> FactorProposal:
        raise NotImplementedError


class MockFactorLLM(FactorLLM):
    """Deterministic offline stand-in that exercises the full structured flow."""

    def generate(self) -> FactorProposal:
        return FactorProposal(
            name="mock_llm_quality_momentum_v1",
            hypothesis=(
                "A short moving-average return combined with a volatility penalty "
                "may favor steadier recent winners after the daily close."
            ),
            economic_rationale=(
                "The momentum term captures gradual price adjustment while the "
                "volatility term penalizes unstable moves that may reverse."
            ),
            expression=(
                "Rank(Mean(close / Delay(close, 1) - 1, 5)) "
                "- 0.3 * Rank(Std(close / Delay(close, 1) - 1, 10))"
            ),
            expected_direction="positive",
            data_requirements=["daily adjusted close"],
            risk_notes=[
                "Short horizon can create high turnover",
                "Momentum crashes",
                "Sensitive to suspension and limit-up execution",
            ],
            iteration_note="Offline deterministic Mock LLM proposal.",
        )

    def improve(
        self, previous: FactorProposal, metrics: dict[str, Any], advice: list[str]
    ) -> FactorProposal:
        return FactorProposal(
            name="mock_llm_quality_momentum_feedback_v2",
            hypothesis=(
                "Lengthening both the return and risk windows may reduce noise and "
                "turnover while retaining the original quality-momentum idea."
            ),
            economic_rationale=(
                "A slower signal reacts less to one-day price noise. A stronger "
                "risk penalty reduces exposure to volatile recent moves."
            ),
            expression=(
                "Rank(Mean(close / Delay(close, 1) - 1, 20)) "
                "- 0.5 * Rank(Std(close / Delay(close, 1) - 1, 20))"
            ),
            expected_direction="positive",
            data_requirements=["daily adjusted close"],
            risk_notes=[
                "Longer windows react slowly to regime changes",
                "Still requires out-of-sample and neutralization tests",
                "Backtest feedback can overfit the same sample",
            ],
            iteration_note=(
                "Mock feedback revision. Inputs included measured metrics and: "
                + "; ".join(advice)
            )[:1000],
        )


class DeepSeekFactorLLM(FactorLLM):
    """OpenAI-compatible DeepSeek client; credentials are environment-only."""

    provider_name = "DeepSeek"
    api_key_env = "DEEPSEEK_API_KEY"
    model_env = "DEEPSEEK_MODEL"
    base_url_env = "DEEPSEEK_BASE_URL"
    timeout_env = "DEEPSEEK_TIMEOUT_SECONDS"
    default_model = CURRENT_DEEPSEEK_FLASH_MODEL
    default_base_url = "https://api.deepseek.com"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. "
                "Use --llm-mode mock for offline runs."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                f"Install the openai package to use {self.provider_name}"
            ) from exc
        self.model = model or os.environ.get(self.model_env, self.default_model)
        self.base_url = base_url or os.environ.get(
            self.base_url_env, self.default_base_url
        )
        timeout_value = timeout or float(
            os.environ.get(self.timeout_env, "60")
        )
        self.client = OpenAI(
            api_key=api_key, base_url=self.base_url, timeout=timeout_value
        )

    def generate(self) -> FactorProposal:
        prompt = (
            "Propose one interpretable daily price-volume factor for a liquid "
            "A-share cross-section. Prefer a medium horizon and explicitly state "
            "risks. Output JSON."
        )
        return self._request(prompt)

    def improve(
        self, previous: FactorProposal, metrics: dict[str, Any], advice: list[str]
    ) -> FactorProposal:
        prompt = (
            "Revise the prior factor once using the measured backtest feedback. "
            "Keep the same broad economic idea unless the evidence is unusable. "
            "Do not claim improvement until it is re-evaluated.\n"
            f"Prior proposal: {previous.model_dump_json()}\n"
            f"Measured metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"Rule-based diagnostics: {json.dumps(advice, ensure_ascii=False)}\n"
            "Output JSON."
        )
        return self._request(prompt)

    def _request(self, user_prompt: str) -> FactorProposal:
        content = self._complete(user_prompt)
        if not content:
            # DeepSeek documents that JSON mode can occasionally return empty
            # content. Retry once with an explicit, compact reminder.
            retry_prompt = (
                user_prompt
                + "\nReturn one complete JSON object immediately. Do not emit "
                "reasoning, Markdown, or any text outside that object."
            )
            content = self._complete(retry_prompt)
        if not content:
            finish_reason = getattr(self, "_last_finish_reason", None) or "unknown"
            raise RuntimeError(
                f"{self.provider_name} returned empty content twice "
                f"(finish_reason={finish_reason}). Try again or use Mock mode."
            )
        try:
            return self._parse_proposal(content)
        except (json.JSONDecodeError, ValidationError) as first_error:
            # JSON mode can still yield a truncated or schema-invalid response.
            # Retry once with a narrowly scoped request before failing the demo.
            repair_prompt = (
                "Your previous response could not be parsed as the required "
                "FactorProposal JSON. Return a complete replacement JSON object "
                "now. Do not use Markdown, explanations, or code fences. "
                f"Parsing error: {first_error}"
            )
            repaired_content = self._complete(repair_prompt)
            try:
                return self._parse_proposal(repaired_content)
            except (json.JSONDecodeError, ValidationError) as repair_error:
                raise RuntimeError(
                    f"{self.provider_name} returned invalid factor JSON after "
                    f"one repair retry: {repair_error}"
                ) from repair_error

    def _complete(self, user_prompt: str) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2400,
            "stream": False,
        }
        if self.provider_name == "DeepSeek":
            # V4.1 Flash enables thinking by default. This app needs a compact,
            # machine-parsed JSON response, so disable thinking for this call.
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        self._last_finish_reason = getattr(choice, "finish_reason", None)
        return choice.message.content or ""

    @staticmethod
    def _parse_proposal(content: str) -> FactorProposal:
        cleaned = content.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        return FactorProposal.model_validate(json.loads(cleaned))


class DashScopeFactorLLM(DeepSeekFactorLLM):
    """Alibaba Cloud Model Studio via its OpenAI-compatible endpoint."""

    provider_name = "DashScope"
    api_key_env = "DASHSCOPE_API_KEY"
    model_env = "DASHSCOPE_MODEL"
    base_url_env = "DASHSCOPE_BASE_URL"
    timeout_env = "DASHSCOPE_TIMEOUT_SECONDS"
    default_model = DASHSCOPE_DEFAULT_MODEL
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_llm(
    mode: str,
    model: str | None = None,
    base_url: str | None = None,
) -> FactorLLM:
    normalized = mode.lower()
    if normalized == "mock":
        return MockFactorLLM()
    if normalized == "deepseek":
        return DeepSeekFactorLLM(model=model, base_url=base_url)
    if normalized == "dashscope":
        return DashScopeFactorLLM(model=model, base_url=base_url)
    raise ValueError("llm mode must be 'mock', 'deepseek', or 'dashscope'")

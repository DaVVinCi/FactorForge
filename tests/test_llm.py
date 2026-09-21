from factorforge.dsl import ExpressionEngine
from factorforge.llm import DeepSeekFactorLLM, DashScopeFactorLLM, MockFactorLLM
from factorforge.models import FactorProposal
from factorforge.research import build_feedback_advice


def test_mock_llm_is_structured_and_dsl_valid():
    llm = MockFactorLLM()
    proposal = llm.generate()
    ExpressionEngine().validate(proposal.expression)
    advice = build_feedback_advice(
        {
            "rank_ic_mean": 0.0,
            "mean_turnover": 0.8,
            "coverage": 0.9,
            "max_drawdown": -0.1,
        }
    )
    revised = llm.improve(proposal, {}, advice)
    ExpressionEngine().validate(revised.expression)
    assert proposal.expression != revised.expression
    assert "overfit" in revised.iteration_note.lower()


def test_provider_single_strings_are_normalized_to_lists():
    proposal = FactorProposal.model_validate(
        {
            "name": "provider_shape_compatibility",
            "hypothesis": "Recent volume expansion may precede price continuation.",
            "economic_rationale": "Volume can proxy for changing investor attention.",
            "expression": "Rank(Mean(volume, 5) / Mean(volume, 20) - 1)",
            "expected_direction": "positive",
            "data_requirements": "Daily price and volume data.",
            "risk_notes": "The factor may be noisy around event-driven spikes.",
            "iteration_note": "Compatibility test.",
        }
    )
    assert proposal.data_requirements == ["Daily price and volume data."]
    assert proposal.risk_notes == [
        "The factor may be noisy around event-driven spikes."
    ]


def test_dashscope_uses_its_own_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    client = DashScopeFactorLLM()
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_provider_retries_once_after_invalid_json():
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = (
                '{"name": "truncated"'
                if self.calls == 1
                else '''{
                    "name": "repaired_factor",
                    "hypothesis": "A medium-term return signal may persist.",
                    "economic_rationale": "Prices can adjust gradually.",
                    "expression": "Rank(Mean(close / Delay(close, 1) - 1, 20))",
                    "expected_direction": "positive",
                    "data_requirements": ["daily adjusted close"],
                    "risk_notes": ["regime sensitivity"],
                    "iteration_note": "Returned after JSON repair retry."
                }'''
            )
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = FakeCompletions()
    client = object.__new__(DeepSeekFactorLLM)
    client.client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    client.model = "deepseek-flash"
    client.provider_name = "DeepSeek"

    proposal = client.generate()
    assert proposal.name == "repaired_factor"
    assert completions.calls == 2


def test_provider_retries_once_after_empty_content():
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            content = "" if self.calls == 1 else '''{
                "name": "empty_response_recovered",
                "hypothesis": "A medium-term return signal may persist.",
                "economic_rationale": "Prices can adjust gradually.",
                "expression": "Rank(Mean(close / Delay(close, 1) - 1, 20))",
                "expected_direction": "positive",
                "data_requirements": ["daily adjusted close"],
                "risk_notes": ["regime sensitivity"],
                "iteration_note": "Returned after an empty-response retry."
            }'''
            message = type("Message", (), {"content": content})()
            choice = type(
                "Choice", (), {"message": message, "finish_reason": "stop"}
            )()
            return type("Response", (), {"choices": [choice]})()

    completions = FakeCompletions()
    client = object.__new__(DeepSeekFactorLLM)
    client.client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": completions})()}
    )()
    client.model = "deepseek-flash"
    client.provider_name = "DeepSeek"

    proposal = client.generate()
    assert proposal.name == "empty_response_recovered"
    assert completions.calls == 2

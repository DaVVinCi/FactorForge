import json

from factorforge.config import load_config
from factorforge.pipeline import DemoPipeline


def test_offline_pipeline_runs_four_required_categories(tmp_path):
    config = load_config()
    config["data"].update({"periods": 80, "symbols": 10, "seed": 11})
    config["research"]["min_cross_section"] = 6
    config["backtest"]["top_k"] = 3
    results, report = DemoPipeline(config).run(tmp_path, llm_mode="mock")
    categories = {item.category for item in results}
    assert {"classic", "random", "llm_single", "llm_feedback"} <= categories
    assert len([item for item in results if item.category == "classic"]) >= 5
    summary = json.loads(report["paths"]["summary"].read_text(encoding="utf-8"))
    assert summary["is_simulated"] is True
    assert "not investment evidence" in summary["disclaimer"]
    assert summary["feedback_advice"]
    assert report["paths"]["feedback"].exists()
    assert report["paths"]["equity_plot"].exists()

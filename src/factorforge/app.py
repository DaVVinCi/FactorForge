from __future__ import annotations

import os
from copy import deepcopy

import pandas as pd
import streamlit as st

from .config import load_config
from .llm import CURRENT_DEEPSEEK_FLASH_MODEL, DASHSCOPE_DEFAULT_MODEL
from .pipeline import DemoPipeline


def main() -> None:
    st.set_page_config(
        page_title="FactorForge",
        page_icon="⚒️",
        layout="wide",
    )
    st.title("FactorForge · AI 量化因子研究工坊")
    st.caption(
        "从结构化假设、受限 DSL、IC/分层测试到含成本 Top-K 回测与反馈迭代"
    )
    st.warning(
        "默认页面使用确定性模拟 A 股风格数据。所有数值仅用于验证流程，"
        "不是历史业绩、投资建议或可交易结论。"
    )

    with st.sidebar:
        st.header("实验设置")
        mode = st.selectbox(
            "LLM 模式", ["mock", "deepseek", "dashscope"], index=0
        )
        model_options = (
            [CURRENT_DEEPSEEK_FLASH_MODEL, "deepseek-v4-pro"]
            if mode == "deepseek"
            else [DASHSCOPE_DEFAULT_MODEL, "deepseek-v4-pro"]
        )
        model = st.selectbox(
            "模型",
            model_options,
            disabled=mode == "mock",
        )
        periods = st.slider("模拟交易日", 100, 500, 260, 20)
        symbols = st.slider("模拟股票数", 10, 100, 30, 5)
        fee_bps = st.slider("单边手续费（bps）", 0.0, 30.0, 10.0, 1.0)
        top_k = st.slider("Top-K", 2, min(20, symbols), min(5, symbols), 1)
        required_key = {
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
        }.get(mode)
        if required_key and not os.environ.get(required_key):
            st.error(f"未检测到 {required_key}；请改用 Mock 或先设置环境变量。")
        run = st.button("运行完整研究流水线", type="primary", use_container_width=True)

    if not run:
        _landing()
        return
    if required_key and not os.environ.get(required_key):
        st.stop()

    config = deepcopy(load_config())
    config["data"]["periods"] = periods
    config["data"]["symbols"] = symbols
    config["backtest"]["fee_bps"] = fee_bps
    config["backtest"]["top_k"] = top_k
    config["llm"]["mode"] = mode
    config["llm"]["model"] = model
    with st.spinner("生成因子、执行检查并运行研究与回测……"):
        try:
            results, report = DemoPipeline(config).run(
                output_dir="artifacts/streamlit", llm_mode=mode
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    comparison = pd.DataFrame([item.summary() for item in results])
    tab_overview, tab_ic, tab_backtest, tab_hypotheses = st.tabs(
        ["四类对比", "IC 与分层", "策略回测", "假设与反馈"]
    )
    with tab_overview:
        required_categories = [
            "classic",
            "random",
            "llm_single",
            "llm_feedback",
        ]
        representatives = (
            comparison.sort_values(["category", "name"])
            .groupby("category", as_index=False)
            .first()
        )
        st.subheader("至少四类基线均已执行")
        cols = st.columns(4)
        labels = {
            "classic": "经典人工因子",
            "random": "随机因子",
            "llm_single": "单轮 LLM",
            "llm_feedback": "反馈迭代 LLM",
        }
        for column, category in zip(cols, required_categories):
            row = representatives[representatives["category"] == category]
            with column:
                st.metric(labels[category], "已完成" if not row.empty else "缺失")
                if not row.empty:
                    st.caption(str(row.iloc[0]["name"]))
        display_columns = [
            "category",
            "name",
            "coverage",
            "ic_mean",
            "rank_ic_mean",
            "rank_icir",
            "mean_turnover",
            "annual_return",
            "max_drawdown",
        ]
        st.dataframe(
            comparison[display_columns].style.format(
                {
                    key: "{:.4f}"
                    for key in display_columns
                    if key not in {"category", "name"}
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_ic:
        chosen_name = st.selectbox(
            "选择因子",
            comparison["name"].tolist(),
            index=len(comparison) - 1,
            key="ic_factor",
        )
        chosen = next(item for item in results if item.name == chosen_name)
        st.line_chart(chosen.daily_ic[["ic", "rank_ic"]])
        if not chosen.quantile_returns.empty:
            st.bar_chart(
                chosen.quantile_returns.filter(regex=r"^Q\d+$").mean()
            )
        st.caption(
            "IC 为当日截面因子值与下一期收益的相关系数；分层收益未年化。"
        )

    with tab_backtest:
        names = st.multiselect(
            "对比净值",
            comparison["name"].tolist(),
            default=[
                results[0].name,
                results[-2].name,
                results[-1].name,
            ],
        )
        if names:
            curves = pd.concat(
                {
                    item.name: item.equity_curve["net_value"]
                    for item in results
                    if item.name in names
                },
                axis=1,
            )
            st.line_chart(curves)
        st.caption(
            f"Top-{top_k} 等权，信号在 t 日收盘后形成，收益使用 t→t+1，"
            f"换仓按 {fee_bps:.1f} bps 扣费。"
        )

    with tab_hypotheses:
        proposals = [item for item in results if item.proposal is not None]
        for item in proposals:
            with st.expander(f"{item.category} · {item.name}", expanded=False):
                proposal = item.proposal
                st.code(proposal.expression, language="text")
                st.markdown(f"**假设：** {proposal.hypothesis}")
                st.markdown(f"**经济逻辑：** {proposal.economic_rationale}")
                st.markdown(f"**迭代说明：** {proposal.iteration_note}")
                st.markdown("**风险：** " + "；".join(proposal.risk_notes))

    st.success(
        f"报告已写入 {report['paths']['summary']}；共完成 {len(results)} 个实验。"
    )


def _landing() -> None:
    col1, col2, col3 = st.columns(3)
    col1.info("① LLM / Mock 输出结构化金融假设与 DSL 表达式")
    col2.info("② 安全检查后计算 IC、Rank IC、分层收益与换手率")
    col3.info("③ 含成本 Top-K 回测，并把诊断反馈给下一轮因子")
    st.subheader("内置因子 DSL")
    st.code(
        "Rank(Mean(close / Delay(close, 1) - 1, 20))\n"
        "- 0.5 * Rank(Std(close / Delay(close, 1) - 1, 20))",
        language="text",
    )
    st.caption(
        "支持 Rank、Delay、Delta、Mean、Std、Corr、Sum、Min、Max、Abs、Sign；"
        "字段和函数均为白名单。"
    )


if __name__ == "__main__":
    main()

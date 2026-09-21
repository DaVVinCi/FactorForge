# FactorForge

> A runnable portfolio MVP for an AI quantitative-research role: an LLM produces a structured financial hypothesis and restricted factor expression; FactorForge validates it, computes the signal, performs IC/quantile analysis, runs a cost-aware Top-K backtest, and feeds diagnostics into one revision.

[中文文档](README_CN.md)

![FactorForge architecture](docs/architecture.svg)

## Important disclaimer

The default demo uses **deterministic, simulated A-share-like daily data**. Synthetic symbols end in `.SIM`. Every chart, IC, return, NAV, and drawdown is computed from that simulation and is shown only to validate the pipeline. It is not real performance, investment advice, or evidence that a factor is tradable. No experiment metric is hard-coded or fabricated.

## Quick start

Python 3.10+ is required.

```powershell
python -m pip install -e ".[ui,dev]"
python -m factorforge demo --config config/default.yaml --output-dir artifacts/demo
```

One-command tests:

```powershell
python -m pytest
```

Launch the interactive demo:

```powershell
streamlit run streamlit_app.py
```

The app is normally served at `http://localhost:8501`.

## What the demo compares

One default run executes 10 experiments across four required groups:

| Group | Implementation | Purpose |
|---|---|---|
| Hand-crafted classics | Seven fixed price-volume expressions | Interpretable baselines |
| Random factor | A valid expression selected with a fixed seed | Chance-discovery control |
| Single-pass LLM | Structured Mock or DeepSeek proposal | Hypothesis-to-research flow |
| Feedback LLM | One revision informed by measured diagnostics | Closed-loop flow without claiming improvement |

The seven classic factors are 20-day momentum, 5-day reversal, 20-day low volatility, price-volume correlation, volume expansion, Amihud-style liquidity, and overnight gap.

## Safe factor DSL

Example:

```text
Rank(Mean(close / Delay(close, 1) - 1, 20))
- 0.5 * Rank(Std(close / Delay(close, 1) - 1, 20))
```

- Field whitelist: `open`, `high`, `low`, `close`, `volume`, `amount`, `vwap`
- Operator whitelist: `Rank`, `Delay`, `Delta`, `Mean`, `Std`, `Corr`, `Sum`, `Min`, `Max`, `Abs`, `Sign`
- Only numeric constants and `+ - * /`
- Windows/lags must be positive integer literals from 1 to 252
- Negative lags, future-return fields, attributes, indexing, imports, and arbitrary calls are rejected
- Near-zero denominators become `NaN`; infinities cannot escape the evaluator
- Python `eval` is never used

Signals are formed after the close on day `t`; IC and backtests use the `t→t+1` forward return.

## Research and backtest outputs

- Daily cross-sectional Pearson IC and Spearman Rank IC
- ICIR, Rank ICIR, positive-IC rate, and coverage
- Quantile returns and Q5−Q1 spread
- Equal-weight Top-K strategy
- Turnover-based one-way fees
- Gross/net returns, NAV, annualized return/volatility, Sharpe, maximum drawdown
- Feedback rules for weak Rank IC, high turnover, low coverage, large drawdown, and same-sample overfitting

Annualized metrics are mechanical transformations of the input sample. For the default simulated sample they are not performance claims.

## DeepSeek V4

FactorForge uses DeepSeek through the OpenAI-compatible Python SDK. Credentials are read only from environment variables. DeepSeek's current official model identifiers are `deepseek-flash` (V4.1 Flash) and `deepseek-v4-pro`; the OpenAI-compatible base URL is `https://api.deepseek.com`.

```powershell
$env:DEEPSEEK_API_KEY="your-key"
$env:DEEPSEEK_MODEL="deepseek-flash"  # or deepseek-v4-pro
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
python -m factorforge demo --llm-mode deepseek --output-dir artifacts/deepseek
```

Without a key, the deterministic Mock LLM exercises the complete offline flow:

```powershell
python -m factorforge demo --llm-mode mock
```

See `.env.example`. FactorForge does not load or save a `.env` file automatically; inject secrets through your shell, IDE, container platform, or secret manager.

Official references: [first DeepSeek API call](https://api-docs.deepseek.com/guides/function_calling/) · [models and pricing](https://api-docs.deepseek.com/quick_start/pricing)

### Alibaba Cloud Model Studio / DashScope

For an Alibaba Cloud Model Studio API key, set one PowerShell environment variable and select the separate `dashscope` mode:

```powershell
$env:DASHSCOPE_API_KEY = "your-model-studio-key"
python -m factorforge demo --llm-mode dashscope --output-dir artifacts/dashscope
```

The defaults are `deepseek-v4-flash` and the Beijing-region general OpenAI-compatible endpoint. Override them when needed:

```powershell
$env:DASHSCOPE_MODEL = "deepseek-v4-pro"
$env:DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

If your Model Studio console supplies a workspace-specific Base URL, use that complete URL instead. See the [official Alibaba Cloud DeepSeek documentation](https://help.aliyun.com/zh/model-studio/deepseek-api).

## Optional real A-share data

AkShare is an optional network dependency:

```powershell
python -m pip install -e ".[realdata]"
python -m factorforge fetch `
  --symbols 000001.SZ,600000.SH `
  --start 2023-01-01 `
  --end 2025-12-31 `
  --adjust qfq `
  --output data/a_share_sample.csv
```

Run the same benchmark on a compatible CSV:

```powershell
python -m factorforge research-csv `
  --input data/a_share_sample.csv `
  --output-dir artifacts/real_csv `
  --llm-mode mock
```

Required columns are `date,symbol,open,high,low,close,volume,amount`, with one row per date/symbol. Users remain responsible for data licensing, quality, adjustment, suspension, and delisting treatment.

## Generated artifacts

`artifacts/demo/` contains:

- `comparison.csv`
- `summary.json`
- `factor_proposals.json`
- `feedback_advice.json`
- `equity_curves.csv`
- `equity_curves.png`
- `ic_comparison.png`
- `quantile_returns.png`

![Simulated equity curves](artifacts/demo/equity_curves.png)

![Simulated IC comparison](artifacts/demo/ic_comparison.png)

## Layout

```text
FactorForge/
├── config/default.yaml
├── docs/architecture.svg
├── src/factorforge/
│   ├── data.py          # simulation, CSV, optional AkShare
│   ├── dsl.py           # AST validation and safe evaluation
│   ├── factors.py       # classics and random baseline
│   ├── llm.py           # Mock / DeepSeek generation
│   ├── research.py      # IC, quantiles, Top-K backtest
│   ├── pipeline.py      # four-group benchmark and feedback
│   ├── reporting.py     # CSV/JSON/PNG artifacts
│   ├── app.py           # Streamlit UI
│   └── cli.py
├── tests/
├── streamlit_app.py
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## Docker

```powershell
docker build -t factorforge .
docker run --rm -p 8501:8501 factorforge
```

## Research limitations

Before treating this as serious A-share research, add a point-in-time tradable universe, listing-age/ST/delisting/suspension/price-limit filters, historical index constituents, industry and size neutralization, corporate-action controls, walk-forward evaluation, multiple-testing correction, and execution-aware slippage/capacity constraints. `research-csv` outputs remain research diagnostics, not trading instructions.

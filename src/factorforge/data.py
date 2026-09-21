from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def generate_demo_data(
    start_date: str = "2023-01-02",
    periods: int = 260,
    symbols: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic A-share-like OHLCV data.

    This is synthetic data, not reconstructed market data. It deliberately
    includes weak serial/cross-sectional structure so the research workflow
    has something to measure; no result should be interpreted as investable.
    """
    if periods < 40:
        raise ValueError("periods must be at least 40")
    if symbols < 5:
        raise ValueError("symbols must be at least 5")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=periods, name="date")
    market_noise = rng.normal(0.00015, 0.008, periods)
    market = np.empty(periods)
    market[0] = market_noise[0]
    for t in range(1, periods):
        market[t] = 0.08 * market[t - 1] + market_noise[t]

    frames: list[pd.DataFrame] = []
    for idx in range(symbols):
        symbol = f"{idx + 1:06d}.SIM"
        beta = rng.uniform(0.65, 1.35)
        idio = rng.normal(0, rng.uniform(0.010, 0.022), periods)
        latent_quality = rng.normal(0, 0.00015)
        returns = np.empty(periods)
        returns[0] = beta * market[0] + idio[0]
        for t in range(1, periods):
            short_reversal = -0.06 * returns[t - 1]
            returns[t] = (
                beta * market[t] + idio[t] + short_reversal + latent_quality
            )
        returns = np.clip(returns, -0.095, 0.095)
        close = rng.uniform(6.0, 60.0) * np.exp(np.cumsum(returns))
        previous_close = np.r_[close[0] / np.exp(returns[0]), close[:-1]]
        open_price = previous_close * np.exp(rng.normal(0, 0.0035, periods))
        intraday_range = np.abs(rng.normal(0.008, 0.004, periods))
        high = np.maximum(open_price, close) * (1.0 + intraday_range)
        low = np.minimum(open_price, close) * (1.0 - intraday_range)
        volume = rng.lognormal(mean=15.1, sigma=0.45, size=periods)
        volume *= 1.0 + 5.0 * np.abs(returns)
        volume = np.maximum(volume, 100.0).round(-2)
        vwap = (open_price + high + low + close) / 4.0
        amount = volume * vwap
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "vwap": vwap,
                    "is_simulated": True,
                }
            )
        )
    return validate_panel(pd.concat(frames, ignore_index=True))


def validate_panel(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    missing = {"date", "symbol", *REQUIRED_COLUMNS}.difference(data.columns)
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"])
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("duplicate date/symbol observations")
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if (data[["open", "high", "low", "close", "volume", "amount"]] < 0).any().any():
        raise ValueError("prices, volume and amount must be non-negative")
    return data.set_index(["date", "symbol"]).sort_index()


@dataclass
class CSVPanelAdapter:
    path: str | Path

    def load(self) -> pd.DataFrame:
        return validate_panel(pd.read_csv(self.path))


@dataclass
class AkShareAshareAdapter:
    """Optional network adapter for unadjusted/qfq/hfq A-share daily data."""

    symbols: Iterable[str]
    start_date: str
    end_date: str
    adjust: str = "qfq"

    def load(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError(
                "AkShare is optional. Install with: pip install -e .[realdata]"
            ) from exc

        frames: list[pd.DataFrame] = []
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        for raw_symbol in self.symbols:
            code = raw_symbol.split(".")[0]
            item = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                adjust=self.adjust,
            ).rename(columns=mapping)
            if item.empty:
                continue
            item["symbol"] = raw_symbol
            frames.append(item[["date", "symbol", *REQUIRED_COLUMNS]])
        if not frames:
            raise ValueError("AkShare returned no observations for requested symbols")
        data = validate_panel(pd.concat(frames, ignore_index=True))
        data["is_simulated"] = False
        data["vwap"] = data["amount"] / data["volume"].replace(0, np.nan)
        return data


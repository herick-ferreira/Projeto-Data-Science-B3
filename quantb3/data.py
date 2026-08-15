"""Extração de dados e engenharia de features sem vazamento temporal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from quantb3.config import BENCHMARK, FORECAST_HORIZON, TARGET_RETURN


def download_ohlcv(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Baixa OHLCV ajustado; auto_adjust incorpora splits e dividendos."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = raw[ticker].copy() if len(tickers) > 1 else raw.copy()
        # Feriados/dias sem negociação são alinhados por ffill; início sem preço é removido.
        frame = frame.sort_index().ffill().dropna(subset=["Close"])
        result[ticker] = frame
    return result


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gains / losses.replace(0, pd.NA))


def add_features(asset: pd.DataFrame, benchmark: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Cria variáveis usando exclusivamente observações até o fechamento atual."""
    df = asset.copy()
    close = df["Close"]
    high, low = df["High"], df["Low"]
    df["ticker"] = ticker
    df["return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["rsi_14"] = _rsi(close)

    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    middle = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    upper, lower = middle + 2 * band_std, middle - 2 * band_std
    df["bb_pct_b"] = (close - lower) / (upper - lower)

    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean() / close
    for period in (9, 21, 50):
        df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean() / close - 1
    df["ema_9_above_21"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema_21_above_50"] = (df["ema_21"] > df["ema_50"]).astype(int)
    for window in (5, 10, 21):
        df[f"return_{window}d"] = close.pct_change(window)
    for window in (10, 21):
        df[f"vol_{window}d"] = df["return"].rolling(window).std() * (252 ** 0.5)
    df["relative_volume_20d"] = df["Volume"] / df["Volume"].rolling(20).mean()

    ibov_return = benchmark["Close"].pct_change().reindex(df.index).ffill()
    df["ibov_return_5d"] = benchmark["Close"].pct_change(5).reindex(df.index).ffill()
    df["relative_strength_5d"] = df["return_5d"] - df["ibov_return_5d"]
    df["beta_60d"] = df["return"].rolling(60).cov(ibov_return) / ibov_return.rolling(60).var()

    future_return = close.shift(-FORECAST_HORIZON) / close - 1
    df["target"] = (future_return > TARGET_RETURN).astype(int)
    # Aqui evitamos look-ahead bias — o target representa informação futura usada apenas no treino/teste, nunca como feature.
    return df.iloc[:-FORECAST_HORIZON].dropna()


def build_dataset(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Baixa todos os ativos e devolve dataset empilhado, séries e benchmark."""
    prices = download_ohlcv(tickers + [BENCHMARK], start, end)
    benchmark = prices.pop(BENCHMARK)
    featured = {
        ticker: add_features(frame, benchmark, ticker)
        for ticker, frame in prices.items()
        if not frame.empty
    }
    return pd.concat(featured.values()).sort_index(), featured, benchmark

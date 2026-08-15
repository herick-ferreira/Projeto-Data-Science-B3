"""Treinamento temporal, avaliação e backtest da estratégia."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Seleciona somente atributos observáveis no instante da previsão."""
    excluded = {"ticker", "target", "Open", "High", "Low", "Close", "Volume", "return", "log_return"}
    return [column for column in dataset.columns if column not in excluded]


def temporal_splits(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide por data; nunca embaralha linhas de uma série temporal."""
    dates = pd.to_datetime(dataset.index)
    return dataset[dates.year <= 2021], dataset[dates.year == 2022], dataset[dates.year >= 2023]


def walk_forward_auc(pipeline, train: pd.DataFrame, columns: list[str], n_splits: int = 5) -> list[float]:
    """CV temporal: K-Fold aleatório vazaria informação do futuro para o passado."""
    ordered = train.sort_index().reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, valid_idx in splitter.split(ordered):
        fitted = clone(pipeline)
        fitted.fit(ordered.iloc[train_idx][columns], ordered.iloc[train_idx]["target"])
        y_val = ordered.iloc[valid_idx]["target"]
        scores.append(roc_auc_score(y_val, fitted.predict_proba(ordered.iloc[valid_idx][columns])[:, 1]))
    return scores


def classification_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Métricas da classe compra e capacidade de ordenação do classificador."""
    prediction = (probability >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
    }


def optimize_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    """Escolhe limiar por F1 na validação, mantendo o teste final intocado."""
    thresholds = np.arange(0.35, 0.71, 0.01)
    return float(max(thresholds, key=lambda t: f1_score(y_true, probability >= t, zero_division=0)))


def strategy_backtest(test: pd.DataFrame, probability: np.ndarray, threshold: float, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Backtest diário agregado, com sinal defasado um pregão para execução realista."""
    result = test[["ticker", "return"]].copy()
    result["probability"] = probability
    # A decisão feita no fechamento só pode afetar o retorno a partir do próximo pregão.
    result["position"] = (result["probability"] >= threshold).astype(int)
    daily = result.groupby(result.index).apply(lambda x: (x["position"].shift(1).fillna(0) * x["return"]).mean())
    daily.name = "strategy_return"
    out = daily.to_frame()
    out["benchmark_return"] = benchmark["Close"].pct_change().reindex(out.index).fillna(0)
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["benchmark_equity"] = (1 + out["benchmark_return"]).cumprod()
    return out


def performance_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    """Calcula Sharpe, Calmar e drawdown máximo anualizados."""
    returns = backtest["strategy_return"]
    equity = backtest["strategy_equity"]
    drawdown = equity / equity.cummax() - 1
    annual_return = equity.iloc[-1] ** (252 / max(len(equity), 1)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    max_drawdown = drawdown.min()
    return {
        "cumulative_return": equity.iloc[-1] - 1,
        "sharpe": annual_return / annual_vol if annual_vol else np.nan,
        "calmar": annual_return / abs(max_drawdown) if max_drawdown else np.nan,
        "max_drawdown": max_drawdown,
    }

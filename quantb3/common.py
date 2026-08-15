"""Serviços compartilhados pela aplicação Streamlit."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from quantb3.config import BENCHMARK, MODEL_DIR, TICKERS
from quantb3.data import add_features, download_ohlcv


@st.cache_data(ttl=3600, show_spinner="Atualizando dados da B3...")
def market_data(period: str = "1y") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Obtém dados recentes uma vez por hora."""
    prices = download_ohlcv(TICKERS + [BENCHMARK], start="2019-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    benchmark = prices.pop(BENCHMARK)
    return prices, benchmark


@st.cache_resource
def load_model():
    """Carrega artefatos apenas uma vez; retorna None antes do treinamento."""
    pipeline_path = MODEL_DIR / "pipeline_lgbm.pkl"
    features_path = MODEL_DIR / "feature_names.json"
    if not pipeline_path.exists() or not features_path.exists():
        return None, []
    return joblib.load(pipeline_path), json.loads(features_path.read_text(encoding="utf-8"))


def latest_features(ticker: str, prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame) -> pd.DataFrame:
    """Calcula o último vetor de atributos real para um ticker."""
    return add_features(prices[ticker], benchmark, ticker).tail(1)


def model_probabilities(prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame) -> pd.DataFrame:
    """Gera a tabela de sinais ou um DataFrame vazio se o modelo ainda não existe."""
    model, columns = load_model()
    if model is None:
        return pd.DataFrame()
    rows = []
    for ticker in TICKERS:
        feature_row = latest_features(ticker, prices, benchmark)
        probability = model.predict_proba(feature_row[columns])[:, 1][0]
        last = prices[ticker].iloc[-1]
        daily_return = prices[ticker]["Close"].pct_change().iloc[-1]
        rows.append({"ticker": ticker, "preco": last["Close"], "retorno_dia": daily_return, "probabilidade": probability})
    return pd.DataFrame(rows)


def require_model() -> tuple[object, list[str]]:
    """Mostra instrução de treino quando o artefato ainda não foi produzido."""
    model, columns = load_model()
    if model is None:
        st.warning("Modelo ainda não treinado. Execute `python -m quantb3.train` no diretório do projeto.")
        st.stop()
    return model, columns

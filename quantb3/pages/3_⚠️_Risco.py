"""Painel interativo de risco de carteira."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantb3.config import TICKERS
from quantb3.common import market_data
from quantb3.risk import portfolio_metrics

st.title("⚠️ Painel de Risco")
prices, ibov = market_data()
selected = st.multiselect("Ativos da carteira (até 5)", TICKERS, default=TICKERS[:3], max_selections=5)
if not selected:
    st.info("Selecione ao menos um ativo.")
    st.stop()
raw_weights = np.array([st.sidebar.slider(f"Peso {ticker}", 0, 100, int(100 / len(selected))) for ticker in selected], dtype=float)
weights = raw_weights / raw_weights.sum() if raw_weights.sum() else np.repeat(1 / len(selected), len(selected))
st.caption("Pesos normalizados automaticamente para somar 100%.")
returns = pd.DataFrame({ticker: prices[ticker]["Close"].pct_change() for ticker in selected}).dropna()
metrics = portfolio_metrics(returns, weights, ibov["Close"].pct_change())
cols = st.columns(5)
for col, (label, value) in zip(cols, [("VaR 95%", metrics["var_95"]), ("VaR 99%", metrics["var_99"]), ("CVaR 95%", metrics["cvar_95"]), ("Vol. anual", metrics["annual_volatility"]), ("Beta", metrics["beta"])]):
    col.metric(label, f"{value:.2%}" if label != "Beta" else f"{value:.2f}")
st.subheader("Correlação da carteira")
st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width='stretch')

# Fronteira eficiente simplificada: 1.000 carteiras aleatórias no mesmo universo selecionado.
rng = np.random.default_rng(42)
sim_weights = rng.dirichlet(np.ones(len(selected)), 1000)
annual_returns = returns.mean().to_numpy() * 252
cov = returns.cov().to_numpy() * 252
sim_ret = sim_weights @ annual_returns
sim_vol = np.sqrt(np.einsum("ij,jk,ik->i", sim_weights, cov, sim_weights))
frontier = go.Figure(go.Scatter(x=sim_vol, y=sim_ret, mode="markers", marker=dict(color=sim_ret / sim_vol, colorscale="Viridis", showscale=True), name="Carteiras aleatórias"))
frontier.add_scatter(x=[np.sqrt(weights @ cov @ weights)], y=[weights @ annual_returns], mode="markers", marker=dict(color="red", size=14, symbol="star"), name="Sua carteira")
frontier.update_layout(template="plotly_dark", xaxis_title="Volatilidade anualizada", yaxis_title="Retorno esperado anualizado")
st.plotly_chart(frontier, width='stretch')

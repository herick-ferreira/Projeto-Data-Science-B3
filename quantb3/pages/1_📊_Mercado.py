"""Dashboard executivo de mercado."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantb3.config import TICKERS
from quantb3.common import market_data, model_probabilities

st.title("📊 Mercado")
prices, ibov = market_data()
ticker = st.sidebar.selectbox("Ativo", TICKERS)
period = st.sidebar.selectbox("Período", ["1M", "3M", "6M", "1A", "5A"], index=3)
days = {"1M": 22, "3M": 66, "6M": 126, "1A": 252, "5A": 1260}[period]
ibov_returns = ibov["Close"].pct_change()
signals = model_probabilities(prices, ibov)
cols = st.columns(5)
for col, label, value in zip(cols[:4], ["Ibovespa hoje", "Semana", "Mês", "Ano"], [ibov_returns.iloc[-1], ibov["Close"].pct_change(5).iloc[-1], ibov["Close"].pct_change(21).iloc[-1], ibov["Close"].pct_change(252).iloc[-1]]):
    col.metric(label, f"{value:.2%}")
cols[4].metric("Volatilidade 21d", f"{ibov_returns.rolling(21).std().iloc[-1] * (252 ** .5):.2%}")
if not signals.empty:
    st.caption(f"Sinais atuais: {(signals.probabilidade >= .55).sum()} compra | {(signals.probabilidade < .55).sum()} venda/espera")

frame = prices[ticker].tail(days)
ema21 = frame["Close"].ewm(span=21, adjust=False).mean()
std = frame["Close"].rolling(20).std()
fig = go.Figure([go.Candlestick(x=frame.index, open=frame.Open, high=frame.High, low=frame.Low, close=frame.Close, name=ticker), go.Scatter(x=frame.index, y=ema21, name="EMA 21"), go.Scatter(x=frame.index, y=frame.Close.rolling(20).mean() + 2 * std, name="BB superior", line=dict(dash="dot")), go.Scatter(x=frame.index, y=frame.Close.rolling(20).mean() - 2 * std, name="BB inferior", line=dict(dash="dot"))])
fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, width='stretch')

heat = pd.DataFrame({symbol.replace(".SA", ""): data["Close"].pct_change().tail(22) for symbol, data in prices.items()})
st.subheader("Mapa de calor: retornos diários")
st.dataframe(heat.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=None), width='stretch')

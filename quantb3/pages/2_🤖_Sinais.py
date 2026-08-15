"""Página de sinais de negociação."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from quantb3.config import TICKERS
from quantb3.common import market_data, model_probabilities, require_model

st.title("🤖 Sinais do Modelo")
st.markdown("<div class='disclaimer'>⚠️ Os sinais gerados são para fins educacionais e de portfólio. Não constituem recomendação de investimento.</div>", unsafe_allow_html=True)
require_model()
prices, ibov = market_data()
signals = model_probabilities(prices, ibov)
signals["classificacao"] = signals.probabilidade.map(lambda x: "🟢 Compra" if x >= .55 else "🔴 Venda")
signals["confianca"] = signals.probabilidade.map(lambda x: f"{x:.0%}")
st.dataframe(signals.rename(columns={"preco": "Último preço", "retorno_dia": "Retorno dia", "probabilidade": "Prob. compra"}).style.format({"Último preço": "R$ {:.2f}", "Retorno dia": "{:.2%}", "Prob. compra": "{:.2%}"}), width='stretch')
ticker = st.selectbox("Ativo para detalhar", TICKERS)
probability = signals.loc[signals.ticker == ticker, "probabilidade"].iloc[0]
gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix": "%"}, title={"text": f"Probabilidade de compra — {ticker}"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22c55e"}, "steps": [{"range": [0, 45], "color": "#ef4444"}, {"range": [45, 55], "color": "#eab308"}]}))
st.plotly_chart(gauge, width='stretch')
history = prices[ticker].tail(30)
st.line_chart(history["Close"], y_label="Preço ajustado (R$)")
st.caption("O histórico de sinais depende do reprocessamento diário; esta visão mostra os últimos preços disponíveis.")

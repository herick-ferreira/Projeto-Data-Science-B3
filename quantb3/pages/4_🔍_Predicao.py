"""Ferramenta de previsão explicável por inputs ou dados reais."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import shap
import streamlit as st

from quantb3.config import TICKERS
from quantb3.common import latest_features, market_data, require_model

st.title("🔍 Predição Interativa")
model, columns = require_model()
prices, ibov = market_data()
ticker = st.selectbox("Ticker", TICKERS)
use_real = st.checkbox("Preencher com dados reais", value=True)
real = latest_features(ticker, prices, ibov) if use_real else pd.DataFrame()
inputs = {}
with st.form("prediction_form"):
    for feature in columns:
        default = float(real[feature].iloc[0]) if use_real and feature in real else 0.0
        inputs[feature] = st.number_input(feature, value=default, format="%.6f")
    submit = st.form_submit_button("Gerar sinal")
if submit:
    row = pd.DataFrame([inputs], columns=columns)
    probability = model.predict_proba(row)[:, 1][0]
    signal = "🟢 COMPRA" if probability >= .55 else "🔴 VENDA / ESPERA"
    st.metric("Probabilidade de compra", f"{probability:.1%}", signal)
    st.caption("⚠️ Uso educacional e de portfólio. Não constitui recomendação de investimento.")
    try:
        classifier = model.named_steps["classifier"]
        scaled = model.named_steps["scaler"].transform(row)
        explainer = shap.TreeExplainer(classifier)
        values = explainer(scaled)
        explanation = shap.Explanation(values.values[0, :, 1] if values.values.ndim == 3 else values.values[0], base_values=values.base_values[0, 1] if np.ndim(values.base_values) > 1 else values.base_values[0], data=row.iloc[0].values, feature_names=columns)
        st.pyplot(shap.plots.waterfall(explanation, max_display=15, show=False).figure, clear_figure=True)
    except Exception as error:
        st.info(f"Explicação SHAP indisponível nesta execução: {error}")

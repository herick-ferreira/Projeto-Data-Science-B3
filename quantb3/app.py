"""Ponto de entrada do dashboard Streamlit."""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="QuantB3", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.hero{padding:1.4rem;border-radius:14px;background:linear-gradient(110deg,#0f172a,#0b4f6c);color:white}.disclaimer{padding:1rem;border-left:5px solid #211604;background:#fff7ed;color:#211604}</style>""", unsafe_allow_html=True)
st.markdown("""<div class='hero'><h1>QuantB3</h1><p>Sinais quantitativos e gestão de risco para ações da B3</p></div>""", unsafe_allow_html=True)
st.info("Navegue pelas páginas no menu lateral. Dados de mercado são obtidos via yfinance.")
st.markdown("<div class='disclaimer'>⚠️ Uso educacional e de portfólio. Não constitui recomendação de investimento.</div>", unsafe_allow_html=True)
st.image(BASE_DIR / "static" / "says-word-financial-it.jpg", caption="Imagem: Say's Word Financial IT", width='stretch')

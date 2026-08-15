"""Configurações centralizadas do projeto."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA",
    "RENT3.SA", "ABEV3.SA", "MGLU3.SA", "B3SA3.SA", "TOTS3.SA",
    "PRIO3.SA", "RADL3.SA", "HAPV3.SA", "CSAN3.SA", "BBAS3.SA",
]
BENCHMARK = "^BVSP"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
FORECAST_HORIZON = 5
TARGET_RETURN = 0.005

SECTORS = {
    "PETR4.SA": "Commodities", "VALE3.SA": "Commodities", "PRIO3.SA": "Commodities",
    "ITUB4.SA": "Bancos", "BBDC4.SA": "Bancos", "B3SA3.SA": "Financeiro",
    "WEGE3.SA": "Industrial", "RENT3.SA": "Consumo", "ABEV3.SA": "Consumo",
    "MGLU3.SA": "Varejo", "TOTS3.SA": "Tecnologia", "BBAS3.SA": "Bancos",
    "RADL3.SA": "Varejo", "HAPV3.SA": "Saúde", "CSAN3.SA": "Utilities",
}

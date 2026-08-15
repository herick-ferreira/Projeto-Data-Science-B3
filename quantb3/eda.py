"""Visualizações exploratórias exigidas para o notebook ou relatório."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.stats import kurtosis, norm, skew


def return_distribution(frame: pd.DataFrame, ticker: str) -> None:
    """Histograma com informação de curtose/assimetria, evidenciando fat tails."""
    returns = frame["return"].dropna()
    sns.histplot(returns, kde=True, stat="density")
    x = pd.Series(returns).sort_values()
    plt.plot(x, norm.pdf(x, returns.mean(), returns.std()), label="Normal ajustada")
    plt.legend()
    plt.title(f"{ticker} | curtose={kurtosis(returns):.2f}, assimetria={skew(returns):.2f}")
    plt.xlabel("Retorno diário")


def correlation_heatmap(prices: dict[str, pd.DataFrame]):
    """Heatmap Plotly para análise de diversificação."""
    returns = pd.DataFrame({ticker: frame["Close"].pct_change() for ticker, frame in prices.items()})
    return px.imshow(returns.corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")


def drawdown_chart(prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame):
    """Drawdowns históricos por ativo contra o Ibovespa."""
    series = {ticker: frame["Close"] / frame["Close"].cummax() - 1 for ticker, frame in prices.items()}
    series["Ibovespa"] = benchmark["Close"] / benchmark["Close"].cummax() - 1
    figure = px.line(pd.DataFrame(series), labels={"value": "Drawdown", "index": "Data"})
    for date, label in [("2015-09-01", "Crise 2015–16"), ("2020-03-01", "COVID-19"), ("2022-01-01", "2022")]:
        figure.add_vline(x=date, line_dash="dot", annotation_text=label)
    return figure


def discriminative_boxplot(dataset: pd.DataFrame) -> None:
    """Compara três features relevantes por classe do target."""
    melted = dataset.melt(id_vars="target", value_vars=["rsi_14", "bb_pct_b", "relative_volume_20d"], var_name="feature", value_name="value")
    sns.boxplot(data=melted, x="feature", y="value", hue="target")

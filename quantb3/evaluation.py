"""Gráficos de avaliação, interpretabilidade e risco para relatório/notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import TimeSeriesSplit


def classification_plots(y_true: pd.Series, probability: np.ndarray, threshold: float) -> None:
    """Plota ROC e matriz de confusão para um classificador binário."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_true, probability, ax=axes[0])
    ConfusionMatrixDisplay.from_predictions(y_true, probability >= threshold, ax=axes[1])
    fig.tight_layout()


def temporal_learning_curve(pipeline, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Curva de aprendizagem temporal; não embaralha o conjunto financeiro."""
    rows = []
    ordered = data.sort_index()
    for fraction in np.linspace(0.3, 1.0, 6):
        subset = ordered.iloc[:int(len(ordered) * fraction)]
        for train_idx, valid_idx in TimeSeriesSplit(n_splits=4).split(subset):
            model = clone(pipeline).fit(subset.iloc[train_idx][columns], subset.iloc[train_idx].target)
            rows.append({"train_size": len(train_idx), "train_score": model.score(subset.iloc[train_idx][columns], subset.iloc[train_idx].target), "validation_score": model.score(subset.iloc[valid_idx][columns], subset.iloc[valid_idx].target)})
    result = pd.DataFrame(rows).groupby("train_size").mean()
    result.plot(marker="o", title="Curva de aprendizagem temporal", ylabel="Accuracy")
    return result


def risk_distribution_plot(returns: pd.Series, var: float, cvar: float) -> None:
    """Distribuição com perdas VaR e CVaR destacadas."""
    plt.figure(figsize=(9, 4))
    plt.hist(returns.dropna(), bins=60, density=True, alpha=.65)
    plt.axvline(var, color="red", label="VaR histórico")
    plt.axvline(cvar, color="darkred", linestyle="--", label="CVaR")
    plt.legend()
    plt.title("Distribuição de retornos, VaR e CVaR")


def shap_plots(pipeline, features: pd.DataFrame, row_index: int = 0) -> None:
    """Beeswarm top-15 e waterfall de uma observação para o LightGBM."""
    scaled = pipeline.named_steps["scaler"].transform(features)
    explanation = shap.TreeExplainer(pipeline.named_steps["classifier"])(scaled)
    # Para classificação binária, a classe positiva ocupa o último eixo quando presente.
    values = explanation[:, :, 1] if explanation.values.ndim == 3 else explanation
    values.feature_names = list(features.columns)
    shap.plots.beeswarm(values, max_display=15)
    shap.plots.waterfall(values[row_index], max_display=15)

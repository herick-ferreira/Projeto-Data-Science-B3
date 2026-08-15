"""Métricas de risco históricas e backtesting de VaR."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def historical_var_cvar(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.DataFrame:
    """Calcula VaR/CVaR históricos rolling; valores negativos representam perda."""
    var = returns.rolling(window).quantile(1 - confidence)
    cvar = returns.rolling(window).apply(lambda x: x[x <= np.quantile(x, 1 - confidence)].mean(), raw=False)
    return pd.DataFrame({"var": var, "cvar": cvar})


def kupiec_test(returns: pd.Series, var: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    """Teste de cobertura incondicional de Kupiec para violações de VaR."""
    valid = pd.concat([returns.rename("return"), var.rename("var")], axis=1).dropna()
    violations = (valid["return"] < valid["var"]).sum()
    n_obs, expected_prob = len(valid), 1 - confidence
    if n_obs == 0 or violations in (0, n_obs):
        return {"observations": n_obs, "violations": int(violations), "p_value": float("nan")}
    observed_prob = violations / n_obs
    lr = -2 * np.log(((1 - expected_prob) ** (n_obs - violations) * expected_prob ** violations) /
                     ((1 - observed_prob) ** (n_obs - violations) * observed_prob ** violations))
    return {"observations": n_obs, "violations": int(violations), "p_value": float(1 - chi2.cdf(lr, 1))}


def portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray, benchmark: pd.Series) -> dict[str, float]:
    """Produz risco diário de uma carteira de pesos informados."""
    portfolio = returns.mul(weights, axis=1).sum(axis=1).dropna()
    benchmark = benchmark.reindex(portfolio.index).dropna()
    portfolio = portfolio.reindex(benchmark.index)
    var95 = portfolio.quantile(0.05)
    var99 = portfolio.quantile(0.01)
    return {
        "var_95": var95, "var_99": var99, "cvar_95": portfolio[portfolio <= var95].mean(),
        "annual_volatility": portfolio.std() * np.sqrt(252),
        "beta": portfolio.cov(benchmark) / benchmark.var(),
    }

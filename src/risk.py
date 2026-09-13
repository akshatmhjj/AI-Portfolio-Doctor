"""Risk, diversification and risk-decomposition analytics.

All VaR/CVaR figures are returned as **positive loss fractions** (e.g. 0.031
means a 3.1% loss), which is the convention risk desks use when quoting VaR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Value at Risk / Expected Shortfall
# ---------------------------------------------------------------------------
def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (non-parametric) VaR at ``confidence``.

    The VaR is the loss at the ``(1 - confidence)`` quantile of the empirical
    return distribution, reported as a positive number.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    quantile = np.percentile(r, (1.0 - confidence) * 100.0)
    return float(-quantile)


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall: mean loss *beyond* the VaR level."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    quantile = np.percentile(r, (1.0 - confidence) * 100.0)
    tail = r[r <= quantile]
    if tail.empty:
        return float(-quantile)
    return float(-tail.mean())


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR: -(mu + z * sigma).

    Provided alongside the historical figure so the two methodologies can be
    compared; large gaps flag non-normal (fat-tailed) return distributions.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)  # negative for typical confidences
    return float(-(mu + z * sigma))


def var_dollar(var_fraction: float, portfolio_value: float) -> float:
    """Convert a fractional VaR into a currency amount."""
    return float(var_fraction * portfolio_value)


# ---------------------------------------------------------------------------
# Covariance / correlation
# ---------------------------------------------------------------------------
def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix."""
    return asset_returns.corr()


def covariance_matrix(
    asset_returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
    annualize: bool = True,
) -> pd.DataFrame:
    """Sample covariance matrix, annualised by default."""
    cov = asset_returns.cov()
    if annualize:
        cov = cov * periods_per_year
    return cov


def average_correlation(asset_returns: pd.DataFrame) -> float:
    """Mean of the off-diagonal correlations (portfolio 'crowdedness')."""
    if asset_returns.shape[1] < 2:
        return float("nan")
    corr = asset_returns.corr().to_numpy()
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off_diag))


def portfolio_volatility(weights, cov: pd.DataFrame) -> float:
    """Portfolio standard deviation from weights and a covariance matrix.

    ``sqrt(w' * Sigma * w)``. If ``cov`` is annualised the result is annualised.
    """
    w = np.asarray(weights, dtype=float)
    C = cov.to_numpy() if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    variance = float(w @ C @ w)
    variance = max(variance, 0.0)  # guard tiny negative round-off
    return float(np.sqrt(variance))


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
def herfindahl_index(weights) -> float:
    """Herfindahl-Hirschman Index = sum of squared (normalised) weights.

    Ranges from ``1/n`` (perfectly equal-weighted) to ``1`` (single holding).
    """
    w = np.abs(np.asarray(weights, dtype=float))
    total = w.sum()
    if total == 0:
        return float("nan")
    w = w / total
    return float(np.sum(w ** 2))


def effective_holdings(weights) -> float:
    """Effective number of holdings = 1 / HHI (a diversification count)."""
    h = herfindahl_index(weights)
    if not h or np.isnan(h):
        return float("nan")
    return float(1.0 / h)


def normalized_hhi(weights) -> float:
    """HHI rescaled to [0, 1]: 0 = equal-weighted, 1 = fully concentrated."""
    n = len(weights)
    if n <= 1:
        return 1.0
    h = herfindahl_index(weights)
    return float((h - 1.0 / n) / (1.0 - 1.0 / n))


# ---------------------------------------------------------------------------
# Risk decomposition (marginal / component contribution to risk)
# ---------------------------------------------------------------------------
def risk_contributions(weights, cov: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio risk into per-asset contributions.

    Returns a DataFrame indexed by ticker with columns:

    * ``weight``            - the portfolio weight
    * ``mctr``              - marginal contribution to risk = (Sigma w)_i / sigma_p
    * ``cctr``              - component contribution = w_i * mctr_i (sums to sigma_p)
    * ``pct_contribution``  - cctr / sigma_p (sums to 1)

    The identity ``sum_i cctr_i = sigma_p`` is what makes this an exact,
    additive decomposition of total portfolio volatility.
    """
    if isinstance(cov, pd.DataFrame):
        index = cov.index
        C = cov.to_numpy()
    else:
        C = np.asarray(cov, dtype=float)
        index = pd.RangeIndex(C.shape[0])

    w = np.asarray(weights, dtype=float)
    port_var = float(w @ C @ w)
    port_vol = np.sqrt(max(port_var, 0.0))

    if port_vol == 0:
        zeros = np.zeros_like(w)
        mctr = zeros
        cctr = zeros
        pct = zeros
    else:
        mctr = (C @ w) / port_vol
        cctr = w * mctr
        pct = cctr / port_vol

    return pd.DataFrame(
        {
            "weight": w,
            "mctr": mctr,
            "cctr": cctr,
            "pct_contribution": pct,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Distribution shape (used by the diagnostics engine & the VaR expander)
# ---------------------------------------------------------------------------
def return_distribution_stats(returns: pd.Series) -> dict:
    """Skewness and excess kurtosis of a return series."""
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return {"skew": float("nan"), "excess_kurtosis": float("nan")}
    return {
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),
    }

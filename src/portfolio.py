"""Return-based portfolio performance analytics.

Conventions
-----------
* Inputs are **simple daily returns** (``price.pct_change()``) unless noted.
* Annualisation uses a 252 trading-day year.
* A constant-weight (daily-rebalanced) portfolio is assumed: the portfolio
  return on day *t* is ``sum_i w_i * r_{i,t}``. This is the standard convention
  for risk analytics because it keeps the weight vector fixed, which is what the
  covariance-based risk decomposition in :mod:`src.risk` requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def normalize_weights(weights) -> pd.Series:
    """Return weights rescaled to sum to 1 (drops NaNs).

    Raises
    ------
    ValueError
        If the weights sum to zero.
    """
    s = pd.Series(weights, dtype=float).dropna()
    total = s.sum()
    if total == 0:
        raise ValueError("Portfolio weights sum to zero.")
    return s / total


def align_weights(weights, columns) -> np.ndarray:
    """Return a numpy weight vector aligned to ``columns`` order."""
    if isinstance(weights, pd.Series):
        w = weights.reindex(columns).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if w.shape[0] != len(columns):
        raise ValueError("Number of weights does not match number of assets.")
    if np.isnan(w).any():
        raise ValueError("Weights contain NaN after alignment (missing ticker?).")
    return w


def portfolio_returns(asset_returns: pd.DataFrame, weights) -> pd.Series:
    """Daily portfolio return series for a constant-weight portfolio.

    Computed as the matrix product ``R · w`` (each day's return is the
    weighted sum of holding returns), returned as a Series on the return index.
    """
    w = align_weights(weights, asset_returns.columns)
    return asset_returns.dot(w)


# ---------------------------------------------------------------------------
# Growth / return
# ---------------------------------------------------------------------------
def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative (compounded) return path, starting near 0."""
    return (1.0 + returns).cumprod() - 1.0


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the full sample."""
    if len(returns) == 0:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return (CAGR)."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0  # portfolio was wiped out
    return growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns (sample std, ddof=1)."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio.

    ``risk_free_rate`` is an *annual* rate; it is converted to a per-period rate
    and subtracted from each daily return. Sharpe = sqrt(P) * mean(excess) / sd.
    """
    if len(returns) < 2:
        return float("nan")
    rf_period = risk_free_rate / periods_per_year
    excess = returns - rf_period
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio (uses downside deviation, not total sigma)."""
    if len(returns) < 2:
        return float("nan")
    rf_period = risk_free_rate / periods_per_year
    excess = returns - rf_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0 or np.isnan(downside_dev):
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_dev)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path: wealth relative to its running peak (<= 0)."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number)."""
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


# ---------------------------------------------------------------------------
# Market sensitivity
# ---------------------------------------------------------------------------
def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta of ``returns`` versus ``market_returns`` (cov / market variance)."""
    df = pd.concat([returns, market_returns], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    market_var = cov[1, 1]
    if market_var == 0:
        return float("nan")
    return float(cov[0, 1] / market_var)


def asset_betas(asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """Beta of every asset column versus the market."""
    return pd.Series(
        {col: beta(asset_returns[col], market_returns) for col in asset_returns.columns}
    )


# ---------------------------------------------------------------------------
# Convenience: full performance summary
# ---------------------------------------------------------------------------
def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Bundle the headline performance metrics into a dict."""
    return {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }

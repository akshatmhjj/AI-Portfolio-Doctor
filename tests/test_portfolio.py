"""Tests for return-based portfolio performance analytics."""
import numpy as np
import pandas as pd

from src import portfolio as pf


def test_normalize_weights_sums_to_one():
    w = pf.normalize_weights({"A": 25, "B": 25, "C": 50})
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(w["C"] - 0.5) < 1e-12


def test_portfolio_returns_is_weighted_sum():
    df = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.03, -0.01, 0.00]})
    w = pd.Series({"A": 0.5, "B": 0.5})
    pr = pf.portfolio_returns(df, w)
    expected = 0.5 * df["A"] + 0.5 * df["B"]
    pd.testing.assert_series_equal(pr, expected, check_names=False)


def test_annualized_volatility_matches_std(asset_returns):
    r = asset_returns["A"]
    expected = r.std(ddof=1) * np.sqrt(252)
    assert abs(pf.annualized_volatility(r) - expected) < 1e-12


def test_annualized_return_is_cagr():
    r = pd.Series([0.001] * 126)  # half a trading year of constant returns
    growth = (1.001) ** 126
    expected = growth ** (252 / 126) - 1
    assert abs(pf.annualized_return(r) - expected) < 1e-10


def test_sharpe_zero_rf(asset_returns):
    r = asset_returns["A"]
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert abs(pf.sharpe_ratio(r, 0.0) - expected) < 1e-10


def test_sharpe_with_risk_free(asset_returns):
    r = asset_returns["A"]
    rf = 0.04
    excess = r - rf / 252
    expected = np.sqrt(252) * excess.mean() / r.std(ddof=1)
    assert abs(pf.sharpe_ratio(r, rf) - expected) < 1e-10


def test_sortino_uses_downside_deviation(asset_returns):
    r = asset_returns["A"]
    rf = 0.02
    excess = r - rf / 252
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    expected = np.sqrt(252) * excess.mean() / dd
    assert abs(pf.sortino_ratio(r, rf) - expected) < 1e-10


def test_max_drawdown_known_path():
    # wealth: 1.1, 0.88, 0.924 -> peak 1.1 -> min drawdown = 0.88/1.1 - 1 = -0.2
    r = pd.Series([0.10, -0.20, 0.05])
    assert abs(pf.max_drawdown(r) - (-0.20)) < 1e-12


def test_max_drawdown_non_positive(asset_returns):
    assert pf.max_drawdown(asset_returns["A"]) <= 0.0


def test_beta_exact_multiple(market_returns):
    asset = 1.5 * market_returns
    assert abs(pf.beta(asset, market_returns) - 1.5) < 1e-9


def test_beta_of_market_is_one(market_returns):
    assert abs(pf.beta(market_returns, market_returns) - 1.0) < 1e-9


def test_total_return_compounds():
    r = pd.Series([0.1, 0.1])
    assert abs(pf.total_return(r) - (1.1 * 1.1 - 1)) < 1e-12

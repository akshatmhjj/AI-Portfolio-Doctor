"""Tests for risk, diversification and risk-decomposition analytics."""
import numpy as np
import pandas as pd
from scipy import stats

from src import risk


# --------------------------------------------------------------------------- VaR / CVaR
def test_historical_var_is_negative_quantile():
    r = pd.Series([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    expected = -np.percentile(r, 5)
    assert abs(risk.historical_var(r, 0.95) - expected) < 1e-12


def test_cvar_equals_mean_of_tail():
    r = pd.Series(np.linspace(-0.10, 0.10, 101))
    q = np.percentile(r, 5)
    tail = r[r <= q]
    expected = -tail.mean()
    assert abs(risk.conditional_var(r, 0.95) - expected) < 1e-12


def test_cvar_at_least_var(asset_returns):
    # Expected shortfall is never smaller than VaR (same confidence).
    r = asset_returns["A"]
    assert risk.conditional_var(r, 0.95) >= risk.historical_var(r, 0.95) - 1e-9


def test_parametric_var_formula(asset_returns):
    r = asset_returns["A"]
    z = stats.norm.ppf(0.05)
    expected = -(r.mean() + z * r.std(ddof=1))
    assert abs(risk.parametric_var(r, 0.95) - expected) < 1e-12


def test_var_dollar():
    assert abs(risk.var_dollar(0.03, 100_000) - 3_000) < 1e-9


# --------------------------------------------------------------------------- Concentration
def test_hhi_equal_weights():
    assert abs(risk.herfindahl_index([0.25, 0.25, 0.25, 0.25]) - 0.25) < 1e-12


def test_hhi_single_asset():
    assert abs(risk.herfindahl_index([1.0]) - 1.0) < 1e-12


def test_hhi_normalizes_inputs():
    # Unnormalised weights should be normalised internally.
    assert abs(risk.herfindahl_index([2.0, 2.0]) - 0.5) < 1e-12


def test_effective_holdings():
    assert abs(risk.effective_holdings([0.25, 0.25, 0.25, 0.25]) - 4.0) < 1e-12


def test_normalized_hhi_equal_is_zero():
    assert abs(risk.normalized_hhi([0.2] * 5) - 0.0) < 1e-12


def test_normalized_hhi_single_is_one():
    assert abs(risk.normalized_hhi([1.0]) - 1.0) < 1e-12


# --------------------------------------------------------------------------- Covariance / vol
def test_covariance_annualization(asset_returns):
    daily = risk.covariance_matrix(asset_returns, annualize=False)
    annual = risk.covariance_matrix(asset_returns, annualize=True)
    assert np.allclose(annual.to_numpy(), daily.to_numpy() * 252)


def test_portfolio_volatility_diagonal():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    w = np.array([0.5, 0.5])
    expected = np.sqrt(0.25 * 0.04 + 0.25 * 0.09)
    assert abs(risk.portfolio_volatility(w, cov) - expected) < 1e-12


def test_portfolio_volatility_matches_quadratic_form(asset_returns):
    cov = risk.covariance_matrix(asset_returns)
    w = np.array([0.5, 0.3, 0.2])
    expected = np.sqrt(w @ cov.to_numpy() @ w)
    assert abs(risk.portfolio_volatility(w, cov) - expected) < 1e-12


# --------------------------------------------------------------------------- Risk decomposition
def test_component_contributions_sum_to_portfolio_vol(asset_returns):
    cov = risk.covariance_matrix(asset_returns)
    w = np.array([0.5, 0.3, 0.2])
    rc = risk.risk_contributions(w, cov)
    port_vol = risk.portfolio_volatility(w, cov)
    assert abs(rc["cctr"].sum() - port_vol) < 1e-10


def test_pct_contributions_sum_to_one(asset_returns):
    cov = risk.covariance_matrix(asset_returns)
    w = np.array([0.4, 0.4, 0.2])
    rc = risk.risk_contributions(w, cov)
    assert abs(rc["pct_contribution"].sum() - 1.0) < 1e-10


def test_equal_symmetric_assets_have_equal_contributions():
    # Same variance, same correlation, equal weights -> equal risk shares.
    cov = pd.DataFrame(
        [[0.04, 0.02, 0.02], [0.02, 0.04, 0.02], [0.02, 0.02, 0.04]],
        index=list("ABC"), columns=list("ABC"),
    )
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    rc = risk.risk_contributions(w, cov)
    assert np.allclose(rc["pct_contribution"].to_numpy(), 1 / 3, atol=1e-12)


def test_average_correlation_matches_offdiagonal_mean(asset_returns):
    corr = asset_returns.corr().to_numpy()
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)]
    assert abs(risk.average_correlation(asset_returns) - off.mean()) < 1e-12

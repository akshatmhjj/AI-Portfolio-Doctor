"""Tests for the interpretable health-scoring engine."""
import numpy as np

from src import diagnostics as dg


def _healthy_metrics():
    return {
        "annualized_volatility": 0.11,
        "max_drawdown": -0.12,
        "var": 0.012,
        "cvar": 0.018,
        "stress_loss": -0.16,
        "hhi": 0.16,
        "effective_holdings": 6.2,
        "max_weight": 0.22,
        "max_weight_ticker": "A",
        "average_correlation": 0.25,
        "beta": 0.9,
        "skew": -0.1,
        "excess_kurtosis": 1.0,
        "n_assets": 6,
    }


def _risky_metrics():
    return {
        "annualized_volatility": 0.45,
        "max_drawdown": -0.55,
        "var": 0.07,
        "cvar": 0.09,
        "stress_loss": -0.50,
        "hhi": 0.8,
        "effective_holdings": 1.25,
        "max_weight": 0.8,
        "max_weight_ticker": "A",
        "average_correlation": 0.85,
        "beta": 1.6,
        "skew": -1.2,
        "excess_kurtosis": 6.0,
        "n_assets": 2,
    }


def test_score_within_range():
    for m in (_healthy_metrics(), _risky_metrics()):
        s = dg.compute_health(m)["score"]
        assert 0 <= s <= 100


def test_healthy_scores_higher_than_risky():
    assert dg.compute_health(_healthy_metrics())["score"] > dg.compute_health(_risky_metrics())["score"]


def test_healthy_is_low_risk_band():
    assert dg.compute_health(_healthy_metrics())["band"] == "Low Risk"


def test_risky_is_high_risk_band():
    assert dg.compute_health(_risky_metrics())["band"] == "High Risk"


def test_volatility_is_monotonic():
    low = _healthy_metrics()
    high = dict(low, annualized_volatility=0.38)
    assert dg.compute_health(high)["score"] <= dg.compute_health(low)["score"]


def test_ramp_score_endpoints():
    assert abs(dg._ramp_score(0.10, 0.10, 0.40) - 100.0) < 1e-9
    assert abs(dg._ramp_score(0.40, 0.10, 0.40) - 0.0) < 1e-9
    assert abs(dg._ramp_score(0.25, 0.10, 0.40) - 50.0) < 1e-9


def test_bands():
    assert dg.band_for_score(85) == "Low Risk"
    assert dg.band_for_score(70) == "Moderate Risk"
    assert dg.band_for_score(50) == "Elevated Risk"
    assert dg.band_for_score(20) == "High Risk"


def test_concentration_diagnostic_present():
    titles = [d["title"] for d in dg.build_diagnostics(_risky_metrics())]
    assert any("concentration" in t.lower() for t in titles)


def test_correlation_undefined_weight_redistributes():
    # Single-asset portfolio: correlation sub-score is NaN but overall score is finite.
    m = _healthy_metrics()
    m["average_correlation"] = float("nan")
    health = dg.compute_health(m)
    assert not np.isnan(health["score"])
    assert np.isnan(health["subscores"]["correlation"])

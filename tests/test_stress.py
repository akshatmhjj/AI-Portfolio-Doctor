"""Tests for the stress-testing engine."""
import numpy as np
import pandas as pd

from src import stress


def test_apply_shocks_basic():
    w = pd.Series({"A": 0.6, "B": 0.4})
    shocks = pd.Series({"A": -0.10, "B": -0.20})
    res = stress.apply_shocks(w, shocks, portfolio_value=1000.0, scenario="t")
    assert abs(res.portfolio_return - (-0.14)) < 1e-12
    assert abs(res.pnl - (-140.0)) < 1e-9
    assert abs(res.ending_value - 860.0) < 1e-9


def test_apply_shocks_contributions_sum_to_return():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    shocks = pd.Series({"A": -0.10, "B": -0.05, "C": 0.02})
    res = stress.apply_shocks(w, shocks, 100_000.0)
    assert abs(res.detail["contribution_to_return"].sum() - res.portfolio_return) < 1e-12
    assert abs(res.detail["contribution_to_pnl"].sum() - res.pnl) < 1e-6


def test_missing_shocks_treated_as_zero():
    w = pd.Series({"A": 0.5, "B": 0.5})
    shocks = pd.Series({"A": -0.10})  # B missing
    res = stress.apply_shocks(w, shocks, 1000.0)
    assert abs(res.portfolio_return - (-0.05)) < 1e-12


def test_beta_adjusted_shocks():
    betas = pd.Series({"A": 1.5, "B": 0.5})
    shocks = stress.beta_adjusted_shocks(betas, -0.20)
    assert abs(shocks["A"] - (-0.30)) < 1e-12
    assert abs(shocks["B"] - (-0.10)) < 1e-12


def test_technology_selloff_targets_tech():
    sectors = {"A": "Technology", "B": "Energy"}
    betas = pd.Series({"A": 1.4, "B": 0.7})
    shocks = stress.technology_selloff(sectors, betas, tech_shock=-0.25, market_spillover=-0.05)
    assert abs(shocks["A"] - (-0.25)) < 1e-12          # direct tech shock
    assert abs(shocks["B"] - (0.7 * -0.05)) < 1e-12    # beta-scaled spillover


def test_market_crash_uses_default():
    betas = pd.Series({"A": 1.0, "B": 2.0})
    shocks = stress.market_crash(betas)
    assert abs(shocks["A"] - (-0.20)) < 1e-12
    assert abs(shocks["B"] - (-0.40)) < 1e-12

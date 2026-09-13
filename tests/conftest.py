"""Shared pytest fixtures and path setup.

Ensures the repository root is importable so ``from src import ...`` works when
pytest is invoked from anywhere, and provides deterministic synthetic data so
none of the tests touch the network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=500)


@pytest.fixture
def asset_returns(rng, dates):
    """Three correlated assets of daily returns."""
    n = len(dates)
    market = rng.normal(0.0004, 0.011, n)
    a = 1.2 * market + rng.normal(0, 0.008, n)
    b = 0.8 * market + rng.normal(0, 0.006, n)
    c = 0.3 * market + rng.normal(0, 0.014, n)
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


@pytest.fixture
def market_returns(rng, dates):
    return pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates, name="MKT")


@pytest.fixture
def price_series(rng, dates):
    steps = rng.normal(0.0005, 0.012, len(dates))
    return pd.Series(100 * np.cumprod(1 + steps), index=dates, name="PX")

"""Tests for technical indicators."""
import numpy as np
import pandas as pd

from src import technicals as ta


def test_sma_last_value():
    s = pd.Series(range(1, 11), dtype=float)  # 1..10
    result = ta.sma(s, 3)
    assert abs(result.iloc[-1] - 9.0) < 1e-12  # mean(8,9,10)
    assert np.isnan(result.iloc[0])            # warm-up


def test_rsi_bounds(price_series):
    r = ta.rsi(price_series, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))  # strictly increasing -> no losses
    r = ta.rsi(s, 14)
    assert abs(r.iloc[-1] - 100.0) < 1e-9


def test_macd_histogram_identity(price_series):
    m = ta.macd(price_series)
    assert np.allclose((m["macd"] - m["signal"]).to_numpy(), m["histogram"].to_numpy(), atol=1e-12)


def test_bollinger_band_width(price_series):
    bb = ta.bollinger_bands(price_series, window=20, num_std=2.0)
    std = price_series.rolling(20, min_periods=20).std(ddof=0)
    width = (bb["upper"] - bb["lower"]).dropna()
    expected = (4.0 * std).dropna()
    assert np.allclose(width.to_numpy(), expected.to_numpy(), atol=1e-9)


def test_bollinger_middle_is_sma(price_series):
    bb = ta.bollinger_bands(price_series, window=20)
    sma20 = ta.sma(price_series, 20)
    assert np.allclose(bb["middle"].dropna().to_numpy(), sma20.dropna().to_numpy(), atol=1e-12)

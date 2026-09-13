"""Market-data retrieval and cleaning.

Responsibilities
----------------
* Download adjusted prices from Yahoo Finance (yfinance) with caching.
* Normalise the many shapes yfinance can return into a tidy price frame.
* Clean the data: drop dead tickers, forward-fill internal gaps, align dates.
* Convert prices to simple daily returns.

The module is UI-agnostic. If Streamlit is importable its ``cache_data``
decorator is used to memoise network calls; otherwise a no-op decorator keeps
the module fully importable inside plain unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Optional Streamlit caching (degrades gracefully outside a Streamlit runtime).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import shim
    import streamlit as st

    def cache_data(**kwargs):
        return st.cache_data(**kwargs)

except ModuleNotFoundError:  # pragma: no cover - exercised only without streamlit

    def cache_data(**kwargs):
        def _decorator(func):
            return func

        return _decorator


BENCHMARK_TICKER = "SPY"


@dataclass
class MarketData:
    """Container for a cleaned market-data pull.

    Attributes
    ----------
    prices : Adjusted close prices for the *valid* portfolio tickers.
    returns : Simple daily returns for the valid portfolio tickers.
    benchmark_prices / benchmark_returns : Same, for the benchmark (SPY).
    valid_tickers : Requested tickers that returned usable data.
    invalid_tickers : Requested tickers that returned no usable data.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_prices: pd.Series
    benchmark_returns: pd.Series
    valid_tickers: list = field(default_factory=list)
    invalid_tickers: list = field(default_factory=list)


def clean_tickers(tickers: Iterable[str]) -> list:
    """Upper-case, strip and de-duplicate a list of tickers (order preserved)."""
    seen = {}
    for t in tickers:
        if t is None:
            continue
        s = str(t).strip().upper()
        if s:
            seen.setdefault(s, None)
    return list(seen.keys())


def _extract_close(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Pull the adjusted-close block out of a yfinance download.

    yfinance returns either a MultiIndex column frame (multiple tickers) or a
    flat frame (single ticker), and the level ordering depends on ``group_by``.
    This helper handles every common case.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        level0 = set(cols.get_level_values(0))
        level1 = set(cols.get_level_values(1))
        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" in cols:
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]] if tickers else ["Close"]
        else:
            close = raw.copy()

    return close


def clean_prices(close: pd.DataFrame) -> pd.DataFrame:
    """Drop empty tickers, forward-fill internal gaps and align start dates."""
    if close is None or close.empty:
        return pd.DataFrame()
    close = close.dropna(axis=1, how="all")          # remove dead tickers
    close = close.sort_index()
    close = close.ffill()                            # bridge sporadic gaps
    close = close.dropna(how="any")                  # align to common history
    return close


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price frame."""
    return prices.pct_change().dropna(how="all")


@cache_data(show_spinner=False, ttl=3600)
def download_prices(tickers: tuple, start, end) -> pd.DataFrame:
    """Download adjusted-close prices. Cached for one hour per argument set.

    ``tickers`` is a tuple (hashable) so Streamlit can key the cache on it.
    """
    import yfinance as yf  # imported lazily so tests never require network libs

    ticker_list = clean_tickers(tickers)
    if not ticker_list:
        return pd.DataFrame()

    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    close = _extract_close(raw, ticker_list)
    # Keep only requested tickers that came back, in the requested order.
    available = [t for t in ticker_list if t in close.columns]
    return close[available] if available else pd.DataFrame()


@cache_data(show_spinner=False, ttl=86400)
def fetch_sectors(tickers: tuple) -> dict:
    """Best-effort GICS sector lookup for stress-test classification.

    yfinance's ``.info`` endpoint is slow and occasionally unavailable, so this
    is wrapped defensively and defaults to ``"Unknown"``.
    """
    import yfinance as yf

    out = {}
    for t in clean_tickers(tickers):
        try:
            info = yf.Ticker(t).info or {}
            out[t] = info.get("sector") or "Unknown"
        except Exception:  # pragma: no cover - network dependent
            out[t] = "Unknown"
    return out


def load_market_data(
    tickers: Iterable[str],
    start,
    end,
    benchmark: str = BENCHMARK_TICKER,
) -> MarketData:
    """Download and clean prices for the portfolio plus a benchmark.

    Returns a :class:`MarketData` bundle. Tickers that yield no usable data are
    reported in ``invalid_tickers`` rather than raising, so the UI can warn and
    continue with whatever is valid.
    """
    requested = clean_tickers(tickers)
    all_symbols = tuple(dict.fromkeys(requested + [benchmark.upper()]))

    raw_close = download_prices(all_symbols, _as_date(start), _as_date(end))
    prices = clean_prices(raw_close)

    if prices.empty:
        return MarketData(
            prices=pd.DataFrame(),
            returns=pd.DataFrame(),
            benchmark_prices=pd.Series(dtype=float),
            benchmark_returns=pd.Series(dtype=float),
            valid_tickers=[],
            invalid_tickers=requested,
        )

    bench = benchmark.upper()
    benchmark_prices = prices[bench] if bench in prices.columns else pd.Series(dtype=float)

    valid = [t for t in requested if t in prices.columns]
    invalid = [t for t in requested if t not in prices.columns]

    asset_prices = prices[valid] if valid else pd.DataFrame(index=prices.index)
    asset_returns = to_returns(asset_prices) if not asset_prices.empty else pd.DataFrame()
    benchmark_returns = (
        benchmark_prices.pct_change().dropna()
        if not benchmark_prices.empty
        else pd.Series(dtype=float)
    )

    # Align portfolio and benchmark returns on shared dates.
    if not asset_returns.empty and not benchmark_returns.empty:
        common = asset_returns.index.intersection(benchmark_returns.index)
        asset_returns = asset_returns.loc[common]
        benchmark_returns = benchmark_returns.loc[common]

    return MarketData(
        prices=asset_prices,
        returns=asset_returns,
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
        valid_tickers=valid,
        invalid_tickers=invalid,
    )


def _as_date(value):
    """Normalise date-like values so the cache key is stable/hashable."""
    if isinstance(value, date):
        return value.isoformat()
    return str(value)

"""Price-based technical indicators.

These are **analytical descriptors** of price behaviour (trend, momentum,
volatility bands) - not trading signals or buy/sell recommendations. Each
function takes a close-price Series and returns aligned indicator Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` periods."""
    return close.rolling(window=window, min_periods=window).mean()


def moving_averages(close: pd.Series, windows=(20, 50, 200)) -> pd.DataFrame:
    """Convenience: several SMAs at once as a DataFrame."""
    return pd.DataFrame({f"SMA{w}": sma(close, w) for w in windows})


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    RSI = 100 - 100 / (1 + RS), where RS = avg gain / avg loss over ``period``.
    Wilder's smoothing is an exponential average with alpha = 1/period.
    Bounded to [0, 100]; ~70 is conventionally 'overbought', ~30 'oversold'.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    # When there are no losses, RS is +inf -> RSI = 100. Make that explicit.
    result = result.where(avg_loss != 0, 100.0)
    # Preserve NaN during the warm-up window.
    result[avg_gain.isna()] = np.nan
    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns columns ``macd`` (fast EMA - slow EMA), ``signal`` (EMA of MACD)
    and ``histogram`` (macd - signal).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: SMA middle band +/- ``num_std`` rolling std deviations."""
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def compute_all(close: pd.Series) -> pd.DataFrame:
    """Assemble every indicator against the close price into one frame."""
    frame = pd.DataFrame({"close": close})
    frame = frame.join(moving_averages(close))
    frame["RSI14"] = rsi(close, 14)
    frame = frame.join(macd(close))
    bb = bollinger_bands(close).rename(
        columns={"middle": "BB_middle", "upper": "BB_upper", "lower": "BB_lower"}
    )
    frame = frame.join(bb)
    return frame

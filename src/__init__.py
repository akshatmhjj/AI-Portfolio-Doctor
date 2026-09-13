"""AI Portfolio Doctor - financial analytics engine.

This package contains the quantitative core of the application. Every module
here is deliberately free of any Streamlit / UI code so the financial logic can
be unit-tested and reused independently of the web layer.

Modules
-------
data         : Market-data retrieval (yfinance) and cleaning.
portfolio    : Return-based performance analytics.
risk         : Risk, diversification and risk-decomposition analytics.
technicals   : Price-based technical indicators (analytical, not signals).
stress       : Scenario / stress-test engine.
diagnostics  : Interpretable, rules-based portfolio health scoring.
"""

TRADING_DAYS: int = 252

__all__ = [
    "TRADING_DAYS",
    "data",
    "portfolio",
    "risk",
    "technicals",
    "stress",
    "diagnostics",
]

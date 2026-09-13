"""Scenario / stress-test engine.

A scenario is expressed as a per-asset shock vector (fractional price moves,
negative for a loss). Given portfolio weights and a portfolio value, the engine
computes the portfolio return, the ending value and each holding's contribution
to the loss.

Two families of scenarios are provided:

* **Beta-adjusted market scenarios** - a market-wide move is translated into
  asset moves via each holding's beta to the market, so higher-beta names take
  bigger hits (and the portfolio's own beta drives the aggregate loss).
* **Sector scenarios** - a targeted shock to holdings in named sectors, with an
  optional market spillover applied to everything else via beta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StressResult:
    """Outcome of applying a shock vector to a portfolio."""

    scenario: str
    portfolio_return: float          # fractional (negative = loss)
    pnl: float                       # currency P&L
    ending_value: float              # currency value after the shock
    detail: pd.DataFrame             # per-asset shock, weight, contribution


def apply_shocks(
    weights: pd.Series,
    shocks: pd.Series,
    portfolio_value: float,
    scenario: str = "Custom",
) -> StressResult:
    """Apply a per-asset shock vector to a weighted portfolio.

    ``weights`` and ``shocks`` are Series indexed by ticker. Missing shocks are
    treated as 0 (no move). Portfolio return = sum_i w_i * shock_i.
    """
    w = pd.Series(weights, dtype=float)
    s = pd.Series(shocks, dtype=float).reindex(w.index).fillna(0.0)

    contribution = w * s                        # contribution to portfolio return
    portfolio_return = float(contribution.sum())
    pnl = float(portfolio_return * portfolio_value)
    ending_value = float(portfolio_value * (1.0 + portfolio_return))

    detail = pd.DataFrame(
        {
            "weight": w,
            "shock": s,
            "contribution_to_return": contribution,
            "contribution_to_pnl": contribution * portfolio_value,
        }
    )
    detail = detail.sort_values("contribution_to_return")

    return StressResult(
        scenario=scenario,
        portfolio_return=portfolio_return,
        pnl=pnl,
        ending_value=ending_value,
        detail=detail,
    )


def beta_adjusted_shocks(betas: pd.Series, market_shock: float) -> pd.Series:
    """Translate a market move into per-asset moves via beta.

    asset_shock_i = beta_i * market_shock.
    """
    return pd.Series(betas, dtype=float).fillna(1.0) * market_shock


def sector_shocks(
    sectors: dict,
    shock_map: dict,
    betas: pd.Series | None = None,
    spillover: float = 0.0,
) -> pd.Series:
    """Build a shock vector that targets named sectors.

    Parameters
    ----------
    sectors : mapping ticker -> sector label.
    shock_map : mapping sector label -> fractional shock for that sector.
    betas : optional per-asset betas, used to spread ``spillover`` to
        holdings *not* named in ``shock_map``.
    spillover : a market move applied (via beta) to non-targeted holdings.
    """
    tickers = list(sectors.keys())
    betas = pd.Series(betas, dtype=float) if betas is not None else pd.Series(dtype=float)

    values = {}
    for t in tickers:
        sector = sectors.get(t, "Unknown")
        if sector in shock_map:
            values[t] = float(shock_map[sector])
        else:
            b = float(betas.get(t, 1.0)) if not betas.empty else 1.0
            values[t] = b * spillover
    return pd.Series(values)


# ---------------------------------------------------------------------------
# Named scenario builders
# ---------------------------------------------------------------------------
def market_crash(betas: pd.Series, shock: float = -0.20) -> pd.Series:
    """Broad market crash (default -20%), transmitted through beta."""
    return beta_adjusted_shocks(betas, shock)


def severe_stress(betas: pd.Series, shock: float = -0.30) -> pd.Series:
    """Severe market stress (default -30%), transmitted through beta."""
    return beta_adjusted_shocks(betas, shock)


def technology_selloff(
    sectors: dict,
    betas: pd.Series,
    tech_shock: float = -0.25,
    market_spillover: float = -0.05,
) -> pd.Series:
    """Technology-led selloff.

    Technology holdings are shocked by ``tech_shock``; everything else receives
    a smaller market spillover scaled by its beta.
    """
    return sector_shocks(
        sectors,
        shock_map={"Technology": tech_shock},
        betas=betas,
        spillover=market_spillover,
    )

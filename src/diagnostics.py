"""Interpretable, rules-based portfolio health engine.

This is the "diagnostic" core of AI Portfolio Doctor. There is no black-box
model here: the health score is a transparent, weighted blend of six risk
sub-scores, each derived from a metric that is actually computed elsewhere in
the package (volatility, drawdown, VaR, stress loss, concentration and
correlation). Every diagnostic message is traceable to a threshold on a real
number, so the output is fully defensible.

Design
------
* Each sub-score maps a risk metric onto 0-100 where **higher is healthier**
  (lower risk), via a clamped linear ramp between a "good" and a "bad" level.
* The overall score is the weighted average of the available sub-scores. If a
  sub-score is undefined (e.g. correlation for a single-asset portfolio) its
  weight is redistributed across the others.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (good, bad) for each metric. "good" -> score 100, "bad" -> 0.
# All expressed as positive magnitudes.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "volatility": (0.10, 0.40),      # annualised
    "drawdown": (0.10, 0.50),        # max drawdown magnitude
    "var": (0.01, 0.06),             # daily historical VaR magnitude
    "stress": (0.15, 0.45),          # severe-scenario loss magnitude
    "concentration": (0.15, 0.50),   # Herfindahl index (raw)
    "correlation": (0.20, 0.80),     # average pairwise correlation
}

WEIGHTS = {
    "volatility": 0.20,
    "drawdown": 0.20,
    "var": 0.15,
    "stress": 0.15,
    "concentration": 0.15,
    "correlation": 0.15,
}

SEVERITY_RANK = {"danger": 0, "warning": 1, "info": 2, "good": 3}


def _ramp_score(value: float, good: float, bad: float) -> float:
    """Linear 0-100 score where value<=good -> 100 and value>=bad -> 0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if bad == good:
        return 100.0
    score = 100.0 * (bad - value) / (bad - good)
    return float(np.clip(score, 0.0, 100.0))


def band_for_score(score: float) -> str:
    """Map a 0-100 score to a qualitative risk band."""
    if np.isnan(score):
        return "Unknown"
    if score >= 80:
        return "Low Risk"
    if score >= 60:
        return "Moderate Risk"
    if score >= 40:
        return "Elevated Risk"
    return "High Risk"


def compute_subscores(metrics: dict) -> dict:
    """Compute each 0-100 sub-score from the raw metrics dict."""
    sub = {}
    sub["volatility"] = _ramp_score(metrics.get("annualized_volatility"), *THRESHOLDS["volatility"])
    sub["drawdown"] = _ramp_score(abs(metrics.get("max_drawdown", np.nan)), *THRESHOLDS["drawdown"])
    sub["var"] = _ramp_score(metrics.get("var"), *THRESHOLDS["var"])
    sub["stress"] = _ramp_score(abs(metrics.get("stress_loss", np.nan)), *THRESHOLDS["stress"])
    sub["concentration"] = _ramp_score(metrics.get("hhi"), *THRESHOLDS["concentration"])
    corr = metrics.get("average_correlation", np.nan)
    sub["correlation"] = _ramp_score(corr, *THRESHOLDS["correlation"])
    return sub


def _weighted_score(subscores: dict) -> float:
    """Weighted average of the sub-scores that are defined (NaN weights drop)."""
    num = 0.0
    denom = 0.0
    for key, score in subscores.items():
        if score is None or np.isnan(score):
            continue
        w = WEIGHTS[key]
        num += w * score
        denom += w
    if denom == 0:
        return float("nan")
    return num / denom


def build_diagnostics(metrics: dict) -> list:
    """Produce the transparent, human-readable diagnostic messages.

    Each entry is a dict: ``{severity, title, detail}`` where severity is one of
    danger / warning / info / good. Messages are sorted most-severe first.
    """
    d = []

    def add(severity, title, detail):
        d.append({"severity": severity, "title": title, "detail": detail})

    n_assets = metrics.get("n_assets", 0)
    eff = metrics.get("effective_holdings", np.nan)
    max_w = metrics.get("max_weight", np.nan)
    max_w_ticker = metrics.get("max_weight_ticker", "")
    hhi = metrics.get("hhi", np.nan)
    corr = metrics.get("average_correlation", np.nan)
    vol = metrics.get("annualized_volatility", np.nan)
    mdd = abs(metrics.get("max_drawdown", np.nan))
    var = metrics.get("var", np.nan)
    cvar = metrics.get("cvar", np.nan)
    stress = abs(metrics.get("stress_loss", np.nan))
    beta = metrics.get("beta", np.nan)
    kurt = metrics.get("excess_kurtosis", np.nan)
    skew = metrics.get("skew", np.nan)

    # --- Concentration -----------------------------------------------------
    if not np.isnan(hhi):
        if (not np.isnan(eff) and eff < 3) or (not np.isnan(max_w) and max_w > 0.50):
            add(
                "danger",
                "High concentration risk",
                f"Effective holdings ≈ {eff:.1f}"
                + (f"; largest position {max_w:.0%} ({max_w_ticker})." if not np.isnan(max_w) else "."),
            )
        elif not np.isnan(eff) and eff >= 8:
            add(
                "good",
                "Well diversified by weight",
                f"Capital is spread across ≈ {eff:.1f} effective holdings (HHI {hhi:.2f}).",
            )

    # --- Correlation -------------------------------------------------------
    if not np.isnan(corr):
        if corr > 0.70:
            add(
                "warning",
                "High correlation exposure",
                f"Average pairwise correlation is {corr:.2f}; holdings move together, "
                "so diversification benefit is limited.",
            )
        elif corr < 0.30:
            add(
                "good",
                "Strong diversification",
                f"Low average pairwise correlation ({corr:.2f}) means holdings "
                "diversify each other well.",
            )

    # --- Volatility --------------------------------------------------------
    if not np.isnan(vol):
        if vol > 0.30:
            add("warning", "Elevated volatility", f"Annualised volatility is {vol:.1%}.")
        elif vol < 0.12:
            add("good", "Low volatility profile", f"Annualised volatility is {vol:.1%}.")

    # --- Downside: drawdown, VaR, CVaR ------------------------------------
    if not np.isnan(mdd) and mdd > 0.35:
        add("danger", "Severe historical drawdown", f"Worst peak-to-trough loss was {mdd:.1%}.")
    if not np.isnan(var) and var > 0.035:
        extra = f" CVaR {cvar:.1%}." if not np.isnan(cvar) else ""
        add("warning", "Elevated downside risk", f"1-day VaR is {var:.1%}.{extra}")

    # --- Stress sensitivity ------------------------------------------------
    if not np.isnan(stress) and stress > 0.30:
        add(
            "warning",
            "High sensitivity to a market crash",
            f"A severe-stress scenario implies a {stress:.1%} portfolio loss.",
        )

    # --- Market beta -------------------------------------------------------
    if not np.isnan(beta):
        if beta > 1.20:
            add("info", "High market beta", f"Portfolio beta to SPY is {beta:.2f}; it amplifies market moves.")
        elif beta < 0.80:
            add("good", "Defensive market beta", f"Portfolio beta to SPY is {beta:.2f}.")

    # --- Distribution shape (justifies historical over Gaussian VaR) -------
    if not np.isnan(kurt) and kurt > 3.0:
        skew_txt = f" and negative skew ({skew:.2f})" if (not np.isnan(skew) and skew < -0.3) else ""
        add(
            "info",
            "Fat-tailed returns",
            f"Excess kurtosis is {kurt:.1f}{skew_txt}; extreme moves are more likely "
            "than a normal distribution predicts, so historical VaR is more reliable here.",
        )

    if n_assets == 1:
        add(
            "warning",
            "Single-holding portfolio",
            "All capital is in one asset; concentration and idiosyncratic risk are maximal.",
        )

    d.sort(key=lambda x: SEVERITY_RANK.get(x["severity"], 9))
    return d


def compute_health(metrics: dict) -> dict:
    """Full health report: overall score, band, sub-scores and diagnostics."""
    subscores = compute_subscores(metrics)
    score = _weighted_score(subscores)
    rounded = int(round(score)) if not np.isnan(score) else float("nan")
    return {
        "score": rounded,
        "band": band_for_score(score),
        "subscores": subscores,
        "diagnostics": build_diagnostics(metrics),
    }

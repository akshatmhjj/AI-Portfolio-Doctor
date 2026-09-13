"""AI Portfolio Doctor - a portfolio risk & analytics terminal.

Run with:  streamlit run app.py

The file is intentionally UI-only: every number shown here is computed by the
pure functions in the ``src`` package, which are unit-tested independently.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import data, diagnostics, portfolio, risk, stress, technicals

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Portfolio Doctor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PORTFOLIO_COLOR = "#4C8BF5"
BENCHMARK_COLOR = "#8A93A2"
POS = "#22c55e"
NEG = "#ef4444"
ACCENT = "#E0A458"
GRID = "#232A36"

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}
.app-header {display:flex; align-items:baseline; gap:14px; border-bottom:1px solid #232A36;
             padding-bottom:14px; margin-bottom:6px;}
.app-title {font-size:1.7rem; font-weight:800; letter-spacing:-0.02em; color:#E6E9EF;}
.app-sub {font-size:0.9rem; color:#8A93A2; font-weight:500; text-transform:uppercase; letter-spacing:0.10em;}
.kpi-card {background:#161B26; border:1px solid #232A36; border-radius:10px;
           padding:15px 18px; height:100%;}
.kpi-label {font-size:0.70rem; letter-spacing:0.08em; text-transform:uppercase;
            color:#8A93A2; margin-bottom:6px; font-weight:700;}
.kpi-value {font-size:1.5rem; font-weight:700; color:#E6E9EF; line-height:1.15;
            font-variant-numeric:tabular-nums;}
.kpi-delta {font-size:0.8rem; margin-top:5px; font-weight:600; font-variant-numeric:tabular-nums;}
.pos {color:#22c55e;} .neg {color:#ef4444;} .muted {color:#8A93A2;}
.score-band {display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700;
             font-size:0.85rem; letter-spacing:0.03em;}
h2, h3 {letter-spacing:-0.01em;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_pct(x, dp=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{dp}f}%"


def fmt_ccy(x, dp=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.{dp}f}"


def fmt_num(x, dp=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{dp}f}"


def kpi_card(label, value, delta=None, delta_cls="muted"):
    delta_html = f'<div class="kpi-delta {delta_cls}">{delta}</div>' if delta is not None else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>'
    )


def render_kpis(cards):
    """cards: list of (label, value, delta, delta_cls)."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        label, value = card[0], card[1]
        delta = card[2] if len(card) > 2 else None
        delta_cls = card[3] if len(card) > 3 else "muted"
        col.markdown(kpi_card(label, value, delta, delta_cls), unsafe_allow_html=True)


def sign_cls(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "muted"
    return "pos" if x >= 0 else "neg"


def base_layout(fig, height=420, title=None):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#E6E9EF"),
        title=title,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><span class="app-title">AI Portfolio Doctor</span>'
    '<span class="app-sub">Portfolio Risk &amp; Analytics Terminal</span></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Educational analytics on real historical market data (Yahoo Finance). "
    "Indicators and diagnostics are analytical, **not investment advice.**"
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
DEFAULT_HOLDINGS = pd.DataFrame(
    {
        "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ"],
        "Weight %": [25.0, 20.0, 15.0, 15.0, 15.0, 10.0],
    }
)

PERIODS = {
    "6 Months": 0.5,
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "Custom": None,
}

with st.sidebar:
    st.subheader("Portfolio")
    st.caption("Enter tickers and weights. Weights are normalised to 100%.")
    holdings = st.data_editor(
        DEFAULT_HOLDINGS,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="e.g. AAPL", width="medium"),
            "Weight %": st.column_config.NumberColumn(
                "Weight %", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"
            ),
        },
        key="holdings_editor",
    )

    st.subheader("Analysis window")
    period_label = st.selectbox("Period", list(PERIODS.keys()), index=3)
    today = dt.date.today()
    if period_label == "Custom":
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start", today - dt.timedelta(days=365 * 3))
        end_date = c2.date_input("End", today)
    else:
        years = PERIODS[period_label]
        start_date = today - dt.timedelta(days=int(round(365 * years)))
        end_date = today

    st.subheader("Parameters")
    rf_pct = st.number_input(
        "Risk-free rate (annual, %)", min_value=0.0, max_value=20.0, value=4.0, step=0.25
    )
    conf_pct = st.select_slider(
        "VaR confidence level", options=[90, 95, 99], value=95
    )
    portfolio_value = st.number_input(
        "Portfolio value ($)", min_value=1000, value=100_000, step=1000
    )
    benchmark = st.text_input("Benchmark", value="SPY").strip().upper() or "SPY"

    st.divider()
    st.caption(
        "Data via yfinance. Metrics assume a constant-weight (daily-rebalanced) "
        "portfolio and a 252-day year."
    )

risk_free_rate = rf_pct / 100.0
confidence = conf_pct / 100.0


# ---------------------------------------------------------------------------
# Parse holdings & load data
# ---------------------------------------------------------------------------
def parse_holdings(df: pd.DataFrame) -> dict:
    weights = {}
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "") or "").strip().upper()
        try:
            wt = float(row.get("Weight %"))
        except (TypeError, ValueError):
            continue
        if not ticker or np.isnan(wt) or wt <= 0:
            continue
        weights[ticker] = weights.get(ticker, 0.0) + wt  # sum duplicates
    return weights


raw_weights = parse_holdings(holdings)

if not raw_weights:
    st.warning("Add at least one ticker with a positive weight in the sidebar to begin.")
    st.stop()

raw_sum = sum(raw_weights.values())
requested_tickers = list(raw_weights.keys())

with st.spinner("Downloading and cleaning market data…"):
    try:
        md = data.load_market_data(requested_tickers, start_date, end_date, benchmark=benchmark)
    except Exception as exc:  # network / library errors
        st.error(f"Failed to download market data: {exc}")
        st.stop()

# Validation & messaging -----------------------------------------------------
if md.invalid_tickers:
    st.warning(
        "No usable data for: **"
        + ", ".join(md.invalid_tickers)
        + "**. These tickers were dropped."
    )

if not md.valid_tickers:
    st.error("None of the entered tickers returned usable data. Check the symbols and try again.")
    st.stop()

if md.returns.shape[0] < 30:
    st.error(
        f"Only {md.returns.shape[0]} trading days of overlapping data were found — "
        "not enough for reliable statistics. Widen the analysis window."
    )
    st.stop()

if abs(raw_sum - 100.0) > 0.1:
    st.info(f"Entered weights sum to {raw_sum:.1f}%. They have been normalised to 100%.")

# Weights aligned to the valid, data-backed tickers --------------------------
valid = md.valid_tickers
dropped_weight = sum(raw_weights[t] for t in requested_tickers if t not in valid)
if dropped_weight > 0 and len(valid) < len(requested_tickers):
    st.info("Weights were renormalised across the remaining valid holdings.")

weights_series = portfolio.normalize_weights({t: raw_weights[t] for t in valid})
asset_returns = md.returns[valid]
weights_series = weights_series.reindex(asset_returns.columns)
w = weights_series.to_numpy()

benchmark_returns = md.benchmark_returns
has_benchmark = not benchmark_returns.empty

# Core return series ---------------------------------------------------------
port_ret = portfolio.portfolio_returns(asset_returns, weights_series)
perf = portfolio.performance_summary(port_ret, risk_free_rate)

if has_benchmark:
    bench_perf = portfolio.performance_summary(benchmark_returns, risk_free_rate)
    port_beta = portfolio.beta(port_ret, benchmark_returns)
else:
    bench_perf = None
    port_beta = float("nan")

data_span = f"{asset_returns.index.min():%d %b %Y} → {asset_returns.index.max():%d %b %Y}"
st.caption(
    f"**{len(valid)}** holdings · **{asset_returns.shape[0]}** trading days · {data_span} · "
    f"benchmark **{benchmark}**"
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_risk, tab_tech, tab_stress, tab_health = st.tabs(
    ["Overview", "Risk & Diversification", "Technical Analysis", "Stress Testing", "Portfolio Health"]
)

# ============================ OVERVIEW =====================================
with tab_overview:
    st.subheader("Performance overview")

    ann_ret_delta = None
    ann_ret_cls = "muted"
    if bench_perf is not None:
        diff = perf["annualized_return"] - bench_perf["annualized_return"]
        ann_ret_delta = f"{fmt_pct(diff)} vs {benchmark}"
        ann_ret_cls = sign_cls(diff)

    render_kpis(
        [
            ("Total Return", fmt_pct(perf["total_return"]), None, sign_cls(perf["total_return"])),
            ("Annualized Return", fmt_pct(perf["annualized_return"]), ann_ret_delta, ann_ret_cls),
            ("Annualized Volatility", fmt_pct(perf["annualized_volatility"])),
            ("Sharpe Ratio", fmt_num(perf["sharpe_ratio"]), f"rf = {rf_pct:.2f}%", "muted"),
            ("Sortino Ratio", fmt_num(perf["sortino_ratio"])),
            ("Max Drawdown", fmt_pct(perf["max_drawdown"]), None, "neg"),
        ]
    )

    # Cumulative return + drawdown ------------------------------------------
    port_growth = (1 + port_ret).cumprod()
    port_growth = port_growth / port_growth.iloc[0] * 100.0
    dd = portfolio.drawdown_series(port_ret) * 100.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
        subplot_titles=("Growth of $100 (rebased)", "Drawdown (%)"),
    )
    fig.add_trace(
        go.Scatter(x=port_growth.index, y=port_growth, name="Portfolio",
                   line=dict(color=PORTFOLIO_COLOR, width=2)),
        row=1, col=1,
    )
    if has_benchmark:
        bench_growth = (1 + benchmark_returns).cumprod()
        bench_growth = bench_growth / bench_growth.iloc[0] * 100.0
        fig.add_trace(
            go.Scatter(x=bench_growth.index, y=bench_growth, name=benchmark,
                       line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot")),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy",
                   line=dict(color=NEG, width=1), showlegend=False),
        row=2, col=1,
    )
    base_layout(fig, height=520)
    st.plotly_chart(fig, use_container_width=True)

    if bench_perf is not None:
        st.markdown("**Portfolio vs benchmark**")
        cmp = pd.DataFrame(
            {
                "Portfolio": [
                    fmt_pct(perf["annualized_return"]),
                    fmt_pct(perf["annualized_volatility"]),
                    fmt_num(perf["sharpe_ratio"]),
                    fmt_pct(perf["max_drawdown"]),
                    fmt_num(port_beta),
                ],
                benchmark: [
                    fmt_pct(bench_perf["annualized_return"]),
                    fmt_pct(bench_perf["annualized_volatility"]),
                    fmt_num(bench_perf["sharpe_ratio"]),
                    fmt_pct(bench_perf["max_drawdown"]),
                    "1.00",
                ],
            },
            index=["Annualized return", "Annualized volatility", "Sharpe", "Max drawdown", "Beta"],
        )
        st.dataframe(cmp, use_container_width=True)

    with st.expander("Methodology — performance metrics"):
        st.markdown(
            """
- **Returns** are simple daily returns of adjusted close prices. The portfolio
  return each day is the weighted sum of holding returns (constant-weight /
  daily-rebalanced assumption).
- **Annualized return** is the geometric CAGR: `(∏(1+r))^(252/N) − 1`.
- **Annualized volatility** = daily standard deviation × √252.
- **Sharpe** = √252 · mean(excess daily return) / σ, with the annual risk-free
  rate converted to a daily rate. **Sortino** replaces σ with the downside
  deviation (only below-target returns penalised).
- **Max drawdown** is the largest peak-to-trough decline of the cumulative
  wealth curve.
            """
        )

# ==================== RISK & DIVERSIFICATION ================================
with tab_risk:
    st.subheader("Risk & diversification")

    cov_ann = risk.covariance_matrix(asset_returns, annualize=True)
    port_vol_ann = risk.portfolio_volatility(w, cov_ann)
    hist_var = risk.historical_var(port_ret, confidence)
    cond_var = risk.conditional_var(port_ret, confidence)
    param_var = risk.parametric_var(port_ret, confidence)
    hhi = risk.herfindahl_index(w)
    eff_n = risk.effective_holdings(w)
    avg_corr = risk.average_correlation(asset_returns)
    dist_stats = risk.return_distribution_stats(port_ret)

    render_kpis(
        [
            ("Annualized Volatility", fmt_pct(port_vol_ann)),
            (f"Historical VaR ({conf_pct}%)", fmt_pct(hist_var), f"{fmt_ccy(risk.var_dollar(hist_var, portfolio_value))} / day", "neg"),
            (f"CVaR ({conf_pct}%)", fmt_pct(cond_var), f"{fmt_ccy(risk.var_dollar(cond_var, portfolio_value))} / day", "neg"),
            ("Portfolio Beta", fmt_num(port_beta)),
            ("Effective Holdings", fmt_num(eff_n, 1), f"HHI {fmt_num(hhi, 3)}", "muted"),
        ]
    )

    left, right = st.columns([1.05, 1])

    # ---- Risk contributions ----
    with left:
        st.markdown("**Contribution to portfolio risk**")
        rc = risk.risk_contributions(w, cov_ann)
        rc_sorted = rc.sort_values("pct_contribution", ascending=False)

        fig_rc = go.Figure()
        fig_rc.add_trace(
            go.Bar(
                x=rc_sorted["pct_contribution"] * 100.0,
                y=rc_sorted.index,
                orientation="h",
                marker_color=[ACCENT if i == 0 else PORTFOLIO_COLOR for i in range(len(rc_sorted))],
                text=[fmt_pct(v, 1) for v in rc_sorted["pct_contribution"]],
                textposition="auto",
            )
        )
        fig_rc.update_layout(yaxis=dict(autorange="reversed"))
        base_layout(fig_rc, height=max(240, 46 * len(rc_sorted)))
        fig_rc.update_xaxes(title_text="% of total portfolio risk")
        st.plotly_chart(fig_rc, use_container_width=True)

        top = rc_sorted.index[0]
        top_pct = rc_sorted["pct_contribution"].iloc[0]
        top_w = rc_sorted["weight"].iloc[0]
        st.markdown(
            f"**{top}** is the largest risk contributor: **{fmt_pct(top_pct,1)}** of total "
            f"risk from a **{fmt_pct(top_w,1)}** weight."
            + ("  Its risk share exceeds its capital share (a concentrated risk source)."
               if top_pct > top_w + 0.02 else "")
        )

        rc_table = pd.DataFrame(
            {
                "Weight": rc_sorted["weight"].map(lambda x: fmt_pct(x, 1)),
                "Comp. vol (ann.)": rc_sorted["cctr"].map(lambda x: fmt_pct(x, 2)),
                "% of risk": rc_sorted["pct_contribution"].map(lambda x: fmt_pct(x, 1)),
            }
        )
        st.dataframe(rc_table, use_container_width=True)

    # ---- Correlation matrix ----
    with right:
        st.markdown("**Correlation matrix (daily returns)**")
        if len(valid) >= 2:
            corr = risk.correlation_matrix(asset_returns)
            show_text = len(valid) <= 12
            fig_c = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1, zmax=1, zmid=0,
                    colorscale="RdBu_r",
                    text=np.round(corr.values, 2) if show_text else None,
                    texttemplate="%{text}" if show_text else None,
                    colorbar=dict(title="ρ"),
                )
            )
            base_layout(fig_c, height=max(300, 40 * len(valid) + 120))
            st.plotly_chart(fig_c, use_container_width=True)
            if not np.isnan(avg_corr):
                st.caption(f"Average pairwise correlation: **{fmt_num(avg_corr)}**")
        else:
            st.info("Add a second holding to see the correlation structure.")

    with st.expander("Methodology — risk & diversification"):
        st.markdown(
            f"""
- **Portfolio volatility** = √(wᵀ Σ w) using the annualised covariance matrix Σ.
- **Historical VaR ({conf_pct}%)** is the loss at the {100-conf_pct:.0f}th percentile
  of daily returns (empirical, no distribution assumed). **CVaR / Expected
  Shortfall** is the average loss *beyond* that VaR level. A parametric
  (Gaussian) VaR of **{fmt_pct(param_var)}** is computed for comparison — a
  gap versus the historical figure signals fat tails (excess kurtosis
  {fmt_num(dist_stats['excess_kurtosis'])}, skew {fmt_num(dist_stats['skew'])}).
- **Beta** = Cov(portfolio, {benchmark}) / Var({benchmark}).
- **HHI** (Herfindahl index) = Σwᵢ²; **effective holdings** = 1/HHI.
- **Risk decomposition:** marginal contribution MCTRᵢ = (Σw)ᵢ/σₚ, component
  contribution CCTRᵢ = wᵢ·MCTRᵢ. The CCTRs sum exactly to σₚ, so "% of risk"
  is an exact, additive attribution — it shows where risk *actually* comes
  from, which is not the same as where the *capital* is.
            """
        )

# ======================= TECHNICAL ANALYSIS ================================
with tab_tech:
    st.subheader("Technical analysis")
    st.caption("Analytical indicators describing price behaviour — **not buy/sell signals.**")

    sel = st.selectbox("Asset", valid, index=0)
    price = md.prices[sel].dropna()

    if len(price) < 20:
        st.info("Not enough price history for this asset to compute indicators.")
    else:
        mas = technicals.moving_averages(price)
        rsi_series = technicals.rsi(price, 14)
        macd_df = technicals.macd(price)
        bb = technicals.bollinger_bands(price)

        last_price = price.iloc[-1]
        last_rsi = rsi_series.iloc[-1]
        last_hist = macd_df["histogram"].iloc[-1]
        sma50 = mas["SMA50"].iloc[-1]

        rsi_state = "overbought" if last_rsi >= 70 else "oversold" if last_rsi <= 30 else "neutral"
        render_kpis(
            [
                ("Last Price", fmt_ccy(last_price, 2)),
                ("RSI (14)", fmt_num(last_rsi, 1), rsi_state, "neg" if rsi_state != "neutral" else "muted"),
                ("MACD Histogram", fmt_num(last_hist, 3), "bullish" if last_hist > 0 else "bearish", sign_cls(last_hist)),
                ("Price vs SMA50", fmt_pct(last_price / sma50 - 1) if not np.isnan(sma50) else "—",
                 "above" if (not np.isnan(sma50) and last_price > sma50) else "below",
                 sign_cls(last_price - sma50) if not np.isnan(sma50) else "muted"),
            ]
        )

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.04,
            subplot_titles=(f"{sel} price, moving averages & Bollinger bands", "RSI (14)", "MACD (12,26,9)"),
        )
        # Price + BB + SMAs
        fig.add_trace(go.Scatter(x=bb.index, y=bb["upper"], name="BB upper",
                                 line=dict(color="rgba(138,147,162,0.35)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb.index, y=bb["lower"], name="BB lower", fill="tonexty",
                                 fillcolor="rgba(76,139,245,0.07)",
                                 line=dict(color="rgba(138,147,162,0.35)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=price.index, y=price, name="Close",
                                 line=dict(color="#E6E9EF", width=1.6)), row=1, col=1)
        for col_name, color in [("SMA20", PORTFOLIO_COLOR), ("SMA50", ACCENT), ("SMA200", "#a78bfa")]:
            fig.add_trace(go.Scatter(x=mas.index, y=mas[col_name], name=col_name,
                                     line=dict(color=color, width=1.2)), row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI",
                                 line=dict(color=PORTFOLIO_COLOR, width=1.3), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line=dict(color=NEG, width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=POS, width=1, dash="dot"), row=2, col=1)
        # MACD
        hist_colors = [POS if v >= 0 else NEG for v in macd_df["histogram"]]
        fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["histogram"], name="Histogram",
                             marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["macd"], name="MACD",
                                 line=dict(color=PORTFOLIO_COLOR, width=1.3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["signal"], name="Signal",
                                 line=dict(color=ACCENT, width=1.3)), row=3, col=1)
        base_layout(fig, height=720)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology — technical indicators"):
        st.markdown(
            """
- **SMA 20/50/200:** simple moving averages of the close — trend descriptors.
- **RSI (14):** Wilder's Relative Strength Index; 100 − 100/(1+RS) where RS is
  the ratio of average gains to average losses. Bounded 0–100; ~70/30 mark
  conventionally over/under-bought regimes.
- **MACD (12,26,9):** 12- vs 26-period EMA difference, with a 9-period signal
  line; the histogram is their gap (momentum).
- **Bollinger Bands (20, 2σ):** SMA20 ± 2 rolling standard deviations —
  a relative volatility envelope.
            """
        )

# =========================== STRESS TESTING ================================
with tab_stress:
    st.subheader("Stress testing")

    betas = portfolio.asset_betas(asset_returns, benchmark_returns) if has_benchmark else pd.Series(
        {t: 1.0 for t in valid}
    )

    with st.spinner("Classifying sectors…"):
        try:
            sectors = data.fetch_sectors(tuple(valid))
        except Exception:
            sectors = {t: "Unknown" for t in valid}

    st.markdown("**Custom scenario**")
    cc1, cc2 = st.columns([1, 1])
    custom_market = cc1.slider("Market shock (beta-adjusted, %)", -60, 20, -15, step=5) / 100.0
    tech_shock_val = cc2.slider("Technology-sector shock (%)", -60, 0, -25, step=5) / 100.0

    scenarios = {
        "Market Crash (−20%)": stress.market_crash(betas, -0.20),
        "Severe Stress (−30%)": stress.severe_stress(betas, -0.30),
        "Technology Selloff": stress.technology_selloff(sectors, betas, tech_shock_val, -0.05),
        f"Custom Market ({custom_market:+.0%})": stress.beta_adjusted_shocks(betas, custom_market),
    }

    rows = []
    results = {}
    for name, shocks in scenarios.items():
        res = stress.apply_shocks(weights_series, shocks, portfolio_value, scenario=name)
        results[name] = res
        rows.append(
            {
                "Scenario": name,
                "Portfolio return": fmt_pct(res.portfolio_return),
                "P&L": fmt_ccy(res.pnl),
                "Ending value": fmt_ccy(res.ending_value),
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Scenario"), use_container_width=True)

    # Severe-stress loss feeds the health score (computed directly so it does
    # not depend on the display label used as a dict key above).
    severe_loss = stress.apply_shocks(
        weights_series, stress.severe_stress(betas, -0.30), portfolio_value
    ).portfolio_return

    st.markdown("**Contribution to loss by holding**")
    focus = st.selectbox("Scenario to break down", list(scenarios.keys()), index=0)
    res = results[focus]
    detail = res.detail.copy()
    fig_s = go.Figure(
        go.Bar(
            x=detail["contribution_to_pnl"],
            y=detail.index,
            orientation="h",
            marker_color=[NEG if v < 0 else POS for v in detail["contribution_to_pnl"]],
            text=[fmt_ccy(v) for v in detail["contribution_to_pnl"]],
            textposition="auto",
        )
    )
    base_layout(fig_s, height=max(240, 46 * len(detail)))
    fig_s.update_xaxes(title_text="Contribution to P&L ($)")
    st.plotly_chart(fig_s, use_container_width=True)

    detail_table = pd.DataFrame(
        {
            "Sector": [sectors.get(t, "Unknown") for t in detail.index],
            "Weight": detail["weight"].map(lambda x: fmt_pct(x, 1)),
            "Asset shock": detail["shock"].map(lambda x: fmt_pct(x, 1)),
            "Contribution to return": detail["contribution_to_return"].map(lambda x: fmt_pct(x, 2)),
            "Contribution to P&L": detail["contribution_to_pnl"].map(fmt_ccy),
        }
    )
    st.dataframe(detail_table, use_container_width=True)

    with st.expander("Methodology — stress testing"):
        st.markdown(
            """
- **Beta-adjusted scenarios** translate a market move into per-asset moves via
  each holding's beta to the benchmark: `shockᵢ = βᵢ · market move`. The
  portfolio loss is therefore driven by its aggregate beta, and higher-beta
  names take larger hits.
- **Technology selloff** applies a direct shock to holdings classified in the
  Technology sector (sector via yfinance, best-effort) and a smaller
  beta-scaled market spillover to the rest.
- **Contribution to loss** for holding *i* is `wᵢ · shockᵢ · portfolio value`;
  these sum to the total P&L. This is a deterministic scenario analysis, not a
  probabilistic forecast.
            """
        )

# ========================== PORTFOLIO HEALTH ===============================
with tab_health:
    st.subheader("Portfolio health")

    max_w_ticker = weights_series.idxmax()
    metrics = {
        "annualized_volatility": perf["annualized_volatility"],
        "max_drawdown": perf["max_drawdown"],
        "var": hist_var,
        "cvar": cond_var,
        "stress_loss": severe_loss,
        "hhi": hhi,
        "effective_holdings": eff_n,
        "max_weight": float(weights_series.max()),
        "max_weight_ticker": max_w_ticker,
        "average_correlation": avg_corr,
        "beta": port_beta,
        "skew": dist_stats["skew"],
        "excess_kurtosis": dist_stats["excess_kurtosis"],
        "n_assets": len(valid),
    }
    health = diagnostics.compute_health(metrics)
    score = health["score"]
    band = health["band"]

    band_color = {
        "Low Risk": POS,
        "Moderate Risk": ACCENT,
        "Elevated Risk": "#f59e0b",
        "High Risk": NEG,
        "Unknown": BENCHMARK_COLOR,
    }[band]

    gcol, dcol = st.columns([1, 1.5])
    with gcol:
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"size": 46, "color": "#E6E9EF"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8A93A2"},
                    "bar": {"color": band_color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239,68,68,0.18)"},
                        {"range": [40, 60], "color": "rgba(245,158,11,0.18)"},
                        {"range": [60, 80], "color": "rgba(224,164,88,0.16)"},
                        {"range": [80, 100], "color": "rgba(34,197,94,0.16)"},
                    ],
                },
            )
        )
        base_layout(fig_g, height=300)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:-12px;">'
            f'<span class="score-band" style="background:{band_color}22;color:{band_color};">'
            f"{band} · {score}/100</span></div>",
            unsafe_allow_html=True,
        )

    with dcol:
        st.markdown("**Risk sub-scores** (higher = healthier)")
        labels = {
            "volatility": "Volatility", "drawdown": "Drawdown", "var": "Value at Risk",
            "stress": "Stress resilience", "concentration": "Concentration", "correlation": "Correlation",
        }
        sub = health["subscores"]
        order = ["volatility", "drawdown", "var", "stress", "concentration", "correlation"]
        vals = [sub[k] for k in order]
        colors = []
        for v in vals:
            if np.isnan(v):
                colors.append(BENCHMARK_COLOR)
            elif v >= 80:
                colors.append(POS)
            elif v >= 60:
                colors.append(ACCENT)
            elif v >= 40:
                colors.append("#f59e0b")
            else:
                colors.append(NEG)
        fig_sub = go.Figure(
            go.Bar(
                x=[0 if np.isnan(v) else v for v in vals],
                y=[labels[k] for k in order],
                orientation="h",
                marker_color=colors,
                text=[("n/a" if np.isnan(v) else f"{v:.0f}") for v in vals],
                textposition="auto",
            )
        )
        fig_sub.update_layout(yaxis=dict(autorange="reversed"))
        fig_sub.update_xaxes(range=[0, 100])
        base_layout(fig_sub, height=300)
        st.plotly_chart(fig_sub, use_container_width=True)

    st.markdown("**Diagnostics**")
    render_fn = {"danger": st.error, "warning": st.warning, "info": st.info, "good": st.success}
    if not health["diagnostics"]:
        st.info("No notable risk flags for this portfolio over the selected window.")
    for item in health["diagnostics"]:
        render_fn.get(item["severity"], st.info)(f"**{item['title']}** — {item['detail']}")

    with st.expander("Methodology — health score"):
        st.markdown(
            """
The score is a **transparent weighted blend** of six risk sub-scores — there is
no machine-learning black box. Each metric is mapped onto 0–100 (higher =
healthier) via a clamped linear ramp between a "good" and a "bad" level, then
combined:

| Sub-score | Metric | good → 100 | bad → 0 | Weight |
|---|---|---|---|---|
| Volatility | Annualised σ | 10% | 40% | 20% |
| Drawdown | Max drawdown | 10% | 50% | 20% |
| Value at Risk | 1-day historical VaR | 1% | 6% | 15% |
| Stress resilience | Severe-stress loss | 15% | 45% | 15% |
| Concentration | HHI | 0.15 | 0.50 | 15% |
| Correlation | Avg pairwise ρ | 0.20 | 0.80 | 15% |

If a sub-score is undefined (e.g. correlation for a single holding) its weight
is redistributed. Every diagnostic message above is triggered by a threshold on
one of these real, computed metrics.
            """
        )

st.divider()
st.caption(
    "AI Portfolio Doctor · built with Streamlit, pandas, NumPy, SciPy & Plotly · "
    "data from Yahoo Finance via yfinance. For educational use only."
)

# AI Portfolio Doctor

A portfolio **risk & analytics terminal** built with Streamlit. Enter a set of
tickers and weights and it pulls real historical market data (Yahoo Finance) to
compute performance, risk, diversification, technical and stress-test analytics
— then distils everything into an interpretable **Portfolio Health Score**.

The "AI" here is an **interpretable, rules-based diagnostic engine**, not a
black-box model: every score and message is traceable to a real, calculated
metric. No paid APIs, no API keys.

> Educational tool. Analytics and indicators are **not investment advice.**

---

## Features

**Overview** — total & annualized return, annualized volatility, Sharpe,
Sortino, max drawdown, beta, and a benchmark (SPY) comparison with a rebased
growth curve and an underwater drawdown chart.

**Risk & Diversification** — historical VaR, CVaR / Expected Shortfall
(with a parametric-Gaussian VaR for a fat-tail comparison), beta, the
correlation matrix, portfolio volatility from the covariance matrix, the
Herfindahl (HHI) concentration index / effective holdings, and a full
**marginal / component contribution-to-risk** decomposition that identifies
which holdings actually drive portfolio risk.

**Technical Analysis** — SMA 20/50/200, RSI(14), MACD(12,26,9) and Bollinger
Bands for a selected asset. Analytical indicators, **not** buy/sell signals.

**Stress Testing** — beta-adjusted Market Crash (−20%), Severe Stress (−30%),
a Technology Selloff, and a custom market/sector shock. Reports portfolio loss,
ending value, and each holding's contribution to the loss.

**Portfolio Health** — a transparent 0–100 score blending volatility, drawdown,
VaR, stress loss, concentration and correlation, with plain-language diagnostics
("High concentration risk", "Strong diversification", "Elevated downside risk",
…) — each backed by the metric that triggered it.

---

## Project structure

```
AI-portfolio-doctor/
├── app.py                  # Streamlit UI (presentation only)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml         # dark "analytics terminal" theme
├── src/                    # pure, UI-free financial engine
│   ├── __init__.py
│   ├── data.py             # yfinance download + cleaning (cached)
│   ├── portfolio.py        # returns, Sharpe/Sortino, drawdown, beta
│   ├── risk.py             # VaR/CVaR, covariance, HHI, risk contributions
│   ├── technicals.py       # SMA, RSI, MACD, Bollinger Bands
│   ├── stress.py           # scenario / stress-test engine
│   └── diagnostics.py      # interpretable health-score engine
└── tests/                  # pytest suite for the financial calculations
    ├── conftest.py
    ├── test_portfolio.py
    ├── test_risk.py
    ├── test_technicals.py
    ├── test_stress.py
    └── test_diagnostics.py
```

The `src` package contains **no Streamlit code**, so the financial logic is
independently testable and reusable.

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Run the tests

```bash
pytest -q
```

(Tests use synthetic data and never touch the network.)

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Select your repo/branch, set **Main file path** to `app.py`.
5. (Optional) **Advanced settings** → choose a Python version (3.11+ recommended).
6. **Deploy**. Dependencies install from `requirements.txt` automatically.

No secrets or API keys are required.

---

## Key formulas

- **Portfolio return (daily):** `r_p,t = Σ_i w_i · r_i,t` (constant weights).
- **Annualized return (CAGR):** `(∏(1+r))^(252/N) − 1`.
- **Annualized volatility:** `std(r) · √252`.
- **Sharpe:** `√252 · mean(r − r_f/252) / std(r)`.
- **Sortino:** as Sharpe but the denominator is the **downside deviation**
  `√(mean(min(r − target, 0)²))`.
- **Max drawdown:** `min_t ( W_t / max_{s≤t} W_s − 1 )` on wealth `W = ∏(1+r)`.
- **Beta:** `Cov(r_p, r_m) / Var(r_m)` vs SPY.
- **Historical VaR(c):** `−` the `(1−c)` empirical quantile of daily returns.
- **CVaR / ES(c):** mean loss in the tail **beyond** VaR.
- **Portfolio volatility:** `√(wᵀ Σ w)`.
- **HHI:** `Σ w_i²`; **effective holdings** = `1 / HHI`.
- **Risk decomposition:** marginal `MCTR_i = (Σw)_i / σ_p`, component
  `CCTR_i = w_i · MCTR_i`, with `Σ_i CCTR_i = σ_p` (exact, additive).

See the in-app **Methodology** expanders for the full details, and
`tests/` for machine-checked correctness of each formula.

---

## Tech stack

Python · Streamlit · pandas · NumPy · SciPy · Plotly · yfinance · pytest

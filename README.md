# Gold (XAU/USD) Multi-Factor AI Trading Research System

A self-contained Python system that fetches gold and macro data, computes
technical / macro / sentiment signals, trains walk-forward XGBoost models,
combines everything into a single recommendation, and backtests the result with
**no lookahead bias** — so you can honestly evaluate whether the strategy has any
statistical edge.

> **This is a research and education tool, not a trading product.** Read the
> *Known limitations* and *Disclaimer* sections before drawing any conclusions.

---

## 1. System overview

```
                    ┌─────────────────────────────────────────────┐
                    │                  main.py                    │
                    │              (orchestrator)                 │
                    └─────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐            ┌───────────────────┐          ┌─────────────────┐
│   data/       │            │    signals/       │          │      ml/        │
│  fetcher.py   │ ─ panel ─▶ │  technical.py     │          │  features.py    │
│ preprocessor  │            │  macro.py         │ ─scores▶  │  trainer.py     │
└───────────────┘            │  sentiment.py     │          │  predictor.py   │
   yfinance + FRED           └───────────────────┘          │  evaluator.py   │
                                                            └─────────────────┘
                                       │                            │
                                       ▼                            ▼
                             ┌───────────────────────────────────────────┐
                             │                 engine/                   │
                             │  combiner.py  →  backtester.py  ←  risk.py │
                             └───────────────────────────────────────────┘
                                       │
                                       ▼
                             ┌───────────────────────────┐
                             │      reporting/           │
                             │     dashboard.py          │
                             │  charts + trades.csv +    │
                             │      summary.txt          │
                             └───────────────────────────┘
```

**Data flow:** raw prices/macro → one aligned, leakage-safe panel → three signal
families + an ML model → a single composite score per day → a no-lookahead
backtest → charts and a plain-English summary, plus today's live recommendation.

---

## 2. Installation

Requires **Python 3.10+**.

```bash
# (recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

The default stack is CPU-only and needs no GPU. FinBERT (transformer-based news
sentiment) is **optional** and commented out in `requirements.txt` — the system
falls back to a lightweight TextBlob scorer automatically.

---

## 3. Free API keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

**FRED (required for macro signals) — free, instant.**
1. Go to <https://fredapi.stlouisfed.org/> (or search "FRED API key").
2. Create an account and request an API key.
3. Paste it into `.env` as `FRED_API_KEY=...`.

Without it, the macro signals that depend on FRED (real yields, CPI, M2, the
recession spread) simply turn off and the rest of the system still runs.

**NewsAPI (optional — live news sentiment only) — free developer tier.**
1. Register at <https://newsapi.org/register>.
2. Copy your key into `.env` as `NEWSAPI_KEY=...`.

If it is missing or the daily quota is exceeded, the system falls back to public
RSS feeds, and if those fail too, sentiment just contributes nothing live.

---

## 4. Running

```bash
python main.py
```

On the first run it downloads ~7 years of data (cached afterwards), trains the
models with walk-forward validation, backtests, and writes everything to
`reports/`. Expect roughly **1–3 minutes** on a typical laptop; the hyperparameter
search is the slow part, and repeat runs are faster thanks to data caching.

Outputs:
- `reports/charts/*.png` — the seven analysis charts (plus ML diagnostics)
- `reports/trades.csv` — every simulated trade
- `reports/summary.txt` — the performance summary
- the same summary and today's recommendation printed to your console

---

## 5. Interpreting the output metrics (plain English)

**Backtest performance**

- **Total return** — how much the €10,000 grew or shrank over the test period.
- **CAGR** — that return expressed as a smooth annual growth rate.
- **Sharpe ratio** — return per unit of *total* risk, above the 4% risk-free
  rate. Higher is better. Below 1 is weak, around 1 is okay, above 2 is strong
  (and rare/suspicious for a retail daily strategy).
- **Sortino ratio** — like Sharpe but only penalises *downside* volatility, so
  it rewards strategies whose "risk" is mostly upside.
- **Max drawdown** — the worst peak-to-trough fall. This is the number that
  tells you how much pain you'd have sat through. **Always read this before
  getting excited about returns.**
- **Calmar ratio** — CAGR divided by max drawdown. Return earned per unit of
  worst-case pain.
- **Win rate** — percentage of trades that made money. A high win rate with a
  bad profit factor means your losers are bigger than your winners.
- **Profit factor** — gross profit ÷ gross loss. Above 1 means profitable;
  1.5+ is healthy.
- **Average win / loss** — the typical size of a winning vs losing trade.
- **Profit/loss is always compared to buy-and-hold** — if simply holding gold
  beat the strategy, the strategy added no value (and added costs and risk).

**ML model quality**

- **Accuracy** — how often the up/down call was right. Compare it to the
  "always predict up" baseline shown next to it; beating that baseline by
  predicting the majority class is *not* skill.
- **AUC-ROC** — can the model rank up-days above down-days? 0.50 is a coin flip;
  durably above ~0.55 out-of-sample is genuinely notable for daily gold.
- **Information Coefficient (IC)** — rank correlation between predicted and
  actual returns; the industry-standard signal measure. Real-world ICs of
  0.03–0.05 are considered good. **Much higher usually means a leak, not genius.**
- **Calibration curve** — when the model says "70% confident", is it right ~70%
  of the time? Points on the diagonal mean well-calibrated probabilities.

---

## 6. What each signal means and why it matters for gold

**Technical (price-based):** RSI, MACD, Bollinger Bands, EMA trend stack,
Stochastic, OBV, golden/death cross, and volume spikes. These capture momentum,
mean-reversion and trend on gold's own price. They matter because gold trends
and overshoots like any liquid market.

**Macro (the real drivers of gold):**
- **10Y real yield** — the single most important driver. Gold pays no interest,
  so when inflation-adjusted bond yields rise, holding gold costs you more
  (bearish); when they fall, gold looks better (bullish).
- **US dollar (DXY)** — gold is priced in dollars, so a stronger dollar is a
  mechanical headwind.
- **CPI inflation** — gold's reputation as an inflation hedge (with the nuance
  that high inflation can trigger rate hikes that hurt gold — the real-yield
  signal is the counterweight).
- **M2 money supply** — more liquidity / debasement tends to support gold.
- **Gold/silver ratio** — a weak contextual mean-reversion gauge for the metals
  complex.
- **Oil** — an inflation proxy.
- **Risk-off (VIX + falling stocks)** — safe-haven demand spikes in panics.
- **10Y minus Fed funds** — an inverted spread flags recession risk, which
  historically supports gold.

**Sentiment:** a VIX-based safe-haven "fear" proxy for the backtest (the only
sentiment signal available historically from free data), plus live news scoring
(FinBERT or TextBlob) for today's recommendation only. See *Known limitations*.

---

## 7. Tuning parameters

Everything tunable lives in **`config.py`**, grouped into clear sections. Common
knobs:

- **Signal weights** (`SIGNAL_WEIGHTS`) — how much each family contributes.
- **Thresholds** (`BUY_THRESHOLD`, `STRONG_BUY_THRESHOLD`, …) — how decisive a
  composite score must be to act.
- **ML** (`ML_PREDICTION_HORIZON`, `ML_TRAIN_YEARS`, `ML_SEARCH_N_ITER`,
  `ML_MIN_CONFIDENCE`) — forecast horizon, training window, search budget, and
  the confidence below which the model abstains. Raise `ML_SEARCH_N_ITER` to
  search harder (slower); lower it to run faster.
- **Risk** (`KELLY_FRACTION`, `STOP_LOSS_ATR_MULTIPLE`,
  `TAKE_PROFIT_ATR_MULTIPLE`, `MAX_POSITION_PCT`, …) — position sizing and exits.
- **Costs** (`TRANSACTION_COST_PCT`, `SLIPPAGE_PCT`) — make these realistic for
  your broker; optimistic costs are the most common way backtests lie.

Change a value, re-run `python main.py`, and compare `reports/summary.txt`.

---

## 8. Known limitations (please read)

- **Edge is probably small or zero.** Predicting 5-day gold direction from free
  daily data is close to a coin flip after costs. The honest value here is the
  *methodology* (clean walk-forward, no lookahead, honest metrics), not a promise
  of profit. Treat any strong backtest with suspicion and check for leaks first.
- **Historical news sentiment is not available** from free APIs (NewsAPI free is
  ~30 days). The backtest's "sentiment" is therefore a VIX fear proxy, which is
  one-sided (it flags fear-driven bullishness, not bearishness). Live news
  scoring applies only to today's recommendation.
- **Free data is imperfect** — `GC=F` is a futures proxy for spot XAU/USD, volume
  on some feeds is unreliable, and Yahoo symbols occasionally change.
- **Overfitting risk.** Even with walk-forward validation, searching
  hyperparameters and hand-picking features on one asset over one history can
  flatter the model. Out-of-sample ≠ future.
- **Single asset, long-only, daily.** No shorting, no intraday, no portfolio
  effects. Kelly sizing assumes you know your edge (you don't, precisely) — hence
  half-Kelly and hard caps.
- **Regime dependence.** Relationships (e.g. gold vs real yields) shift over
  time; a model trained on one regime can mislead in another.
- **Macro publication lag is modelled conservatively**, not exactly — it can
  only remove edge, never fabricate it, which is the safe direction.

---

## 9. Extending the system

The architecture is modular, so additions are localised:

- **Add a new signal** — write a function in the relevant `signals/*.py` that
  returns a `signal`/`strength` frame, register it in that module's signal
  registry, and it flows into the aggregate score automatically.
- **Add a new feature** — add a column in `ml/features.py` (it will be lagged
  with everything else; never use a forward-looking value).
- **Add a new data source** — add it to `YF_TICKERS` or `FRED_SERIES` in
  `config.py`; the fetcher and preprocessor pick it up generically.
- **Trade a different asset** — change `PRIMARY_ASSET` and the ticker map in
  `config.py`. Most signals are gold-flavoured, so review the macro logic.
- **Swap the model** — `ml/trainer.py` isolates the estimators; replace XGBoost
  with another time-series-aware model behind the same interface.
- **Tune the search** — widen `ML_PARAM_GRID` and raise `ML_SEARCH_N_ITER`.

---

## 10. Project structure

```
gold_system/
├── main.py                 # orchestrator (run this)
├── config.py               # ALL parameters
├── .env.example            # API-key template
├── requirements.txt
├── data/
│   ├── fetcher.py          # yfinance + FRED ingestion, caching, graceful degrade
│   └── preprocessor.py     # alignment, publication-lag, master panel
├── signals/
│   ├── technical.py        # RSI, MACD, Bollinger, EMA, Stoch, OBV, crosses, volume
│   ├── macro.py            # real yields, DXY, CPI, M2, ratios, risk-off, recession
│   └── sentiment.py        # VIX fear proxy (backtest) + live news NLP
├── ml/
│   ├── features.py         # strictly-lagged feature matrix + forward targets
│   ├── trainer.py          # walk-forward XGBoost (classifier + regressor)
│   ├── predictor.py        # today's prediction + SHAP explanation
│   └── evaluator.py        # AUC, IC, calibration, baselines, charts
├── engine/
│   ├── combiner.py         # composite score with dynamic weights
│   ├── backtester.py       # no-lookahead long-only backtest + metrics
│   └── risk.py             # half-Kelly sizing, ATR stops, regime, guards
└── reporting/
    └── dashboard.py        # 7 charts + trades.csv + summary.txt
```

---

## 11. Disclaimer

This software is provided for **educational and research purposes only**. It is
**not financial advice**, not a recommendation to buy or sell any asset, and not
a solicitation. Backtested and simulated results are hypothetical, do not reflect
real trading, and **past performance does not guarantee future results**. Trading
leveraged or commodity instruments carries substantial risk of loss. Do your own
research and consult a licensed financial professional before making any
investment decision. The authors accept no liability for any use of this code.

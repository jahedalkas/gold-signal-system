"""
config.py
=========
Single source of truth for every tunable parameter in the gold signal system.

Nothing in this project should hard-code a magic number; import it from here
instead. This keeps experiments reproducible and lets you sweep parameters
without hunting through modules.

The file is organised into sections:
    1. Paths & I/O
    2. Data sources (tickers + FRED series + date range)
    3. Technical indicator parameters
    4. Macro signal thresholds
    5. Sentiment parameters
    6. Machine-learning parameters
    7. Signal combiner weights & thresholds
    8. Risk-management parameters
    9. Backtest parameters
   10. Logging configuration
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

# =============================================================================
# 1. PATHS & I/O
# =============================================================================
# Resolve everything relative to this file so the project runs from any cwd.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_CACHE_DIR: Final[Path] = PROJECT_ROOT / "data" / "_cache"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
CHARTS_DIR: Final[Path] = REPORTS_DIR / "charts"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "ml" / "_models"

# Create directories on import so downstream modules never hit "no such dir".
for _d in (DATA_CACHE_DIR, REPORTS_DIR, CHARTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Cache freshness: re-download a series only if its cache file is older than
# this many hours. Keeps repeated runs fast and avoids hammering free APIs.
CACHE_MAX_AGE_HOURS: Final[float] = 12.0

# =============================================================================
# 2. DATA SOURCES
# =============================================================================
# How many years of daily history to pull. Gold futures (GC=F) on Yahoo go
# back well beyond this; 7 years gives enough data for a 4-year train window
# plus a multi-year walk-forward test.
HISTORY_YEARS: Final[int] = 7

# yfinance tickers. Keys are the clean internal names used everywhere
# downstream; values are the Yahoo Finance symbols.
YF_TICKERS: Final[dict[str, str]] = {
    "gold":   "GC=F",       # Gold futures (XAU/USD proxy) — the asset we trade
    "silver": "SI=F",       # Silver — for the gold/silver ratio signal
    "dxy":    "DX-Y.NYB",   # US Dollar Index — inverse gold correlation
    "vix":    "^VIX",       # Volatility / fear index — safe-haven proxy
    "spx":    "^GSPC",      # S&P 500 — risk-on/risk-off proxy
    "oil":    "CL=F",       # WTI crude — inflation proxy
    "tnx":    "^TNX",       # US 10Y Treasury yield (x10, i.e. 42.0 == 4.20%)
    "gld":    "GLD",        # SPDR Gold ETF — used for volume / flow proxy
}

# The instrument we actually trade. Must be a key in YF_TICKERS.
PRIMARY_ASSET: Final[str] = "gold"

# FRED series IDs. Keys are internal names; values are FRED's series codes.
FRED_SERIES: Final[dict[str, str]] = {
    "cpi":        "CPIAUCSL",  # CPI (all urban consumers) — inflation level
    "real_yield": "DFII10",    # 10Y TIPS real yield — key gold driver
    "m2":         "M2SL",      # M2 money supply — liquidity / debasement proxy
    "fed_funds":  "DFF",       # Effective federal funds rate — for recession spread
}

# -----------------------------------------------------------------------------
# CRITICAL anti-lookahead setting.
# FRED dates economic series by their REFERENCE PERIOD, not their RELEASE date.
# Example: January CPI is dated 2023-01-01 on FRED but is not actually published
# until ~mid-February. If we aligned macro data by its FRED date, the backtest
# would "know" January's inflation in early January — ~6 weeks of future
# information leaking in, which would massively inflate results.
#
# We therefore delay each macro series' availability by a conservative
# publication lag (calendar days) before aligning it to the price calendar.
# Erring slightly LONG here is safe: it can only remove edge, never fabricate it.
# DFII10 (real yield) is a daily market series released with ~1 business day lag.
# -----------------------------------------------------------------------------
FRED_PUBLICATION_LAG_DAYS: Final[dict[str, int]] = {
    "cpi":        45,   # monthly; released ~2 weeks after month-end
    "m2":         45,   # monthly; released ~4 weeks after reference period
    "real_yield": 2,    # daily market data; ~next-business-day availability
    "fed_funds":  2,    # daily policy rate; ~next-business-day availability
}

# =============================================================================
# 3. TECHNICAL INDICATOR PARAMETERS
# =============================================================================
RSI_PERIOD: Final[int] = 14
RSI_PERIOD_FAST: Final[int] = 7        # second, faster RSI used as an ML feature
RSI_OVERBOUGHT: Final[float] = 70.0
RSI_OVERSOLD: Final[float] = 30.0

MACD_FAST: Final[int] = 12
MACD_SLOW: Final[int] = 26
MACD_SIGNAL: Final[int] = 9

BOLLINGER_PERIOD: Final[int] = 20
BOLLINGER_STD: Final[float] = 2.0

EMA_FAST: Final[int] = 20
EMA_SLOW: Final[int] = 50
EMA_LONG: Final[int] = 200

STOCH_PERIOD: Final[int] = 14
STOCH_SMOOTH: Final[int] = 3
STOCH_OVERBOUGHT: Final[float] = 80.0
STOCH_OVERSOLD: Final[float] = 20.0

ATR_PERIOD: Final[int] = 14

SMA_GOLDEN_FAST: Final[int] = 50       # Golden/Death cross fast SMA
SMA_GOLDEN_SLOW: Final[int] = 200      # Golden/Death cross slow SMA

VOLUME_ANOMALY_LOOKBACK: Final[int] = 20
VOLUME_ANOMALY_MULTIPLE: Final[float] = 2.0   # flag volume > 2x 20-day average

# =============================================================================
# 4. MACRO SIGNAL THRESHOLDS
# =============================================================================
MACRO_TREND_LOOKBACK: Final[int] = 20          # days for DXY / real-yield trend
CPI_BULLISH_LEVEL: Final[float] = 3.0          # YoY CPI % above this -> bullish
GOLD_SILVER_RATIO_HIGH: Final[float] = 85.0    # extreme high -> mean reversion
GOLD_SILVER_RATIO_LOW: Final[float] = 65.0     # extreme low for context
M2_GROWTH_LOOKBACK_MONTHS: Final[int] = 3       # months to measure M2 acceleration

# =============================================================================
# 5. SENTIMENT PARAMETERS
# =============================================================================
SENTIMENT_QUERY: Final[str] = "gold price OR XAU OR bullion OR gold market"
SENTIMENT_LANGUAGE: Final[str] = "en"
SENTIMENT_MAX_HEADLINES: Final[int] = 60
SENTIMENT_ROLLING_DAYS: Final[int] = 7
# RSS fallbacks used when NewsAPI is unavailable or quota-exceeded.
SENTIMENT_RSS_FEEDS: Final[tuple[str, ...]] = (
    "https://www.investing.com/rss/commodities_Gold.rss",
    "https://www.kitco.com/rss/category/news.xml",
)
# Try FinBERT first if transformers+torch are installed; else TextBlob.
SENTIMENT_USE_FINBERT_IF_AVAILABLE: Final[bool] = True

# =============================================================================
# 6. MACHINE-LEARNING PARAMETERS
# =============================================================================
ML_PREDICTION_HORIZON: Final[int] = 5    # predict direction/return 5 days ahead
ML_TRAIN_YEARS: Final[int] = 4           # initial training window length
ML_REGRESSOR_RETURN_SCALE = 0.02
ML_RETRAIN_MONTHS: Final[int] = 6        # walk-forward step / retrain cadence
ML_MIN_CONFIDENCE: Final[float] = 0.55   # below this, ML abstains (signal = 0)
ML_FEATURE_LAG: Final[int] = 1           # MINIMUM lag (days) on every feature
# Combiner parameters
COMBINER_AUC_WINDOW = 126        # 6-month rolling window for AUC check
COMBINER_AUC_BOOST_THRESHOLD = 0.60   # AUC above this = increase ML weight
COMBINER_AUC_FLOOR = 0.50        # AUC below this = disable ML signal
COMBINER_CRISIS_VIX = 30.0       # VIX above this = crisis mode
COMBINER_LOW_CONF_THRESHOLD = 0.55    # ML confidence below this = abstain

# Regressor scale
ML_REGRESSOR_RETURN_SCALE = 0.02  # 2% predicted return = score of 1.0

# TimeSeriesSplit folds for hyperparameter search (NEVER a random split).
ML_CV_SPLITS: Final[int] = 5

# Laptop-friendly hyperparameter search. We use RandomizedSearchCV with this
# many sampled combinations (instead of an exhaustive GridSearch over the full
# grid) and tune ONCE on the initial training window, then refit those params
# at each walk-forward step. This is the main lever keeping runtime in seconds.
ML_SEARCH_N_ITER: Final[int] = 12
ML_RANDOM_STATE: Final[int] = 42      # global seed for reproducibility

# XGBoost hyperparameter search grid (kept small to respect the 3-min budget).
ML_PARAM_GRID: Final[dict[str, list]] = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# =============================================================================
# 7. SIGNAL COMBINER — WEIGHTS & THRESHOLDS
# =============================================================================
SIGNAL_WEIGHTS: Final[dict[str, float]] = {
    "technical":     0.25,
    "macro":         0.25,
    "sentiment":     0.15,
    "ml_classifier": 0.20,
    "ml_regressor":  0.15,
}

# Dynamic re-weighting triggers (applied in engine/combiner.py).
COMBINER_ML_AUC_BOOST_THRESHOLD: Final[float] = 0.60   # AUC above -> trust ML more
COMBINER_ML_BOOSTED_WEIGHT: Final[float] = 0.40
COMBINER_CRISIS_VIX: Final[float] = 30.0               # VIX above -> macro dominates
COMBINER_CRISIS_MACRO_WEIGHT: Final[float] = 0.45

# Composite score thresholds in [-1, +1].
STRONG_BUY_THRESHOLD: Final[float] = 0.35
BUY_THRESHOLD: Final[float] = 0.20
SELL_THRESHOLD: Final[float] = -0.20
STRONG_SELL_THRESHOLD: Final[float] = -0.35

# =============================================================================
# 8. RISK-MANAGEMENT PARAMETERS
# =============================================================================
KELLY_FRACTION: Final[float] = 0.5          # half-Kelly for safety
MAX_POSITION_PCT: Final[float] = 1.0        # cap: 100% of capital
MIN_POSITION_PCT: Final[float] = 0.10       # floor when in a position: 10%

STOP_LOSS_ATR_MULTIPLE: Final[float] = 2.0      # stop = entry - 2*ATR
TAKE_PROFIT_ATR_MULTIPLE: Final[float] = 3.0    # target = entry + 3*ATR (3:1)
TRAILING_STOP_ATR_MULTIPLE: Final[float] = 1.5  # trail activates at +1.5*ATR

MAX_CONSECUTIVE_LOSSES: Final[int] = 3      # pause system after N losses in a row
DAILY_LOSS_LIMIT_PCT: Final[float] = 0.03   # pause if down >3% in a single day

# Volatility scaling: shrink size when ATR is this far above its average.
VOL_SCALING_LOOKBACK: Final[int] = 20
VOL_SCALING_TRIGGER: Final[float] = 1.5     # ATR > 1.5x its 20d avg -> shrink

# Regime detection.
ADX_TREND_THRESHOLD: Final[float] = 25.0    # ADX above -> trending
ADX_RANGE_THRESHOLD: Final[float] = 20.0    # ADX below -> ranging
VIX_CRISIS_THRESHOLD: Final[float] = 30.0   # crisis: halve positions
VIX_FEAR_THRESHOLD: Final[float] = 25.0     # elevated fear: safe-haven tilt

# =============================================================================
# 9. BACKTEST PARAMETERS
# =============================================================================
STARTING_CAPITAL: Final[float] = 10_000.0   # EUR
BACKTEST_YEARS: Final[int] = 3              # out-of-sample test span (after ML train)
TRANSACTION_COST_PCT: Final[float] = 0.0015  # 0.15% per trade
SLIPPAGE_PCT: Final[float] = 0.0005          # 0.05% per trade
LONG_ONLY: Final[bool] = True                # BUY = hold gold, SELL = hold cash
ALLOW_LEVERAGE: Final[bool] = False

# Risk-free rate for Sharpe/Sortino — EUR context.
RISK_FREE_RATE: Final[float] = 0.04          # 4% annual
TRADING_DAYS_PER_YEAR: Final[int] = 252

# =============================================================================
# 10. LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL: Final[int] = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
LOG_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | None = None) -> None:
    """Configure root logging once for the whole application.

    Call this near the top of ``main.py``. Idempotent: safe to call repeatedly
    (it will not stack duplicate handlers).

    Args:
        level: Optional override for the log level. Defaults to ``LOG_LEVEL``.
    """
    root = logging.getLogger()
    if root.handlers:  # already configured — don't add duplicate handlers
        if level is not None:
            root.setLevel(level)
        return
    logging.basicConfig(
        level=level if level is not None else LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )
    # yfinance/urllib are chatty at INFO; quiet them down.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_fred_api_key() -> str | None:
    """Return the FRED API key from the environment, or ``None`` if unset."""
    key = os.getenv("FRED_API_KEY", "").strip()
    return key or None


def get_newsapi_key() -> str | None:
    """Return the NewsAPI key from the environment, or ``None`` if unset."""
    key = os.getenv("NEWSAPI_KEY", "").strip()
    return key or None

# ── Asset configuration ──────────────────────────────────────────
PRIMARY_ASSET = "gold"
YF_TICKERS = {
    "gold":   "GC=F",
    "silver": "SI=F",
    "dxy":    "DX-Y.NYB",
    "vix":    "^VIX",
    "spx":    "^GSPC",
    "oil":    "CL=F",
    "tnx":    "^TNX",
    "gld":    "GLD",
}

# ── Signal thresholds ─────────────────────────────────────────────
BUY_THRESHOLD          = 0.20
STRONG_BUY_THRESHOLD   = 0.35
SELL_THRESHOLD         = -0.20
STRONG_SELL_THRESHOLD  = -0.35

# ── Signal weights ────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "technical":     0.25,
    "macro":         0.25,
    "sentiment":     0.15,
    "ml_classifier": 0.20,
    "ml_regressor":  0.15,
}

# ── Combiner ──────────────────────────────────────────────────────
COMBINER_ML_AUC_BOOST_THRESHOLD = 0.60
COMBINER_ML_BOOSTED_WEIGHT      = 0.40
COMBINER_CRISIS_MACRO_WEIGHT    = 0.45

# ── Technical indicators ──────────────────────────────────────────
ATR_PERIOD   = 14
EMA_FAST     = 20
EMA_SLOW     = 50
EMA_LONG     = 200

# ── ML parameters ─────────────────────────────────────────────────
ML_PREDICTION_HORIZON = 5
ML_MIN_CONFIDENCE     = 0.55

# ── Risk management ───────────────────────────────────────────────
STARTING_CAPITAL          = 10_000.0
KELLY_FRACTION            = 0.5
MAX_POSITION_PCT          = 1.0
MIN_POSITION_PCT          = 0.10
STOP_LOSS_ATR_MULTIPLE    = 2.0
TAKE_PROFIT_ATR_MULTIPLE  = 3.0
TRAILING_STOP_ATR_MULTIPLE = 1.5
MAX_CONSECUTIVE_LOSSES    = 3
DAILY_LOSS_LIMIT_PCT      = 0.03
PAUSE_COOLDOWN_DAYS       = 5
VIX_CRISIS_THRESHOLD      = 30.0
ADX_TREND_THRESHOLD       = 25.0
ADX_RANGE_THRESHOLD       = 20.0
VOL_SCALING_TRIGGER       = 1.5
VOL_SCALING_LOOKBACK      = 20

# ── Backtesting ───────────────────────────────────────────────────
TRANSACTION_COST_PCT  = 0.0015
SLIPPAGE_PCT          = 0.0005
RISK_FREE_RATE        = 0.04
TRADING_DAYS_PER_YEAR = 252

# ── Reporting ─────────────────────────────────────────────────────
from pathlib import Path
REPORTS_DIR = Path("reports")
CHARTS_DIR  = Path("reports/charts")
ROLLING_METRIC_WINDOW = 126
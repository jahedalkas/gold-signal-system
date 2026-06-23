"""
ml/features.py
==============
Build the model-ready feature matrix from the aligned master panel.

THE LEAKAGE RULE (read this — it is the most important thing in the project)
---------------------------------------------------------------------------
Every feature is computed *contemporaneously* (using data through each day's
close) and then the entire feature block is shifted forward by
``config.ML_FEATURE_LAG`` (default 1 day). The target looks *forward*.

Concretely, for a prediction dated ``t``:

    X[t]      = raw_features[t - LAG]        # model sees t-1 and earlier only
    y_clf[t]  = 1 if close[t+H] > close[t]   # did gold rise over the next H days?
    y_reg[t]  = close[t+H] / close[t] - 1    # the realised H-day forward return

So the model uses information from **t-1 and earlier** to predict the move from
``t``'s close to ``t+H``'s close. The backtester later executes that signal at
``t+1``'s open. There is therefore at least a one-day gap between the newest
data the model used and the price it transacts at — zero lookahead, with a
deliberate safety margin. The last ``H`` rows (unknown future) and the first
warm-up rows (indicator burn-in + lag) are dropped automatically.

Why lag the *whole* block uniformly rather than per-feature? Because a single
uniform shift is trivially auditable: there is exactly one place where "now"
becomes "as-of", and it is impossible for one stray un-lagged column to leak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

import config
import signals.sentiment as sentiment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureMatrix:
    """Container for the aligned, leakage-safe modelling data.

    Attributes:
        X: Feature matrix (already lagged), indexed by date.
        y_clf: Binary direction target (1 = up over the horizon), aligned to X.
        y_reg: Forward-return target (fraction, e.g. 0.023 == +2.3%), aligned.
        feature_names: Ordered list of feature column names.
    """
    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series
    feature_names: list[str]

    @property
    def index(self) -> pd.DatetimeIndex:
        """Dates common to features and targets."""
        return self.X.index


# =============================================================================
# Raw (contemporaneous) feature construction
# =============================================================================
def _price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Price/technical features from gold OHLCV (computed through each close)."""
    close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    f = pd.DataFrame(index=ohlcv.index)

    # Lagged-return features over several horizons.
    for h in (1, 3, 5, 10, 21):
        f[f"ret_{h}d"] = close.pct_change(h)

    # Rolling volatility of daily returns.
    daily_ret = close.pct_change()
    f["vol_10d"] = daily_ret.rolling(10).std()
    f["vol_21d"] = daily_ret.rolling(21).std()

    # Price position within the trailing 52-week range, normalised 0..1.
    win = config.TRADING_DAYS_PER_YEAR
    roll_min = close.rolling(win, min_periods=20).min()
    roll_max = close.rolling(win, min_periods=20).max()
    f["pos_52w"] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    # % distance from each EMA (negative = below the EMA).
    for span, label in ((config.EMA_FAST, "ema20"), (config.EMA_SLOW, "ema50"),
                        (config.EMA_LONG, "ema200")):
        ema = ta.trend.EMAIndicator(close, window=span).ema_indicator()
        f[f"dist_{label}"] = (close - ema) / ema

    # Momentum / oscillator levels (values, not signals).
    f["rsi_14"] = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()
    f["rsi_7"] = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD_FAST).rsi()
    macd = ta.trend.MACD(close, window_slow=config.MACD_SLOW,
                         window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    f["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close, window=config.BOLLINGER_PERIOD,
                                      window_dev=config.BOLLINGER_STD)
    f["bb_pct_b"] = bb.bollinger_pband()

    # ATR as a fraction of price (scale-free volatility).
    atr = ta.volatility.AverageTrueRange(high, low, close,
                                         window=config.ATR_PERIOD).average_true_range()
    f["atr_pct"] = atr / close

    return f


def _macro_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Macro features from the panel (each guarded for missing columns)."""
    f = pd.DataFrame(index=panel.index)

    def has(col: str) -> bool:
        return col in panel.columns

    if has("dxy_close"):
        for h in (1, 5, 21):
            f[f"dxy_ret_{h}d"] = panel["dxy_close"].pct_change(h)
    if has("vix_close"):
        f["vix_level"] = panel["vix_close"]
        f["vix_chg_5d"] = panel["vix_close"].diff(5)
    if has("spx_close"):
        f["spx_ret_5d"] = panel["spx_close"].pct_change(5)
        f["spx_ret_21d"] = panel["spx_close"].pct_change(21)
    if has("oil_close"):
        f["oil_ret_5d"] = panel["oil_close"].pct_change(5)
    if has("tnx_close"):
        f["tnx_level"] = panel["tnx_close"]
        f["tnx_chg_5d"] = panel["tnx_close"].diff(5)
    if has("real_yield"):
        f["real_yield_level"] = panel["real_yield"]
        f["real_yield_chg_5d"] = panel["real_yield"].diff(5)
    if has("gold_close") and has("silver_close"):
        f["gold_silver_ratio"] = panel["gold_close"] / panel["silver_close"].replace(0, np.nan)
    if has("gld_volume"):
        vol = panel["gld_volume"]
        mean20 = vol.rolling(20, min_periods=5).mean()
        std20 = vol.rolling(20, min_periods=5).std().replace(0, np.nan)
        f["gld_vol_z"] = (vol - mean20) / std20

    return f


def _sentiment_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Sentiment features from the backtestable VIX-based proxy.

    Note: historical news sentiment is not available from free APIs, so these
    features derive from the VIX safe-haven proxy (see ``signals/sentiment.py``).
    """
    f = pd.DataFrame(index=panel.index)
    try:
        sent = sentiment.compute_sentiment_signals(panel)["sentiment_score"]
    except Exception as exc:
        logger.warning("Sentiment features unavailable: %s", exc)
        return f
    f["sentiment"] = sent
    f["sentiment_avg_7d"] = sent.rolling(config.SENTIMENT_ROLLING_DAYS, min_periods=2).mean()
    f["sentiment_momentum"] = sent - f["sentiment_avg_7d"]
    return f


# =============================================================================
# Assembly with strict lagging + forward target
# =============================================================================
def _assemble_raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Assemble all contemporaneous (un-lagged) features from the panel.

    Shared by both the training matrix and the live-prediction path so the two
    can never drift apart. Infinities from tiny denominators are nulled.
    """
    p = config.PRIMARY_ASSET
    ohlcv = panel[[c for c in panel.columns if c.startswith(f"{p}_")]].rename(columns={
        f"{p}_open": "open", f"{p}_high": "high", f"{p}_low": "low",
        f"{p}_close": "close", f"{p}_volume": "volume",
    })
    raw = pd.concat(
        [_price_features(ohlcv), _macro_features(panel), _sentiment_features(panel)],
        axis=1,
    )
    return raw.replace([np.inf, -np.inf], np.nan)


def build_live_features(panel: pd.DataFrame, lag: int | None = None) -> pd.DataFrame:
    """Lagged feature matrix WITHOUT targets, for live prediction.

    Unlike ``build_feature_matrix`` (which drops the final ``horizon`` rows
    because their forward target is unknown), this keeps every row with complete
    lagged features — so the **last row is the most recent tradeable feature
    vector**, used to generate today's prediction.

    Args:
        panel: Aligned master panel.
        lag: Feature lag in days. Defaults to ``config.ML_FEATURE_LAG``.

    Returns:
        DataFrame of lagged features (rows with any NaN feature dropped).
    """
    lg = lag if lag is not None else config.ML_FEATURE_LAG
    raw = _assemble_raw_features(panel)
    return raw.shift(lg).dropna()


def build_feature_matrix(
    panel: pd.DataFrame,
    horizon: int | None = None,
    lag: int | None = None,
) -> FeatureMatrix:
    """Assemble the leakage-safe feature matrix and targets.

    Args:
        panel: Aligned master panel from ``preprocessor.build_master_panel``.
        horizon: Forward prediction horizon in trading days. Defaults to
            ``config.ML_PREDICTION_HORIZON``.
        lag: Days to lag every feature. Defaults to ``config.ML_FEATURE_LAG``.
            Must be >= 1 to guarantee no lookahead.

    Returns:
        A ``FeatureMatrix`` with aligned ``X``, ``y_clf``, ``y_reg``.

    Raises:
        ValueError: If ``lag`` < 1 (would permit lookahead) or gold close is
            missing from the panel.
    """
    h = horizon if horizon is not None else config.ML_PREDICTION_HORIZON
    lg = lag if lag is not None else config.ML_FEATURE_LAG
    if lg < 1:
        raise ValueError("Feature lag must be >= 1 to prevent lookahead bias.")

    close_col = f"{config.PRIMARY_ASSET}_close"
    if close_col not in panel.columns:
        raise ValueError(f"Panel missing '{close_col}' — cannot build features.")

    # 1) Contemporaneous features (through each day's close).
    raw = _assemble_raw_features(panel)

    # 2) Lag the ENTIRE feature block — the single, auditable "as-of" shift.
    X = raw.shift(lg)

    # 3) Forward-looking targets from the (un-lagged) close.
    close = panel[close_col]
    fwd_return = close.shift(-h) / close - 1.0
    y_reg = fwd_return.rename("target_return")
    y_clf = (fwd_return > 0).astype("Int64").rename("target_binary")
    # Where the forward window runs off the end, the target is unknown -> NaN.
    y_clf = y_clf.where(fwd_return.notna())

    # 4) Align and drop rows with any NaN in features or target.
    combined = pd.concat([X, y_reg, y_clf], axis=1).dropna()
    feature_names = list(raw.columns)
    X_clean = combined[feature_names].astype(float)
    y_reg_clean = combined["target_return"].astype(float)
    y_clf_clean = combined["target_binary"].astype(int)

    logger.info(
        "Feature matrix built: %d rows x %d features (horizon=%dd, lag=%dd), "
        "usable dates %s..%s. Up-day base rate=%.1f%%.",
        len(X_clean), len(feature_names), h, lg,
        X_clean.index.min().date(), X_clean.index.max().date(),
        100.0 * y_clf_clean.mean(),
    )
    return FeatureMatrix(
        X=X_clean, y_clf=y_clf_clean, y_reg=y_reg_clean, feature_names=feature_names
    )


if __name__ == "__main__":
    # Manual smoke test with an explicit leakage check:  python -m ml.features
    config.configure_logging()
    rng = np.random.default_rng(7)
    n = 900
    idx = pd.bdate_range("2021-01-01", periods=n)

    def walk(s, v):
        return s + np.cumsum(rng.normal(0, v, n))

    panel = pd.DataFrame({
        "gold_open": walk(1800, 12), "gold_high": walk(1810, 12),
        "gold_low": walk(1790, 12), "gold_close": walk(1800, 12),
        "gold_volume": rng.integers(1e4, 1e5, n).astype(float),
        "silver_close": walk(23, 0.3), "dxy_close": walk(103, 0.4),
        "vix_close": np.abs(walk(18, 1.2)) + 10, "spx_close": walk(4200, 35),
        "oil_close": walk(80, 1.5), "tnx_close": np.abs(walk(40, 0.5)),
        "real_yield": walk(1.8, 0.03), "gld_volume": rng.integers(5e6, 2e7, n).astype(float),
    }, index=idx)

    fm = build_feature_matrix(panel)
    print(f"\nX shape: {fm.X.shape}")
    print(f"Features: {fm.feature_names}")

    # Leakage check: the feature row for date t must equal the raw features
    # computed at t-LAG. We verify ret_1d lines up with the lagged close change.
    t = fm.X.index[-1]
    lag = config.ML_FEATURE_LAG
    prior = panel.index[panel.index.get_loc(t) - lag]
    expected_ret1d = panel["gold_close"].pct_change().loc[prior]
    actual = fm.X.loc[t, "ret_1d"]
    print(f"\nLeakage check on {t.date()}: ret_1d in X = {actual:.6f}, "
          f"raw ret_1d at t-{lag} ({prior.date()}) = {expected_ret1d:.6f}")
    assert abs(actual - expected_ret1d) < 1e-9, "LEAKAGE: feature not properly lagged!"
    print("PASS — features are lagged exactly, no lookahead.")

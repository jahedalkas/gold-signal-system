"""
signals/technical.py
====================
Technical-analysis signals computed on gold OHLCV using the ``ta`` library.

Output contract
---------------
Every indicator produces, for the **whole history**, a small DataFrame with two
columns:

    signal   : int in {-1, 0, +1}   (-1 bearish, 0 neutral, +1 bullish)
    strength : float in [0.0, 1.0]  (conviction of that signal)

Working with the full history (not just "today") is what lets the backtester
replay decisions day by day without recomputation. For human-readable output
(the dashboard / live predictor), ``latest_reasons`` builds the plain-English
explanation string for the most recent bar only.

Lookahead note
--------------
Indicators here are computed from each bar's *close*. That is correct: at the
close of day T you know day T's close. The 1-day execution lag (trade on day
T+1's open using day T's signal) is enforced in the **backtester**, and the
1-day minimum feature lag for the ML model is enforced in **ml/features.py**.
This module must therefore never be used to peek: read a row's signal as
"what I would have known at that day's close", nothing more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalResult:
    """A single signal reading for one point in time.

    Attributes:
        name: Human-readable indicator name.
        signal: -1 (bearish), 0 (neutral), or +1 (bullish).
        strength: Conviction in [0.0, 1.0].
        reason: Plain-English explanation of the reading.
    """
    name: str
    signal: int
    strength: float
    reason: str


# Column schema for the per-indicator history frames.
_COLS = ["signal", "strength"]


def _empty_history(index: pd.Index) -> pd.DataFrame:
    """Return a neutral (signal=0, strength=0) history frame on ``index``."""
    return pd.DataFrame(
        {"signal": np.zeros(len(index), dtype=int), "strength": 0.0},
        index=index,
    )


def _clip01(x: pd.Series | float) -> pd.Series | float:
    """Clip a value or series to [0, 1]."""
    if isinstance(x, pd.Series):
        return x.clip(0.0, 1.0)
    return float(min(1.0, max(0.0, x)))


# =============================================================================
# Individual indicators — each returns a full-history DataFrame[signal,strength]
# =============================================================================
def rsi_signal(df: pd.DataFrame) -> pd.DataFrame:
    """RSI(14): oversold (<30) is bullish, overbought (>70) is bearish.

    Strength scales with how far RSI is beyond the threshold (the deeper the
    oversold reading, the stronger the bullish signal).
    """
    rsi = ta.momentum.RSIIndicator(df["close"], window=config.RSI_PERIOD).rsi()
    out = _empty_history(df.index)

    oversold = rsi < config.RSI_OVERSOLD
    overbought = rsi > config.RSI_OVERBOUGHT
    out.loc[oversold, "signal"] = 1
    out.loc[overbought, "signal"] = -1

    # Strength: distance past the band, normalised by the 0..30 / 70..100 range.
    out.loc[oversold, "strength"] = _clip01(
        (config.RSI_OVERSOLD - rsi[oversold]) / config.RSI_OVERSOLD
    )
    out.loc[overbought, "strength"] = _clip01(
        (rsi[overbought] - config.RSI_OVERBOUGHT) / (100.0 - config.RSI_OVERBOUGHT)
    )
    return out


def macd_signal(df: pd.DataFrame) -> pd.DataFrame:
    """MACD(12,26,9): histogram sign gives direction; crossovers add strength.

    Bullish when the MACD line is above its signal line (histogram > 0),
    bearish when below. Strength scales with the absolute histogram value
    relative to its own recent range.
    """
    macd = ta.trend.MACD(
        df["close"],
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    hist = macd.macd_diff()
    out = _empty_history(df.index)

    out.loc[hist > 0, "signal"] = 1
    out.loc[hist < 0, "signal"] = -1

    # Normalise |hist| by a rolling scale so strength stays in [0,1] sensibly.
    scale = hist.abs().rolling(50, min_periods=10).mean().replace(0, np.nan)
    out["strength"] = _clip01((hist.abs() / scale).fillna(0.0))
    return out


def bollinger_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands(20,2): price below lower band bullish, above upper bearish.

    Mean-reversion logic: a close piercing the lower band is a stretched,
    potentially-oversold condition (bullish), and vice versa.
    """
    bb = ta.volatility.BollingerBands(
        df["close"], window=config.BOLLINGER_PERIOD, window_dev=config.BOLLINGER_STD
    )
    pct_b = bb.bollinger_pband()  # 0 at lower band, 1 at upper band
    out = _empty_history(df.index)

    below = pct_b < 0.0
    above = pct_b > 1.0
    out.loc[below, "signal"] = 1
    out.loc[above, "signal"] = -1
    out.loc[below, "strength"] = _clip01(-pct_b[below])
    out.loc[above, "strength"] = _clip01(pct_b[above] - 1.0)
    return out


def ema_cross_signal(df: pd.DataFrame) -> pd.DataFrame:
    """EMA trend stack (20/50/200): alignment of fast/slow/long EMAs.

    Fully bullish when EMA20 > EMA50 > EMA200 (uptrend stack); fully bearish
    when EMA20 < EMA50 < EMA200. Mixed alignment yields a weaker signal.
    """
    close = df["close"]
    ema_f = ta.trend.EMAIndicator(close, window=config.EMA_FAST).ema_indicator()
    ema_s = ta.trend.EMAIndicator(close, window=config.EMA_SLOW).ema_indicator()
    ema_l = ta.trend.EMAIndicator(close, window=config.EMA_LONG).ema_indicator()
    out = _empty_history(df.index)

    bull_stack = (ema_f > ema_s) & (ema_s > ema_l)
    bear_stack = (ema_f < ema_s) & (ema_s < ema_l)
    # Partial agreement: fast above/below slow only.
    fast_above = (ema_f > ema_s) & ~bull_stack
    fast_below = (ema_f < ema_s) & ~bear_stack

    out.loc[bull_stack, ["signal", "strength"]] = [1, 1.0]
    out.loc[bear_stack, ["signal", "strength"]] = [-1, 1.0]
    out.loc[fast_above, ["signal", "strength"]] = [1, 0.5]
    out.loc[fast_below, ["signal", "strength"]] = [-1, 0.5]
    return out


def stochastic_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic Oscillator(14,3): oversold (<20) bullish, overbought (>80) bearish."""
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=config.STOCH_PERIOD, smooth_window=config.STOCH_SMOOTH,
    )
    k = stoch.stoch()
    out = _empty_history(df.index)

    oversold = k < config.STOCH_OVERSOLD
    overbought = k > config.STOCH_OVERBOUGHT
    out.loc[oversold, "signal"] = 1
    out.loc[overbought, "signal"] = -1
    out.loc[oversold, "strength"] = _clip01(
        (config.STOCH_OVERSOLD - k[oversold]) / config.STOCH_OVERSOLD
    )
    out.loc[overbought, "strength"] = _clip01(
        (k[overbought] - config.STOCH_OVERBOUGHT) / (100.0 - config.STOCH_OVERBOUGHT)
    )
    return out


def obv_signal(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume: rising OBV trend confirms bullish, falling bearish.

    We compare OBV to its own 20-day EMA: above = accumulation (bullish),
    below = distribution (bearish). Volume can be flat/zero for some futures
    feeds, in which case this signal stays neutral.
    """
    out = _empty_history(df.index)
    if "volume" not in df.columns or df["volume"].fillna(0).sum() == 0:
        logger.debug("OBV skipped — no usable volume data.")
        return out

    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"].fillna(0)
    ).on_balance_volume()
    obv_ema = obv.ewm(span=20, min_periods=10).mean()

    out.loc[obv > obv_ema, "signal"] = 1
    out.loc[obv < obv_ema, "signal"] = -1
    # Strength: normalised distance of OBV from its EMA.
    scale = obv.rolling(50, min_periods=10).std().replace(0, np.nan)
    out["strength"] = _clip01(((obv - obv_ema).abs() / scale).fillna(0.0))
    return out


def golden_death_cross_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Golden Cross (50 SMA > 200 SMA) bullish; Death Cross bearish.

    This is a slow, structural trend signal. Strength is boosted briefly right
    at the crossover bar (a fresh cross is more informative than a long-standing
    one), then settles to a steady baseline while the regime persists.
    """
    close = df["close"]
    sma_f = close.rolling(config.SMA_GOLDEN_FAST).mean()
    sma_s = close.rolling(config.SMA_GOLDEN_SLOW).mean()
    out = _empty_history(df.index)

    golden = sma_f > sma_s
    death = sma_f < sma_s
    out.loc[golden, "signal"] = 1
    out.loc[death, "signal"] = -1
    out["strength"] = 0.4  # steady baseline while regime holds

    # Boost the bar where the cross actually happens.
    cross_up = golden & ~golden.shift(1, fill_value=False)
    cross_down = death & ~death.shift(1, fill_value=False)
    out.loc[cross_up | cross_down, "strength"] = 1.0
    out.loc[~(golden | death), "strength"] = 0.0
    return out


def volume_anomaly_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Volume spike (>2x 20-day average) on an up/down day confirms that move.

    A surge in volume aligned with a green candle is bullish confirmation; with
    a red candle, bearish. No spike -> neutral. Useful as a conviction overlay.
    """
    out = _empty_history(df.index)
    if "volume" not in df.columns or df["volume"].fillna(0).sum() == 0:
        return out

    vol = df["volume"].fillna(0)
    avg = vol.rolling(config.VOLUME_ANOMALY_LOOKBACK, min_periods=5).mean()
    ratio = vol / avg.replace(0, np.nan)
    spike = ratio > config.VOLUME_ANOMALY_MULTIPLE

    direction = np.sign(df["close"].diff()).fillna(0)
    out.loc[spike & (direction > 0), "signal"] = 1
    out.loc[spike & (direction < 0), "signal"] = -1
    # Strength scales with how extreme the spike is (2x..4x -> 0..1).
    out.loc[spike, "strength"] = _clip01(
        (ratio[spike] - config.VOLUME_ANOMALY_MULTIPLE) / config.VOLUME_ANOMALY_MULTIPLE
    )
    return out


def atr_value(df: pd.DataFrame) -> pd.Series:
    """Average True Range(14) — a volatility *measure*, not a directional signal.

    Returned as a raw series (price units) for the risk module to size stops,
    take-profits and position scaling. It is intentionally not part of the
    bullish/bearish aggregation.
    """
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=config.ATR_PERIOD
    ).average_true_range()
    atr.name = "atr"
    return atr


# Registry of directional indicators -> function. ATR is excluded (volatility).
_INDICATORS = {
    "RSI": rsi_signal,
    "MACD": macd_signal,
    "Bollinger": bollinger_signal,
    "EMA_stack": ema_cross_signal,
    "Stochastic": stochastic_signal,
    "OBV": obv_signal,
    "Golden_Death_Cross": golden_death_cross_signal,
    "Volume_anomaly": volume_anomaly_signal,
}


# =============================================================================
# Aggregation
# =============================================================================
def compute_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run all directional technical indicators over the full history.

    Args:
        df: OHLCV DataFrame (columns ``open, high, low, close`` and optionally
            ``volume``) indexed by date — e.g. from
            ``preprocessor.get_primary_ohlcv``.

    Returns:
        A DataFrame indexed like ``df`` with two columns per indicator:
        ``<name>_signal`` and ``<name>_strength``, plus an aggregate
        ``technical_score`` in [-1, +1] (the strength-weighted mean of the
        individual signals). This aggregate is what ``engine/combiner.py``
        consumes for the "technical" component.
    """
    pieces: list[pd.DataFrame] = []
    weighted_sum = pd.Series(0.0, index=df.index)
    strength_sum = pd.Series(0.0, index=df.index)

    for name, func in _INDICATORS.items():
        try:
            res = func(df)
        except Exception as exc:
            logger.warning("Technical indicator '%s' failed: %s", name, exc)
            res = _empty_history(df.index)
        res = res.rename(columns={"signal": f"{name}_signal",
                                  "strength": f"{name}_strength"})
        pieces.append(res)
        weighted_sum = weighted_sum.add(
            res[f"{name}_signal"] * res[f"{name}_strength"], fill_value=0.0
        )
        strength_sum = strength_sum.add(res[f"{name}_strength"], fill_value=0.0)

    out = pd.concat(pieces, axis=1)
    # Aggregate score: strength-weighted average of signals, in [-1, +1].
    out["technical_score"] = (weighted_sum / strength_sum.replace(0, np.nan)).fillna(0.0)
    return out


def latest_reasons(df: pd.DataFrame) -> dict[str, SignalResult]:
    """Build human-readable ``SignalResult`` objects for the most recent bar.

    Used by the dashboard and live predictor to explain *why* each indicator is
    bullish/bearish/neutral right now. Computed only for the final row, so it is
    cheap regardless of history length.

    Args:
        df: OHLCV DataFrame indexed by date.

    Returns:
        Mapping of indicator name -> ``SignalResult`` for the latest date.
    """
    results: dict[str, SignalResult] = {}
    last = df.index[-1]

    # Recompute the indicator histories once and read their last row, then add
    # tailored explanatory text per indicator.
    rsi = ta.momentum.RSIIndicator(df["close"], window=config.RSI_PERIOD).rsi().iloc[-1]

    for name, func in _INDICATORS.items():
        try:
            hist = func(df)
            sig = int(hist["signal"].iloc[-1])
            strength = float(hist["strength"].iloc[-1])
        except Exception as exc:
            logger.warning("latest_reasons: '%s' failed: %s", name, exc)
            sig, strength = 0, 0.0

        reason = _reason_text(name, sig, strength, rsi=rsi)
        results[name] = SignalResult(name=name, signal=sig,
                                     strength=round(strength, 3), reason=reason)

    logger.debug("Computed latest technical reasons for %s.", last.date())
    return results


def _reason_text(name: str, signal: int, strength: float, rsi: float) -> str:
    """Compose a short plain-English reason for one indicator's latest reading."""
    direction = {1: "bullish", -1: "bearish", 0: "neutral"}[signal]
    if name == "RSI":
        if signal == 1:
            return f"RSI {rsi:.0f} is oversold (<{config.RSI_OVERSOLD:.0f}) — bullish"
        if signal == -1:
            return f"RSI {rsi:.0f} is overbought (>{config.RSI_OVERBOUGHT:.0f}) — bearish"
        return f"RSI {rsi:.0f} is in the neutral zone"
    descriptions = {
        "MACD": "MACD histogram",
        "Bollinger": "price vs Bollinger Bands",
        "EMA_stack": "EMA 20/50/200 trend stack",
        "Stochastic": "Stochastic oscillator",
        "OBV": "On-Balance Volume trend",
        "Golden_Death_Cross": "50/200 SMA cross",
        "Volume_anomaly": "volume spike confirmation",
    }
    base = descriptions.get(name, name)
    if signal == 0:
        return f"{base}: neutral"
    return f"{base}: {direction} (strength {strength:.2f})"


if __name__ == "__main__":
    # Manual smoke test on synthetic data:  python -m signals.technical
    config.configure_logging()
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = pd.Series(1800 + np.cumsum(rng.normal(0, 12, n)), index=idx)
    ohlcv = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + rng.random(n) * 6,
        "low": close - rng.random(n) * 6,
        "close": close,
        "volume": rng.integers(1e4, 1e5, n).astype(float),
    }, index=idx)

    sig = compute_technical_signals(ohlcv)
    print(sig[["technical_score"]].tail())
    print("\nLatest reasons:")
    for r in latest_reasons(ohlcv).values():
        print(f"  {r.name:20s} signal={r.signal:+d} strength={r.strength:.2f} | {r.reason}")

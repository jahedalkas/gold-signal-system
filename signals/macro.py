"""
signals/macro.py
================
Macro / fundamental signals for gold, computed on the aligned master panel
produced by ``data/preprocessor.py``.

Why macro matters for gold
--------------------------
Gold pays no yield, so its opportunity cost is the *real* (inflation-adjusted)
return on safe assets. The cleanest macro driver is therefore the 10Y real
yield: when real yields rise, holding non-yielding gold gets more expensive
(bearish); when they fall, gold gets relatively more attractive (bullish). The
US dollar is the second lever — gold is priced in USD, so a stronger dollar is
a mechanical headwind. The rest (inflation level, money supply, oil, risk-off
fear, the recession spread) are slower, contextual drivers.

Output contract
---------------
Identical to ``signals/technical.py``: every signal returns a full-history
DataFrame with ``signal ∈ {-1,0,+1}`` and ``strength ∈ [0,1]``, and
``compute_macro_signals`` adds an aggregate ``macro_score`` in [-1,+1].

Graceful degradation
--------------------
Each signal needs specific panel columns (e.g. the real-yield signal needs
``real_yield``, which only exists if FRED was reachable). If a required column
is missing, that signal returns a neutral (all-zero) series and logs a WARNING,
so the macro layer still works on whatever data is present.

A note on direction conventions you should sanity-check
------------------------------------------------------
The CPI signal treats inflation >3% as bullish (the classic "gold is an
inflation hedge" view). In reality the picture is more nuanced — high inflation
can trigger rate hikes that *raise* real yields and hurt gold (see 2022). The
real-yield signal is the counterweight that captures that. Keep this tension in
mind rather than reading the CPI signal in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalResult:
    """A single macro signal reading for one point in time."""
    name: str
    signal: int
    strength: float
    reason: str


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


def _require(panel: pd.DataFrame, *cols: str) -> bool:
    """Return True if all ``cols`` are present in ``panel``, else log and False."""
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        logger.warning("Macro signal skipped — missing columns: %s", missing)
        return False
    return True


def _zscore_strength(metric: pd.Series, lookback: int = 252) -> pd.Series:
    """Map a metric to a [0,1] strength via a rolling z-score (2σ -> full).

    A move of ~2 standard deviations relative to its own recent history is
    treated as maximum conviction. This keeps strengths comparable across
    signals without hand-tuning a scale for each one.
    """
    mean = metric.rolling(lookback, min_periods=20).mean()
    std = metric.rolling(lookback, min_periods=20).std().replace(0, np.nan)
    z = (metric - mean) / std
    return _clip01((z.abs() / 2.0).fillna(0.0))


def _to_pct_yield(series: pd.Series) -> pd.Series:
    """Normalise a Treasury-yield series to percent units.

    Yahoo's ``^TNX`` has historically been quoted either as the yield in percent
    (e.g. 4.25) or scaled x10 (e.g. 42.5), depending on era/version. If the
    median looks implausibly large for a yield (>20), assume the x10 convention
    and divide. This guards a real, version-dependent data-shape gotcha.
    """
    med = series.dropna().median()
    if pd.notna(med) and med > 20:
        return series / 10.0
    return series


# =============================================================================
# Individual macro signals
# =============================================================================
def dxy_trend_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """US Dollar Index trend: rising DXY is bearish gold, falling is bullish."""
    out = _empty_history(panel.index)
    if not _require(panel, "dxy_close"):
        return out
    lb = config.MACRO_TREND_LOOKBACK
    ret = panel["dxy_close"].pct_change(lb)
    out.loc[ret > 0, "signal"] = -1   # dollar up -> gold headwind
    out.loc[ret < 0, "signal"] = 1    # dollar down -> gold tailwind
    out["strength"] = _zscore_strength(ret)
    return out


def real_yield_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """10Y real yield direction: rising real yields bearish, falling bullish.

    The single most important macro driver of gold. We look at the change over
    the trend lookback window and lean on it with a touch more conviction.
    """
    out = _empty_history(panel.index)
    if not _require(panel, "real_yield"):
        return out
    lb = config.MACRO_TREND_LOOKBACK
    change = panel["real_yield"].diff(lb)
    out.loc[change > 0, "signal"] = -1
    out.loc[change < 0, "signal"] = 1
    # Real yields are the strongest driver — give a modest conviction boost.
    out["strength"] = _clip01(_zscore_strength(change) * 1.2)
    return out


def cpi_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """CPI year-over-year inflation: above 3% is treated as bullish gold.

    CPIAUCSL is an index *level*, not a rate, so we derive YoY inflation as the
    ~252-trading-day percentage change of the (publication-lagged, forward-
    filled) index. See the module docstring for the nuance on this signal.
    """
    out = _empty_history(panel.index)
    if not _require(panel, "cpi"):
        return out
    yoy = panel["cpi"].pct_change(config.TRADING_DAYS_PER_YEAR) * 100.0
    bullish = yoy > config.CPI_BULLISH_LEVEL
    out.loc[bullish, "signal"] = 1
    # Strength scales with how far inflation runs above the 3% threshold.
    out.loc[bullish, "strength"] = _clip01(
        (yoy[bullish] - config.CPI_BULLISH_LEVEL) / config.CPI_BULLISH_LEVEL
    )
    return out


def m2_growth_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """M2 money supply: accelerating growth bullish, outright contraction bearish.

    Expanding liquidity / debasement is a tailwind for gold; the rare episodes
    of M2 *contraction* (e.g. 2022-23) are a meaningful liquidity headwind.
    """
    out = _empty_history(panel.index)
    if not _require(panel, "m2"):
        return out
    window = config.M2_GROWTH_LOOKBACK_MONTHS * 21  # ~trading days per month
    growth = panel["m2"].pct_change(window)
    prev_growth = growth.shift(window)
    accelerating = growth > prev_growth

    out.loc[(growth > 0) & accelerating, "signal"] = 1
    out.loc[growth < 0, "signal"] = -1
    out["strength"] = _zscore_strength(growth)
    return out


def gold_silver_ratio_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Gold/silver ratio extremes as a contextual mean-reversion signal.

    A very high ratio (>85) means gold is historically expensive versus silver —
    often a sentiment extreme for the precious-metals complex that has preceded
    broad metal strength (read here as mildly bullish gold). A very low ratio
    (<65) is the opposite. This is a *weak, contextual* signal — treat its
    conviction as low and never trade it in isolation.
    """
    out = _empty_history(panel.index)
    if not _require(panel, "gold_close", "silver_close"):
        return out
    ratio = panel["gold_close"] / panel["silver_close"].replace(0, np.nan)
    high = ratio > config.GOLD_SILVER_RATIO_HIGH
    low = ratio < config.GOLD_SILVER_RATIO_LOW
    out.loc[high, "signal"] = 1
    out.loc[low, "signal"] = -1
    # Modest strength, capped well below 1 to reflect low conviction.
    out.loc[high, "strength"] = _clip01(
        (ratio[high] - config.GOLD_SILVER_RATIO_HIGH) / 15.0
    ) * 0.6
    out.loc[low, "strength"] = _clip01(
        (config.GOLD_SILVER_RATIO_LOW - ratio[low]) / 15.0
    ) * 0.6
    return out


def oil_trend_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """WTI crude trend as an inflation proxy: rising oil mildly bullish gold."""
    out = _empty_history(panel.index)
    if not _require(panel, "oil_close"):
        return out
    lb = config.MACRO_TREND_LOOKBACK
    ret = panel["oil_close"].pct_change(lb)
    out.loc[ret > 0, "signal"] = 1
    out.loc[ret < 0, "signal"] = -1
    out["strength"] = _zscore_strength(ret) * 0.7  # secondary driver
    return out


def risk_off_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Risk-off safe-haven demand: VIX elevated AND S&P falling is bullish gold.

    Both conditions must hold (fear *and* falling equities). When they don't,
    the signal is neutral rather than bearish — gold can also rise in risk-on
    regimes for other reasons, so absence of fear is not a sell.
    """
    out = _empty_history(panel.index)
    if not _require(panel, "vix_close", "spx_close"):
        return out
    spx_5d = panel["spx_close"].pct_change(5)
    risk_off = (panel["vix_close"] > config.VIX_FEAR_THRESHOLD) & (spx_5d < 0)
    out.loc[risk_off, "signal"] = 1
    out.loc[risk_off, "strength"] = _clip01(
        (panel["vix_close"][risk_off] - config.VIX_FEAR_THRESHOLD) / 25.0
    )
    return out


def recession_spread_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """10Y yield minus Fed funds rate: an inverted (negative) spread is bullish.

    When the policy rate sits above the 10Y yield, the market is pricing future
    cuts / slowdown — a classic late-cycle recession signal that historically
    supports gold (anticipated easing + safe-haven demand).
    """
    out = _empty_history(panel.index)
    if not _require(panel, "tnx_close", "fed_funds"):
        return out
    tnx_pct = _to_pct_yield(panel["tnx_close"])
    spread = tnx_pct - panel["fed_funds"]
    inverted = spread < 0
    out.loc[inverted, "signal"] = 1
    out.loc[inverted, "strength"] = _clip01(spread[inverted].abs() / 1.0)
    return out


# Registry of macro signals.
_SIGNALS = {
    "DXY_trend": dxy_trend_signal,
    "Real_yield": real_yield_signal,
    "CPI_inflation": cpi_signal,
    "M2_growth": m2_growth_signal,
    "Gold_Silver_ratio": gold_silver_ratio_signal,
    "Oil_trend": oil_trend_signal,
    "Risk_off": risk_off_signal,
    "Recession_spread": recession_spread_signal,
}


# =============================================================================
# Aggregation
# =============================================================================
def compute_macro_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Run all macro signals over the full history and aggregate.

    Args:
        panel: The aligned master panel from ``preprocessor.build_master_panel``.

    Returns:
        DataFrame indexed like ``panel`` with ``<name>_signal`` and
        ``<name>_strength`` per signal, plus an aggregate ``macro_score`` in
        [-1, +1] (strength-weighted mean of the individual signals). This
        aggregate is what ``engine/combiner.py`` consumes for "macro".
    """
    pieces: list[pd.DataFrame] = []
    weighted_sum = pd.Series(0.0, index=panel.index)
    strength_sum = pd.Series(0.0, index=panel.index)

    for name, func in _SIGNALS.items():
        try:
            res = func(panel)
        except Exception as exc:
            logger.warning("Macro signal '%s' failed: %s", name, exc)
            res = _empty_history(panel.index)
        res = res.rename(columns={"signal": f"{name}_signal",
                                  "strength": f"{name}_strength"})
        pieces.append(res)
        weighted_sum = weighted_sum.add(
            res[f"{name}_signal"] * res[f"{name}_strength"], fill_value=0.0
        )
        strength_sum = strength_sum.add(res[f"{name}_strength"], fill_value=0.0)

    out = pd.concat(pieces, axis=1)
    out["macro_score"] = (weighted_sum / strength_sum.replace(0, np.nan)).fillna(0.0)
    return out


def latest_reasons(panel: pd.DataFrame) -> dict[str, SignalResult]:
    """Build human-readable ``SignalResult`` objects for the most recent bar."""
    results: dict[str, SignalResult] = {}
    for name, func in _SIGNALS.items():
        try:
            hist = func(panel)
            sig = int(hist["signal"].iloc[-1])
            strength = float(hist["strength"].iloc[-1])
        except Exception as exc:
            logger.warning("latest_reasons: macro '%s' failed: %s", name, exc)
            sig, strength = 0, 0.0
        results[name] = SignalResult(
            name=name, signal=sig, strength=round(strength, 3),
            reason=_reason_text(name, sig, strength),
        )
    return results


def _reason_text(name: str, signal: int, strength: float) -> str:
    """Compose a short plain-English reason for one macro signal's reading."""
    direction = {1: "bullish", -1: "bearish", 0: "neutral"}[signal]
    labels = {
        "DXY_trend": "Dollar (DXY) trend",
        "Real_yield": "10Y real yield direction",
        "CPI_inflation": "CPI inflation level",
        "M2_growth": "M2 money-supply growth",
        "Gold_Silver_ratio": "gold/silver ratio extreme",
        "Oil_trend": "oil (inflation proxy) trend",
        "Risk_off": "risk-off / safe-haven demand",
        "Recession_spread": "10Y–Fed funds recession spread",
    }
    base = labels.get(name, name)
    if signal == 0:
        return f"{base}: neutral"
    return f"{base}: {direction} (strength {strength:.2f})"


if __name__ == "__main__":
    # Manual smoke test on a synthetic panel:  python -m signals.macro
    config.configure_logging()
    rng = np.random.default_rng(11)
    n = 600
    idx = pd.bdate_range("2022-01-01", periods=n)

    def walk(start, vol):
        return start + np.cumsum(rng.normal(0, vol, n))

    panel = pd.DataFrame({
        "gold_close": walk(1800, 12),
        "silver_close": walk(23, 0.3),
        "dxy_close": walk(103, 0.4),
        "vix_close": np.abs(walk(18, 1.2)) + 10,
        "spx_close": walk(4200, 35),
        "oil_close": walk(80, 1.5),
        "tnx_close": np.abs(walk(40, 0.5)),       # ^TNX in x10 convention
        "real_yield": walk(1.8, 0.03),
        "cpi": 290 + np.cumsum(np.abs(rng.normal(0.05, 0.02, n))),
        "m2": 21000 + np.cumsum(rng.normal(2, 5, n)),
        "fed_funds": np.clip(walk(4.5, 0.05), 0, None),
    }, index=idx)

    sig = compute_macro_signals(panel)
    print(sig[["macro_score"]].tail())
    print("\nLatest macro reasons:")
    for r in latest_reasons(panel).values():
        print(f"  {r.name:20s} signal={r.signal:+d} strength={r.strength:.2f} | {r.reason}")

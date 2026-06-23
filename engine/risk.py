"""
engine/risk.py
=============
Position sizing and risk controls.

Pieces
------
* **Half-Kelly sizing** from the model's win probability and the take-profit /
  stop-loss payoff ratio, then shrunk for high volatility and crisis, and
  clamped to [``MIN_POSITION_PCT``, ``MAX_POSITION_PCT``].
* **ATR-based exits**: stop-loss at entry - ``STOP_LOSS_ATR_MULTIPLE``*ATR,
  take-profit at entry + ``TAKE_PROFIT_ATR_MULTIPLE``*ATR, and a trailing stop
  that activates once price is ``TRAILING_STOP_ATR_MULTIPLE``*ATR in profit.
* **Regime detection** from ADX and VIX (trending / ranging / crisis / normal).
* **RiskGuard**: stateful guard that pauses *new entries* after too many
  consecutive losses or a daily loss-limit breach.

A frank caveat on Kelly
-----------------------
Kelly sizing assumes you actually know your edge and odds. Here the "win
probability" is the model's classifier confidence, which is itself uncertain —
so full Kelly would badly over-bet. We use **half**-Kelly and hard caps as
deliberate humility. Treat position sizes as risk-control, not a promise of
optimality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ta

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Indicators used by the risk layer
# =============================================================================
def adx_series(ohlcv: pd.DataFrame) -> pd.Series:
    """Average Directional Index — trend strength (not direction)."""
    adx = ta.trend.ADXIndicator(
        high=ohlcv["high"], low=ohlcv["low"], close=ohlcv["close"],
        window=config.ATR_PERIOD,
    ).adx()
    adx.name = "adx"
    return adx


# =============================================================================
# Position sizing
# =============================================================================
def payoff_ratio() -> float:
    """Reward/risk ratio implied by the ATR take-profit and stop multiples."""
    return config.TAKE_PROFIT_ATR_MULTIPLE / config.STOP_LOSS_ATR_MULTIPLE


def half_kelly_fraction(win_prob: float, payoff: float | None = None) -> float:
    """Half-Kelly bet fraction for a binary payoff.

    Kelly f* = p - q/b, where p = win prob, q = 1-p, b = reward/risk. We return
    half of that (capped at 0 on the downside — a non-positive Kelly means
    "don't bet").

    Args:
        win_prob: Estimated probability of a winning trade (the model's
            confidence), in [0, 1].
        payoff: Reward/risk ratio. Defaults to ``payoff_ratio()``.

    Returns:
        Half-Kelly fraction in [0, 1] (0 if the edge is non-positive).
    """
    b = payoff if payoff is not None else payoff_ratio()
    p = float(np.clip(win_prob, 0.0, 1.0))
    q = 1.0 - p
    kelly = p - q / b if b > 0 else 0.0
    return float(max(0.0, kelly * config.KELLY_FRACTION))


def volatility_scalar(atr: float, atr_avg: float) -> float:
    """Shrink factor in (0,1] applied when ATR is well above its average.

    When ATR exceeds ``VOL_SCALING_TRIGGER`` x its 20-day average, scale down
    proportionally so that elevated volatility reduces exposure; otherwise 1.0.
    """
    if not (atr and atr_avg) or atr_avg <= 0:
        return 1.0
    trigger = config.VOL_SCALING_TRIGGER * atr_avg
    if atr <= trigger:
        return 1.0
    return float(np.clip(trigger / atr, 0.25, 1.0))


def position_size(
    win_prob: float,
    atr: float,
    atr_avg: float,
    vix: float,
) -> float:
    """Final position size as a fraction of capital, after all adjustments.

    Pipeline: half-Kelly -> volatility shrink -> crisis halving -> clamp. If the
    Kelly edge is non-positive the result is 0 (no position). Otherwise the
    result is clamped to [``MIN_POSITION_PCT``, ``MAX_POSITION_PCT``].

    Args:
        win_prob: Model win probability / confidence.
        atr: Current ATR (price units).
        atr_avg: Trailing average ATR (price units).
        vix: Current VIX level (NaN tolerated).

    Returns:
        Position size fraction in {0} ∪ [MIN_POSITION_PCT, MAX_POSITION_PCT].
    """
    size = half_kelly_fraction(win_prob)
    if size <= 0.0:
        return 0.0
    size *= volatility_scalar(atr, atr_avg)
    if pd.notna(vix) and vix > config.VIX_CRISIS_THRESHOLD:
        size *= 0.5  # crisis: halve exposure
    return float(np.clip(size, config.MIN_POSITION_PCT, config.MAX_POSITION_PCT))


# =============================================================================
# Stop / take-profit levels
# =============================================================================
@dataclass(frozen=True)
class StopLevels:
    """Exit levels for an open long position (price units)."""
    stop_loss: float
    take_profit: float
    trail_trigger: float   # price at which the trailing stop activates


def stop_levels(entry_price: float, atr: float) -> StopLevels:
    """Compute initial stop, take-profit, and trailing-activation levels."""
    return StopLevels(
        stop_loss=entry_price - config.STOP_LOSS_ATR_MULTIPLE * atr,
        take_profit=entry_price + config.TAKE_PROFIT_ATR_MULTIPLE * atr,
        trail_trigger=entry_price + config.TRAILING_STOP_ATR_MULTIPLE * atr,
    )


def updated_trailing_stop(high_water: float, atr: float, current_stop: float) -> float:
    """Raise the stop to ``high_water - TRAILING_STOP_ATR_MULTIPLE*ATR`` if higher.

    A trailing stop only ever moves up (for a long), locking in gains as the
    high-water mark rises. Returns the greater of the current stop and the new
    trailed level.

    Args:
        high_water: Highest price seen since entry.
        atr: Current ATR (price units).
        current_stop: The stop level currently in force.

    Returns:
        The (possibly raised) stop level.
    """
    trailed = high_water - config.TRAILING_STOP_ATR_MULTIPLE * atr
    return float(max(current_stop, trailed))


# =============================================================================
# Regime detection
# =============================================================================
def detect_regime(adx: float, vix: float) -> str:
    """Classify the market regime from ADX (trend strength) and VIX (fear).

    Returns one of: ``"crisis"`` (VIX high), ``"trending"`` (ADX high),
    ``"ranging"`` (ADX low), or ``"normal"`` otherwise.
    """
    if pd.notna(vix) and vix > config.VIX_CRISIS_THRESHOLD:
        return "crisis"
    if pd.notna(adx) and adx > config.ADX_TREND_THRESHOLD:
        return "trending"
    if pd.notna(adx) and adx < config.ADX_RANGE_THRESHOLD:
        return "ranging"
    return "normal"


# =============================================================================
# Stateful guard: pause new entries after losses
# =============================================================================
@dataclass
class RiskGuard:
    """Tracks loss streaks and daily losses to pause *new entries*.

    The guard never forces an exit — it only blocks opening new positions while
    a cooldown is active. Existing positions are still managed by their stops.

    Attributes:
        consecutive_losses: Running count of consecutive losing trades.
        cooldown_remaining: Trading days left in a pause.
        max_consecutive_losses: Trips a pause when reached.
        cooldown_days: Length of the pause once tripped.
    """
    consecutive_losses: int = 0
    cooldown_remaining: int = 0
    max_consecutive_losses: int = field(default_factory=lambda: config.MAX_CONSECUTIVE_LOSSES)
    cooldown_days: int = field(default_factory=lambda: config.PAUSE_COOLDOWN_DAYS)

    def can_enter(self) -> bool:
        """True if new entries are currently allowed (no active cooldown)."""
        return self.cooldown_remaining <= 0

    def record_trade(self, trade_return: float) -> None:
        """Update loss streak after a closed trade; trip a pause if needed."""
        if trade_return < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_remaining = self.cooldown_days
                logger.info(
                    "RiskGuard tripped: %d consecutive losses -> pausing new "
                    "entries for %d trading days.",
                    self.consecutive_losses, self.cooldown_days,
                )
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0

    def register_daily_loss(self, daily_return: float) -> None:
        """Trip a pause if a single day's loss breaches the daily limit."""
        if daily_return < -config.DAILY_LOSS_LIMIT_PCT and self.cooldown_remaining <= 0:
            self.cooldown_remaining = self.cooldown_days
            logger.info(
                "RiskGuard tripped: daily loss %.2f%% breached the %.0f%% limit "
                "-> pausing new entries for %d trading days.",
                daily_return * 100, config.DAILY_LOSS_LIMIT_PCT * 100, self.cooldown_days,
            )

    def step_day(self) -> None:
        """Advance one trading day, decrementing any active cooldown."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1


if __name__ == "__main__":
    # Smoke test:  python -m engine.risk
    config.configure_logging()
    print("--- Kelly sizing (payoff %.2f) ---" % payoff_ratio())
    for p in (0.50, 0.55, 0.60, 0.70, 0.80):
        print(f"  win_prob={p:.2f} -> half-Kelly={half_kelly_fraction(p):.3f}, "
              f"sized={position_size(p, atr=20, atr_avg=18, vix=18):.3f}")
    print("\n--- High-vol + crisis shrink (win_prob 0.70) ---")
    print(f"  calm:   {position_size(0.70, atr=18, atr_avg=18, vix=15):.3f}")
    print(f"  hi-vol: {position_size(0.70, atr=40, atr_avg=18, vix=15):.3f}")
    print(f"  crisis: {position_size(0.70, atr=18, atr_avg=18, vix=35):.3f}")

    print("\n--- Stops (entry 2000, ATR 25) ---")
    s = stop_levels(2000, 25)
    print(f"  stop={s.stop_loss:.0f} target={s.take_profit:.0f} trail_trigger={s.trail_trigger:.0f}")
    print(f"  trailed stop at high-water 2080: {updated_trailing_stop(2080, 25, s.stop_loss):.0f}")

    print("\n--- Regime ---")
    for adx, vix in ((30, 18), (15, 18), (22, 18), (22, 35)):
        print(f"  adx={adx} vix={vix} -> {detect_regime(adx, vix)}")

    print("\n--- RiskGuard ---")
    g = RiskGuard()
    for r in (-0.01, -0.02, -0.01):
        g.record_trade(r)
    print(f"  after 3 losses, can_enter={g.can_enter()} cooldown={g.cooldown_remaining}")

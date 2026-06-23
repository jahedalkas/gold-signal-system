"""
engine/backtester.py
===================
Event-driven, long-only backtest with strict no-lookahead.

The no-lookahead contract (the whole point)
-------------------------------------------
* A recommendation is read at the **close of day t** (from the combiner, whose
  ML component is the genuine walk-forward out-of-sample prediction).
* Any resulting entry/exit is executed at the **open of day t+1**.
* While a position is open, stop-loss / take-profit / trailing-stop levels are
  checked against day t's own high/low — valid, because the position was opened
  on an earlier bar. Within a bar we check the **stop before the target**
  (worst-case fill) so results are never optimistic.

Costs are charged on every fill (``TRANSACTION_COST_PCT`` + ``SLIPPAGE_PCT``).
Position size comes from the risk module (half-Kelly, vol/crisis shrink). Cash
not allocated to gold earns nothing (a conservative simplification).

Outputs (``BacktestResult``)
---------------------------
* ``daily``        — per-day close, equity, benchmark, position size, action.
* ``trades``       — one row per closed trade with entry/exit, P&L, reason.
* ``metrics``      — ``{"strategy": {...}, "benchmark": {...}}`` full stats.
* ``equity_curve`` / ``benchmark_curve`` — convenience series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta

import config
from engine import risk

logger = logging.getLogger(__name__)

_BUY_ACTIONS = {"BUY", "STRONG BUY"}
_SELL_ACTIONS = {"SELL", "STRONG SELL"}


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    equity_curve: pd.Series
    benchmark_curve: pd.Series


# =============================================================================
# Open-position bookkeeping
# =============================================================================
@dataclass
class _Position:
    """Mutable state for a single open long position."""
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    notional: float          # cash put to work at entry (pre-fee)
    entry_fee: float
    entry_score: float
    entry_confidence: float
    stop_loss: float
    take_profit: float
    trail_trigger: float
    high_water: float
    trailing_active: bool = False
    min_close: float = np.inf  # for in-trade drawdown


# =============================================================================
# Main engine
# =============================================================================
def run_backtest(ohlcv_full: pd.DataFrame, combined: pd.DataFrame) -> BacktestResult:
    """Run the long-only backtest over the dates covered by ``combined``.

    Args:
        ohlcv_full: Full-history gold OHLCV (``open/high/low/close`` indexed by
            date). Indicators are computed on this so there is no warm-up gap,
            and next-day opens are available for execution.
        combined: Output of ``combiner.combine_signals`` — its index defines the
            tradeable period and it supplies ``action``, ``composite_score``,
            ``ml_confidence`` and ``vix`` per day.

    Returns:
        A populated ``BacktestResult``.

    Raises:
        ValueError: If the combined dates are not a subset of the OHLCV index.
    """
    if not combined.index.isin(ohlcv_full.index).all():
        raise ValueError("Combined signal dates must be a subset of the OHLCV index.")

    # --- Indicators for the risk layer (computed on full history) ------------
    close, high, low = ohlcv_full["close"], ohlcv_full["high"], ohlcv_full["low"]
    atr = ta.volatility.AverageTrueRange(
        high, low, close, window=config.ATR_PERIOD
    ).average_true_range()
    atr_avg = atr.rolling(config.VOL_SCALING_LOOKBACK, min_periods=5).mean()
    adx = risk.adx_series(ohlcv_full)

    test_dates = combined.index
    all_dates = ohlcv_full.index
    cost = config.TRANSACTION_COST_PCT + config.SLIPPAGE_PCT

    capital = float(config.STARTING_CAPITAL)
    cash = capital
    pos: _Position | None = None
    guard = risk.RiskGuard()

    daily_rows: list[dict] = []
    trades: list[dict] = []
    prev_equity = capital

    for t in test_dates:
        i = all_dates.get_loc(t)
        px_close = float(close.iloc[i])
        bar_high, bar_low = float(high.iloc[i]), float(low.iloc[i])

        # --- 1. Manage an open position against THIS bar's range -------------
        if pos is not None:
            pos.high_water = max(pos.high_water, bar_high)
            pos.min_close = min(pos.min_close, px_close)
            # Activate / advance the trailing stop once in enough profit.
            if pos.high_water >= pos.trail_trigger:
                pos.trailing_active = True
            if pos.trailing_active:
                pos.stop_loss = risk.updated_trailing_stop(
                    pos.high_water, float(atr.iloc[i]), pos.stop_loss
                )

            exit_price, reason = None, None
            # Worst-case ordering: stop checked before target.
            if bar_low <= pos.stop_loss:
                exit_price = pos.stop_loss
                reason = "trailing_stop" if pos.trailing_active else "stop_loss"
            elif bar_high >= pos.take_profit:
                exit_price = pos.take_profit
                reason = "take_profit"

            if exit_price is not None:
                cash, trade = _close_position(pos, t, exit_price, reason, cost, cash)
                trades.append(trade)
                guard.record_trade(trade["return_pct"])
                pos = None

        # --- 2. Read the signal at THIS close, execute at NEXT open ----------
        action = str(combined.at[t, "action"])
        has_next = i + 1 < len(all_dates)
        next_open = float(ohlcv_full["open"].iloc[i + 1]) if has_next else None

        if has_next:
            if pos is not None and action in _SELL_ACTIONS:
                cash, trade = _close_position(pos, all_dates[i + 1], next_open,
                                              "signal", cost, cash)
                trades.append(trade)
                guard.record_trade(trade["return_pct"])
                pos = None
            elif pos is None and action in _BUY_ACTIONS and guard.can_enter():
                size = risk.position_size(
                    win_prob=float(combined.at[t, "ml_confidence"]),
                    atr=float(atr.iloc[i]), atr_avg=float(atr_avg.iloc[i]),
                    vix=float(combined.at[t, "vix"]) if "vix" in combined else np.nan,
                )
                if size > 0:
                    pos, cash = _open_position(
                        cash=cash, equity=prev_equity, size=size,
                        date=all_dates[i + 1], price=next_open, atr=float(atr.iloc[i]),
                        score=float(combined.at[t, "composite_score"]),
                        confidence=float(combined.at[t, "ml_confidence"]), cost=cost,
                    )

        # --- 3. Mark-to-market, record the day, advance guard ----------------
        equity = cash + (pos.units * px_close if pos is not None else 0.0)
        daily_return = equity / prev_equity - 1.0 if prev_equity > 0 else 0.0
        guard.register_daily_loss(daily_return)
        guard.step_day()

        daily_rows.append({
            "close": px_close,
            "equity": equity,
            "position_size": (pos.units * px_close / equity) if pos is not None and equity > 0 else 0.0,
            "in_position": pos is not None,
            "action": action,
            "composite_score": float(combined.at[t, "composite_score"]),
        })
        prev_equity = equity

    # --- Close any position still open at the end ----------------------------
    if pos is not None:
        last = test_dates[-1]
        cash, trade = _close_position(pos, last, float(close.loc[last]),
                                      "end_of_data", cost, cash)
        trades.append(trade)
        daily_rows[-1]["equity"] = cash
        daily_rows[-1]["in_position"] = False
        daily_rows[-1]["position_size"] = 0.0

    daily = pd.DataFrame(daily_rows, index=test_dates)
    trades_df = pd.DataFrame(trades)
    benchmark = _buy_and_hold(ohlcv_full, test_dates, cost)
    daily["benchmark"] = benchmark

    metrics = {
        "strategy": _performance_metrics(daily["equity"], trades_df),
        "benchmark": _performance_metrics(benchmark, pd.DataFrame()),
    }
    _log_headline(metrics)

    return BacktestResult(
        daily=daily, trades=trades_df, metrics=metrics,
        equity_curve=daily["equity"], benchmark_curve=benchmark,
    )


# =============================================================================
# Entry / exit helpers
# =============================================================================
def _open_position(cash, equity, size, date, price, atr, score, confidence, cost):
    """Open a long position sized as ``size`` of current equity; charge fees."""
    notional = equity * size
    fee = notional * cost
    units = notional / price
    new_cash = cash - notional - fee
    levels = risk.stop_levels(price, atr)
    pos = _Position(
        entry_date=date, entry_price=price, units=units, notional=notional,
        entry_fee=fee, entry_score=score, entry_confidence=confidence,
        stop_loss=levels.stop_loss, take_profit=levels.take_profit,
        trail_trigger=levels.trail_trigger, high_water=price, min_close=price,
    )
    logger.debug("ENTER %s @ %.2f size=%.0f%% stop=%.2f tp=%.2f",
                 date.date(), price, size * 100, levels.stop_loss, levels.take_profit)
    return pos, new_cash


def _close_position(pos: _Position, date, price, reason, cost, cash):
    """Close ``pos`` at ``price``; return (updated_cash, trade dict).

    Adds net proceeds (after exit fee) to ``cash`` and builds a trade record
    with currency P&L and percentage return on the entry notional.
    """
    proceeds = pos.units * price
    exit_fee = proceeds * cost
    net_proceeds = proceeds - exit_fee
    pnl = net_proceeds - pos.notional - pos.entry_fee
    return_pct = pnl / pos.notional if pos.notional > 0 else 0.0
    holding_days = max(1, (date - pos.entry_date).days)
    in_trade_dd = (pos.min_close / pos.entry_price - 1.0)

    trade = {
        "entry_date": pos.entry_date.date(), "entry_price": round(pos.entry_price, 2),
        "exit_date": date.date(), "exit_price": round(price, 2),
        "notional": round(pos.notional, 2), "pnl_eur": round(pnl, 2),
        "return_pct": return_pct, "holding_days": holding_days,
        "entry_score": round(pos.entry_score, 3),
        "entry_confidence": round(pos.entry_confidence, 3),
        "max_drawdown_in_trade": round(in_trade_dd, 4), "exit_reason": reason,
    }
    logger.debug("EXIT  %s @ %.2f (%s) pnl=%.2f (%.2f%%)",
                 date.date(), price, reason, pnl, return_pct * 100)
    return cash + net_proceeds, trade


def _buy_and_hold(ohlcv_full: pd.DataFrame, test_dates, cost: float) -> pd.Series:
    """Buy-and-hold benchmark: invest all capital at the first test open, hold."""
    first = test_dates[0]
    i = ohlcv_full.index.get_loc(first)
    entry = float(ohlcv_full["open"].iloc[i])
    units = config.STARTING_CAPITAL * (1.0 - cost) / entry
    curve = units * ohlcv_full["close"].reindex(test_dates)
    curve.name = "benchmark"
    return curve


# =============================================================================
# Performance metrics
# =============================================================================
def _performance_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Compute the full performance metric set for an equity curve + trade log."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    n_days = len(equity)
    years = n_days / config.TRADING_DAYS_PER_YEAR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    rf_daily = config.RISK_FREE_RATE / config.TRADING_DAYS_PER_YEAR
    excess = rets - rf_daily
    ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = (excess.mean() / rets.std() * ann) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std()
    sortino = (excess.mean() / downside * ann) if downside and downside > 0 else np.nan

    max_dd, dd_dur = _max_drawdown(equity)
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

    m = {
        "total_return": total_return, "cagr": cagr, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "max_dd_duration_days": dd_dur,
        "calmar": calmar, "volatility_annual": rets.std() * ann,
        "n_days": n_days,
    }

    if not trades.empty:
        r = trades["return_pct"]
        wins, losses = r[r > 0], r[r <= 0]
        gross_profit = trades.loc[trades["pnl_eur"] > 0, "pnl_eur"].sum()
        gross_loss = abs(trades.loc[trades["pnl_eur"] <= 0, "pnl_eur"].sum())
        m.update({
            "n_trades": int(len(trades)),
            "win_rate": float((r > 0).mean()),
            "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
            "avg_holding_days": float(trades["holding_days"].mean()),
            "best_trade_pct": float(r.max()), "worst_trade_pct": float(r.min()),
        })
    else:
        m.update({"n_trades": 0, "win_rate": np.nan})
    return m


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return (max drawdown as a negative fraction, longest underwater days)."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min())
    # Longest stretch below a prior peak.
    underwater = dd < 0
    longest, current = 0, 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max_dd, longest


def _log_headline(metrics: dict) -> None:
    """Log strategy-vs-benchmark headline numbers, drawdown-first (no hype)."""
    s, b = metrics["strategy"], metrics["benchmark"]
    logger.info(
        "Backtest result (strategy vs buy & hold):\n"
        "  Max drawdown:  %6.1f%%  vs %6.1f%%\n"
        "  Sharpe:        %6.2f   vs %6.2f\n"
        "  CAGR:          %6.1f%%  vs %6.1f%%\n"
        "  Total return:  %6.1f%%  vs %6.1f%%\n"
        "  Trades: %s | Win rate: %s | Profit factor: %s",
        s.get("max_drawdown", float("nan")) * 100, b.get("max_drawdown", float("nan")) * 100,
        s.get("sharpe", float("nan")), b.get("sharpe", float("nan")),
        s.get("cagr", float("nan")) * 100, b.get("cagr", float("nan")) * 100,
        s.get("total_return", float("nan")) * 100, b.get("total_return", float("nan")) * 100,
        s.get("n_trades", 0),
        f"{s.get('win_rate', float('nan')):.1%}" if not np.isnan(s.get("win_rate", float("nan"))) else "n/a",
        f"{s.get('profit_factor', float('nan')):.2f}" if not np.isnan(s.get("profit_factor", float("nan"))) else "n/a",
    )


if __name__ == "__main__":
    # Smoke test on synthetic data:  python -m engine.backtester
    config.configure_logging()
    rng = np.random.default_rng(31)
    n = 800
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(1800 + np.cumsum(rng.normal(0.2, 12, n)), index=idx)
    ohlcv = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + np.abs(rng.normal(0, 6, n)),
        "low": close - np.abs(rng.normal(0, 6, n)),
        "close": close,
    }, index=idx)
    # Fabricate a combiner output over the back half.
    test = idx[400:]
    score = pd.Series(np.tanh(np.cumsum(rng.normal(0, 0.3, len(test))) / 5), index=test)
    combined = pd.DataFrame({
        "composite_score": score,
        "action": score.apply(lambda s: "STRONG BUY" if s > 0.35 else "BUY" if s > 0.2
                               else "STRONG SELL" if s < -0.35 else "SELL" if s < -0.2 else "HOLD"),
        "ml_confidence": pd.Series(rng.uniform(0.55, 0.8, len(test)), index=test),
        "vix": pd.Series(np.abs(rng.normal(20, 5, len(test))), index=test),
    }, index=test)

    res = run_backtest(ohlcv, combined)
    print(f"\nTrades: {len(res.trades)}")
    if not res.trades.empty:
        print(res.trades[["entry_date", "exit_date", "return_pct", "exit_reason"]].head(8).to_string(index=False))
    s = res.metrics["strategy"]
    print(f"\nStrategy: total={s['total_return']:.1%} sharpe={s['sharpe']:.2f} "
          f"maxDD={s['max_drawdown']:.1%} winrate={s.get('win_rate', float('nan')):.1%}")

"""
reporting/dashboard.py
=====================
Turn a ``BacktestResult`` (plus the combined signals and ML result) into the
full set of charts and text reports.

Charts saved to ``reports/charts`` as PNG:
  1. Gold price with EMAs, entry/exit markers, and volume
  2. Equity curve: strategy vs buy-and-hold, with drawdown shaded
  3. Rolling drawdown
  4. ML model performance (delegated to ml.evaluator: ROC, SHAP, calibration,
     predicted-vs-actual)
  5. Signal-component heatmap (which signals drove decisions over time)
  6. Monthly returns heatmap (calendar grid)
  7. Rolling risk metrics (Sharpe, win rate, volatility)

Also writes ``reports/trades.csv`` and ``reports/summary.txt`` and prints a
console summary. Every chart is wrapped so a single failure never aborts the
whole report.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ta  # noqa: E402

import config  # noqa: E402
from engine.backtester import BacktestResult  # noqa: E402

logger = logging.getLogger(__name__)


def generate_dashboard(
    ohlcv_full: pd.DataFrame,
    result: BacktestResult,
    combined: pd.DataFrame,
    walk_forward_result=None,
) -> None:
    """Generate all charts and text reports for a completed backtest.

    Args:
        ohlcv_full: Full gold OHLCV (for the price chart and EMAs).
        result: The completed ``BacktestResult``.
        combined: The combiner output (for the signal-component heatmap).
        walk_forward_result: Optional ``WalkForwardResult`` for the ML charts.
    """
    test_idx = result.daily.index
    _chart_price_signals(ohlcv_full, result, test_idx)
    _chart_equity(result)
    _chart_drawdown(result)
    if walk_forward_result is not None:
        _chart_ml(walk_forward_result)
    _chart_signal_heatmap(combined)
    _chart_monthly_returns(result.equity_curve)
    _chart_rolling_metrics(result)

    save_trade_log(result.trades)
    save_summary(result.metrics, result.trades)
    print_summary(result.metrics, result.trades)
    logger.info("Dashboard complete — charts in %s, reports in %s.",
                config.CHARTS_DIR, config.REPORTS_DIR)


def _safe(fig, name: str) -> None:
    """Save and close a figure, logging (not raising) on failure."""
    try:
        fig.tight_layout()
        fig.savefig(config.CHARTS_DIR / name, dpi=120, bbox_inches="tight")
    except Exception as exc:
        logger.warning("Failed to save chart '%s': %s", name, exc)
    finally:
        plt.close(fig)


# =============================================================================
# 1. Price + signals
# =============================================================================
def _chart_price_signals(ohlcv_full, result: BacktestResult, test_idx) -> None:
    """Candlestick gold price over the test window with EMAs, markers, volume."""
    try:
        df = ohlcv_full.reindex(test_idx)
        close = ohlcv_full["close"]
        ema_f = ta.trend.EMAIndicator(close, config.EMA_FAST).ema_indicator().reindex(test_idx)
        ema_s = ta.trend.EMAIndicator(close, config.EMA_SLOW).ema_indicator().reindex(test_idx)
        ema_l = ta.trend.EMAIndicator(close, config.EMA_LONG).ema_indicator().reindex(test_idx)

        has_vol = "volume" in df.columns and df["volume"].fillna(0).sum() > 0
        fig, axes = plt.subplots(
            2 if has_vol else 1, 1, figsize=(13, 8 if has_vol else 6),
            sharex=True, gridspec_kw={"height_ratios": [3, 1]} if has_vol else None,
        )
        ax = axes[0] if has_vol else axes
        x = np.arange(len(df))

        # Simplified candlesticks.
        for k, (o, h, l, c) in enumerate(zip(df["open"], df["high"], df["low"], df["close"])):
            if np.isnan(o) or np.isnan(c):
                continue
            colour = "#2ca02c" if c >= o else "#d62728"
            ax.plot([k, k], [l, h], color=colour, linewidth=0.5, zorder=1)
            ax.add_patch(plt.Rectangle((k - 0.3, min(o, c)), 0.6, abs(c - o) or 1e-6,
                                       color=colour, zorder=2))

        ax.plot(x, ema_f.values, label=f"EMA{config.EMA_FAST}", linewidth=1)
        ax.plot(x, ema_s.values, label=f"EMA{config.EMA_SLOW}", linewidth=1)
        ax.plot(x, ema_l.values, label=f"EMA{config.EMA_LONG}", linewidth=1)

        # Entry/exit markers from the trade log.
        pos_of = {d: k for k, d in enumerate(test_idx)}
        for _, tr in result.trades.iterrows():
            ed, xd = pd.Timestamp(tr["entry_date"]), pd.Timestamp(tr["exit_date"])
            if ed in pos_of:
                ax.scatter(pos_of[ed], tr["entry_price"], marker="^", color="green",
                           s=70, zorder=5)
            if xd in pos_of:
                ax.scatter(pos_of[xd], tr["exit_price"], marker="v", color="red",
                           s=70, zorder=5)

        ax.set_title("Gold price with signals (▲ entry, ▼ exit)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=8)
        _set_date_ticks(ax, test_idx, x)

        if has_vol:
            axes[1].bar(x, df["volume"].values, color="grey", alpha=0.6)
            axes[1].set_ylabel("Volume")
            _set_date_ticks(axes[1], test_idx, x)
        _safe(fig, "01_price_signals.png")
    except Exception as exc:
        logger.warning("Price/signals chart failed: %s", exc)


def _set_date_ticks(ax, idx, x, n: int = 8) -> None:
    """Place ~n readable date ticks along an integer x-axis."""
    step = max(1, len(idx) // n)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in idx[::step]], rotation=45, ha="right")


# =============================================================================
# 2 & 3. Equity + drawdown
# =============================================================================
def _chart_equity(result: BacktestResult) -> None:
    """Strategy vs buy-and-hold equity, with strategy drawdown shaded."""
    try:
        eq, bench = result.equity_curve, result.benchmark_curve
        dd = eq / eq.cummax() - 1.0
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(eq.index, eq.values, label="Strategy", linewidth=1.5)
        ax.plot(bench.index, bench.values, label="Buy & hold", linewidth=1.2, alpha=0.8)
        ax.fill_between(eq.index, eq.values, eq.cummax().values,
                        where=dd.values < 0, color="red", alpha=0.12,
                        label="Strategy drawdown")
        ax.set_title("Equity curve: strategy vs buy-and-hold")
        ax.set_ylabel("Portfolio value (€)")
        ax.legend(loc="upper left")
        _safe(fig, "02_equity_curve.png")
    except Exception as exc:
        logger.warning("Equity chart failed: %s", exc)


def _chart_drawdown(result: BacktestResult) -> None:
    """Rolling drawdown of the strategy equity curve."""
    try:
        eq = result.equity_curve
        dd = (eq / eq.cummax() - 1.0) * 100
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
        ax.set_title("Strategy drawdown over time")
        ax.set_ylabel("Drawdown (%)")
        _safe(fig, "03_drawdown.png")
    except Exception as exc:
        logger.warning("Drawdown chart failed: %s", exc)


# =============================================================================
# 4. ML performance (delegated to the evaluator)
# =============================================================================
def _chart_ml(walk_forward_result) -> None:
    """Generate ML diagnostic charts via the evaluator (ROC/SHAP/calibration)."""
    try:
        from ml.evaluator import evaluate

        evaluate(walk_forward_result, save_charts=True)
    except Exception as exc:
        logger.warning("ML charts failed: %s", exc)


# =============================================================================
# 5. Signal-component heatmap
# =============================================================================
def _chart_signal_heatmap(combined: pd.DataFrame) -> None:
    """Heatmap of each signal component's score over time (drivers of decisions)."""
    try:
        cols = [c for c in combined.columns if c.startswith("score_")]
        if not cols:
            return
        data = combined[cols].T
        labels = [c.replace("score_", "") for c in cols]
        fig, ax = plt.subplots(figsize=(13, 4))
        im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(labels)), labels)
        idx = combined.index
        step = max(1, len(idx) // 8)
        ax.set_xticks(range(0, len(idx), step),
                      [d.strftime("%Y-%m") for d in idx[::step]], rotation=45, ha="right")
        ax.set_title("Signal components over time (green=bullish, red=bearish)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        _safe(fig, "05_signal_heatmap.png")
    except Exception as exc:
        logger.warning("Signal heatmap failed: %s", exc)


# =============================================================================
# 6. Monthly returns heatmap
# =============================================================================
def _chart_monthly_returns(equity: pd.Series) -> None:
    """Calendar-style year x month grid of strategy returns."""
    try:
        monthly = equity.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            return
        frame = pd.DataFrame({
            "year": monthly.index.year, "month": monthly.index.month,
            "ret": monthly.values * 100,
        })
        grid = frame.pivot(index="year", columns="month", values="ret")
        fig, ax = plt.subplots(figsize=(11, 1 + 0.6 * len(grid)))
        im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto",
                       vmin=-np.nanmax(np.abs(grid.values)),
                       vmax=np.nanmax(np.abs(grid.values)))
        ax.set_xticks(range(len(grid.columns)),
                      [pd.Timestamp(2000, m, 1).strftime("%b") for m in grid.columns])
        ax.set_yticks(range(len(grid.index)), grid.index)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("Monthly returns (%)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        _safe(fig, "06_monthly_returns.png")
    except Exception as exc:
        logger.warning("Monthly returns chart failed: %s", exc)


# =============================================================================
# 7. Rolling risk metrics
# =============================================================================
def _chart_rolling_metrics(result: BacktestResult) -> None:
    """Rolling Sharpe, win rate (over recent trades), and annualised volatility."""
    try:
        eq = result.equity_curve
        rets = eq.pct_change().dropna()
        w = config.ROLLING_METRIC_WINDOW
        ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
        rf_daily = config.RISK_FREE_RATE / config.TRADING_DAYS_PER_YEAR
        roll_sharpe = (rets - rf_daily).rolling(w).mean() / rets.rolling(w).std() * ann
        roll_vol = rets.rolling(w).std() * ann * 100

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(roll_sharpe.index, roll_sharpe.values, color="#1f77b4")
        axes[0].axhline(0, color="grey", lw=0.6)
        axes[0].set_ylabel("Rolling Sharpe")
        axes[0].set_title(f"Rolling metrics ({w}-day window)")

        # Rolling win rate over the last 10 closed trades.
        if not result.trades.empty:
            wins = (result.trades.assign(
                exit_dt=pd.to_datetime(result.trades["exit_date"]))
                .set_index("exit_dt")["return_pct"] > 0).astype(float)
            roll_wr = wins.rolling(10, min_periods=3).mean() * 100
            axes[1].plot(roll_wr.index, roll_wr.values, color="#2ca02c", marker="o", ms=3)
        axes[1].axhline(50, color="grey", lw=0.6, ls="--")
        axes[1].set_ylabel("Win rate % (last 10)")

        axes[2].plot(roll_vol.index, roll_vol.values, color="#d62728")
        axes[2].set_ylabel("Ann. volatility %")
        _safe(fig, "07_rolling_metrics.png")
    except Exception as exc:
        logger.warning("Rolling metrics chart failed: %s", exc)


# =============================================================================
# Text reports
# =============================================================================
def save_trade_log(trades: pd.DataFrame) -> None:
    """Write the full trade log to ``reports/trades.csv``."""
    path = config.REPORTS_DIR / "trades.csv"
    trades.to_csv(path, index=False)
    logger.info("Saved %d trades to %s", len(trades), path)


def _format_summary(metrics: dict, trades: pd.DataFrame) -> str:
    """Build the plain-text performance summary (strategy vs benchmark)."""
    s, b = metrics.get("strategy", {}), metrics.get("benchmark", {})

    def pct(d, k):
        v = d.get(k, float("nan"))
        return f"{v * 100:7.2f}%" if not (isinstance(v, float) and np.isnan(v)) else "    n/a"

    def num(d, k):
        v = d.get(k, float("nan"))
        return f"{v:7.2f}" if not (isinstance(v, float) and np.isnan(v)) else "    n/a"

    lines = [
        "=" * 64,
        "  GOLD MULTI-FACTOR STRATEGY — PERFORMANCE SUMMARY",
        "=" * 64,
        "",
        f"{'Metric':<26}{'Strategy':>12}{'Buy & Hold':>14}",
        "-" * 64,
        f"{'Total return':<26}{pct(s,'total_return'):>12}{pct(b,'total_return'):>14}",
        f"{'CAGR':<26}{pct(s,'cagr'):>12}{pct(b,'cagr'):>14}",
        f"{'Sharpe ratio':<26}{num(s,'sharpe'):>12}{num(b,'sharpe'):>14}",
        f"{'Sortino ratio':<26}{num(s,'sortino'):>12}{num(b,'sortino'):>14}",
        f"{'Max drawdown':<26}{pct(s,'max_drawdown'):>12}{pct(b,'max_drawdown'):>14}",
        f"{'Calmar ratio':<26}{num(s,'calmar'):>12}{num(b,'calmar'):>14}",
        f"{'Annual volatility':<26}{pct(s,'volatility_annual'):>12}{pct(b,'volatility_annual'):>14}",
        "",
        "  Trade statistics (strategy)",
        "-" * 64,
        f"{'Number of trades':<26}{s.get('n_trades', 0):>12}",
        f"{'Win rate':<26}{pct(s,'win_rate'):>12}",
        f"{'Average win':<26}{pct(s,'avg_win_pct'):>12}",
        f"{'Average loss':<26}{pct(s,'avg_loss_pct'):>12}",
        f"{'Profit factor':<26}{num(s,'profit_factor'):>12}",
        f"{'Avg holding (days)':<26}{num(s,'avg_holding_days'):>12}",
        f"{'Best trade':<26}{pct(s,'best_trade_pct'):>12}",
        f"{'Worst trade':<26}{pct(s,'worst_trade_pct'):>12}",
        f"{'Max DD duration (days)':<26}{s.get('max_dd_duration_days', 0):>12}",
        "",
        "=" * 64,
        "  Educational/research use only. Not financial advice.",
        "  Past performance does not guarantee future results.",
        "=" * 64,
    ]
    return "\n".join(lines)


def save_summary(metrics: dict, trades: pd.DataFrame) -> None:
    """Write the performance summary to ``reports/summary.txt``."""
    text = _format_summary(metrics, trades)
    (config.REPORTS_DIR / "summary.txt").write_text(text)
    logger.info("Saved summary to %s", config.REPORTS_DIR / "summary.txt")


def print_summary(metrics: dict, trades: pd.DataFrame) -> None:
    """Print the performance summary to the console."""
    print("\n" + _format_summary(metrics, trades) + "\n")


if __name__ == "__main__":
    # Smoke test: synthetic backtest then full dashboard.
    from engine.backtester import run_backtest

    config.configure_logging()
    rng = np.random.default_rng(33)
    n = 800
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(1800 + np.cumsum(rng.normal(0.3, 12, n)), index=idx)
    ohlcv = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + np.abs(rng.normal(0, 6, n)),
        "low": close - np.abs(rng.normal(0, 6, n)),
        "close": close, "volume": rng.integers(1e4, 1e5, n).astype(float),
    }, index=idx)
    test = idx[400:]
    score = pd.Series(np.tanh(np.cumsum(rng.normal(0, 0.3, len(test))) / 5), index=test)
    combined = pd.DataFrame({
        "composite_score": score,
        "score_technical": rng.uniform(-1, 1, len(test)),
        "score_macro": rng.uniform(-1, 1, len(test)),
        "score_sentiment": rng.uniform(0, 1, len(test)),
        "score_ml_classifier": rng.uniform(-1, 1, len(test)),
        "score_ml_regressor": rng.uniform(-1, 1, len(test)),
        "action": score.apply(lambda s: "STRONG BUY" if s > 0.35 else "BUY" if s > 0.2
                               else "STRONG SELL" if s < -0.35 else "SELL" if s < -0.2 else "HOLD"),
        "ml_confidence": rng.uniform(0.55, 0.8, len(test)),
        "vix": np.abs(rng.normal(20, 5, len(test))),
    }, index=test)

    res = run_backtest(ohlcv, combined)
    generate_dashboard(ohlcv, res, combined, walk_forward_result=None)

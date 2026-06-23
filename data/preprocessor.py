"""
data/preprocessor.py
====================
Turn the raw per-source frames from ``fetcher.py`` into one clean, aligned
master panel indexed by gold's trading calendar.

What this module does
---------------------
1. Renames every column to a clear, prefixed schema, e.g. ``gold_close``,
   ``dxy_close``, ``vix_close``, ``cpi``, ``real_yield``.
2. Aligns all series onto the **gold trading calendar** (gold is the master
   index — we only ever trade on days gold trades).
3. Forward-fills lower-frequency / differently-calendared series.
4. Adds a handful of basic derived columns (simple returns) that almost every
   downstream module needs.

The lookahead rule that matters here
-----------------------------------
Forward-filling macro data is **legitimate and necessary**, not leakage:
CPI is published monthly, so on any given trading day the most recent *known*
CPI print is genuinely the information you had. Forward-filling carries that
last-known value forward — which is exactly what a real trader sees.

What would be leakage (and we never do it):
  * back-filling (using a *future* value to fill a past gap),
  * forward-filling the GOLD PRICE itself onto days it didn't trade
    (we don't — gold defines the calendar, so it has no gaps to fill),
  * computing any feature from future rows (that happens — correctly lagged —
    in ``ml/features.py``, never here).

The full lagged ML feature matrix is built later in ``ml/features.py``. This
module deliberately stops at a clean, aligned, *contemporaneous* panel so the
separation of concerns stays auditable.
"""

from __future__ import annotations

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Series that are prices (have OHLCV) vs. single-value macro series.
# Used to decide how to rename/keep columns.
_PRICE_KEYS = set(config.YF_TICKERS.keys())
_MACRO_KEYS = set(config.FRED_SERIES.keys())


def _prefix_price_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rename a price frame's columns to ``<name>_<field>`` (e.g. gold_close)."""
    renamed = df.rename(columns={c: f"{name}_{c}" for c in df.columns})
    return renamed


def build_master_panel(
    raw: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Assemble the aligned master panel from raw per-source frames.

    Args:
        raw: Output of ``fetcher.fetch_all`` — a mapping of internal name to
            DataFrame. Missing sources are simply absent from the dict; this
            function tolerates any subset (graceful degradation).

    Returns:
        A DataFrame indexed by the primary asset's trading dates, with clearly
        prefixed columns for every available series and a few derived return
        columns. Rows before the primary asset's first valid close are dropped.

    Raises:
        RuntimeError: If the primary asset frame is missing (cannot proceed).
    """
    primary = config.PRIMARY_ASSET
    if primary not in raw:
        raise RuntimeError(
            f"Cannot build panel: primary asset '{primary}' missing from raw data."
        )

    # --- 1. Master calendar = the primary asset's index ----------------------
    master_index = raw[primary].index
    logger.info(
        "Building master panel on '%s' calendar: %d trading days (%s..%s).",
        primary, len(master_index),
        master_index.min().date(), master_index.max().date(),
    )

    frames: list[pd.DataFrame] = []

    # --- 2. Price series: prefix columns, as-of align onto master ------------
    for name in config.YF_TICKERS:
        if name not in raw:
            continue
        df = _prefix_price_columns(name, raw[name])
        if name == primary:
            df = df.reindex(master_index)          # the master itself — no fill
        else:
            # As-of forward fill: for each gold trading day, take the asset's
            # last known close on or before that day. Correctly handles days
            # where the other market was closed but gold traded (and vice versa).
            df = df.reindex(master_index, method="ffill")
        frames.append(df)

    # --- 3. Macro series: apply publication lag, then as-of forward-fill -----
    for name in config.FRED_SERIES:
        if name not in raw:
            continue
        s = raw[name].copy()
        lag_days = config.FRED_PUBLICATION_LAG_DAYS.get(name, 0)
        if lag_days:
            # Shift the index forward so each value only becomes visible on/after
            # its real-world release date — this is the anti-lookahead step.
            s.index = s.index + pd.Timedelta(days=lag_days)
            s = s[~s.index.duplicated(keep="last")].sort_index()
        df = s.reindex(master_index, method="ffill")
        frames.append(df)

    panel = pd.concat(frames, axis=1)

    # --- 4. Drop the warm-up period before gold has a valid close ------------
    close_col = f"{primary}_close"
    panel = panel.loc[panel[close_col].notna()].copy()

    # --- 5. Basic derived columns every module needs -------------------------
    panel = _add_basic_returns(panel)

    # --- 6. Report data quality ---------------------------------------------
    _log_quality_report(panel)

    return panel


def _add_basic_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add simple 1-day pct-change returns for each asset's close.

    These are *contemporaneous* returns (today's close vs yesterday's). Any
    feature fed to the ML model is lagged later in ``ml/features.py``; these
    raw return columns are convenience inputs, not model features as-is.
    """
    out = panel.copy()
    for name in config.YF_TICKERS:
        col = f"{name}_close"
        if col in out.columns:
            out[f"{name}_ret_1d"] = out[col].pct_change()
    return out


def _log_quality_report(panel: pd.DataFrame) -> None:
    """Log a concise data-quality summary (shape, span, worst NaN columns)."""
    n_rows, n_cols = panel.shape
    na_frac = panel.isna().mean().sort_values(ascending=False)
    worst = na_frac[na_frac > 0].head(5)
    logger.info("Master panel ready: %d rows x %d cols.", n_rows, n_cols)
    if not worst.empty:
        summary = ", ".join(f"{c}={f:.1%}" for c, f in worst.items())
        logger.info("Columns with most missing values: %s", summary)
    else:
        logger.info("No missing values remain in the master panel.")


def get_primary_ohlcv(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract a clean OHLCV frame for the primary asset from the panel.

    Convenience for the technical-signal and backtest modules, which want a
    plain ``open/high/low/close/volume`` frame rather than the wide panel.

    Args:
        panel: The master panel from ``build_master_panel``.

    Returns:
        DataFrame with columns ``open, high, low, close, volume`` (those that
        exist), indexed by date.
    """
    primary = config.PRIMARY_ASSET
    mapping = {
        f"{primary}_open": "open",
        f"{primary}_high": "high",
        f"{primary}_low": "low",
        f"{primary}_close": "close",
        f"{primary}_volume": "volume",
    }
    cols = {src: dst for src, dst in mapping.items() if src in panel.columns}
    ohlcv = panel[list(cols)].rename(columns=cols)
    return ohlcv


if __name__ == "__main__":
    # Manual smoke test:  python -m data.preprocessor
    from dotenv import load_dotenv

    import data.fetcher as fetcher

    load_dotenv()
    config.configure_logging()
    raw_data = fetcher.fetch_all()
    master = build_master_panel(raw_data)
    print(master.tail())
    print("\nColumns:", list(master.columns))

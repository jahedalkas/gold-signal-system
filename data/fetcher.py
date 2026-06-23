"""
data/fetcher.py
===============
Ingest all raw market and macro data the system needs, from free sources.

Sources
-------
* yfinance  -> gold, silver, DXY, VIX, S&P 500, oil, US 10Y yield, GLD ETF.
* FRED      -> CPI, 10Y real yield (TIPS), M2 money supply.

Design principles
-----------------
* **Graceful degradation.** Each series is fetched independently inside a
  try/except. If one source fails (network, rate limit, bad symbol), we log a
  WARNING and continue with whatever else succeeded. The pipeline never dies
  because one feed is down — it just runs with fewer signals.
* **Disk caching.** Every series is cached as Parquet under ``data/_cache``.
  On the next run within ``CACHE_MAX_AGE_HOURS`` we read the cache instead of
  re-hitting the API. This keeps repeated runs fast and polite to free APIs.
* **No lookahead here.** This module only *fetches* raw observations. It does
  not lag, shift, or engineer anything — that is the job of ``preprocessor.py``
  and ``ml/features.py``. Keeping ingestion "dumb" makes leakage easier to
  reason about later.

Returned data
-------------
``fetch_all`` returns a ``dict[str, pandas.DataFrame]`` keyed by the clean
internal names from ``config`` (e.g. ``"gold"``, ``"dxy"``, ``"cpi"``).
Price frames have OHLCV-style columns; FRED frames have a single value column.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


# =============================================================================
# Caching helpers
# =============================================================================
def _cache_path(name: str) -> Path:
    """Return the on-disk cache path for a named series."""
    return config.DATA_CACHE_DIR / f"{name}.parquet"


def _is_cache_fresh(path: Path, max_age_hours: float) -> bool:
    """Return True if ``path`` exists and is younger than ``max_age_hours``."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_hours * 3600.0


def _read_cache(name: str) -> pd.DataFrame | None:
    """Read a cached series if present and readable, else ``None``."""
    path = _cache_path(name)
    if not _is_cache_fresh(path, config.CACHE_MAX_AGE_HOURS):
        return None
    try:
        df = pd.read_parquet(path)
        logger.debug("Loaded '%s' from cache (%d rows).", name, len(df))
        return df
    except Exception as exc:  # corrupt cache -> ignore and re-fetch
        logger.warning("Could not read cache for '%s': %s", name, exc)
        return None


def _write_cache(name: str, df: pd.DataFrame) -> None:
    """Persist a series to the Parquet cache; failures are non-fatal."""
    try:
        df.to_parquet(_cache_path(name))
    except Exception as exc:
        logger.warning("Could not write cache for '%s': %s", name, exc)


# =============================================================================
# yfinance price fetching
# =============================================================================
def fetch_price_series(
    name: str,
    ticker: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch one OHLCV price series from Yahoo Finance.

    Args:
        name: Clean internal name (used for the cache file and logging).
        ticker: Yahoo Finance symbol, e.g. ``"GC=F"``.
        start: First date to request (inclusive).
        end: Last date to request (inclusive).
        use_cache: If True, read from / write to the local Parquet cache.

    Returns:
        A DataFrame indexed by date with columns ``open, high, low, close,
        volume`` (lower-cased, flattened), or ``None`` if the fetch failed.
    """
    if use_cache:
        cached = _read_cache(name)
        if cached is not None:
            return cached

    try:
        import yfinance as yf  # imported lazily so config import stays cheap

        logger.info("Fetching '%s' (%s) from yfinance...", name, ticker)
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            logger.warning("No data returned for '%s' (%s).", name, ticker)
            return None

        df = _normalise_yf_frame(raw)
        if use_cache:
            _write_cache(name, df)
        return df

    except Exception as exc:
        logger.warning("Failed to fetch '%s' (%s): %s", name, ticker, exc)
        return None


def _normalise_yf_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten and clean a raw yfinance frame to lower-case OHLCV columns.

    yfinance may return a single- or multi-level column index depending on
    version and the number of tickers. This collapses either form to a tidy
    ``open/high/low/close/volume`` schema with a tz-naive DatetimeIndex.
    """
    df = raw.copy()

    # Collapse a possible MultiIndex (e.g. ('Close', 'GC=F')) to the field name.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Keep a stable, predictable subset; "adj_close" is dropped (we use raw
    # close consistently across all assets to avoid mixing adjusted/unadjusted).
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    # Ensure a clean, sorted, tz-naive index and numeric dtypes.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# =============================================================================
# FRED macro fetching
# =============================================================================
def fetch_fred_series(
    name: str,
    series_id: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Fetch one macro series from FRED.

    Requires ``FRED_API_KEY`` in the environment. If the key is missing, this
    logs a WARNING and returns ``None`` (the system continues without it).

    Args:
        name: Clean internal name (cache file + logging).
        series_id: FRED series code, e.g. ``"CPIAUCSL"``.
        start: First date to request.
        end: Last date to request.
        use_cache: If True, use the local Parquet cache.

    Returns:
        A single-column DataFrame (column == ``name``) indexed by date, or
        ``None`` on failure / missing key.
    """
    if use_cache:
        cached = _read_cache(f"fred_{name}")
        if cached is not None:
            return cached

    api_key = config.get_fred_api_key()
    if not api_key:
        logger.warning(
            "FRED_API_KEY not set — skipping macro series '%s' (%s). "
            "Add it to your .env to enable macro signals.", name, series_id
        )
        return None

    try:
        from fredapi import Fred  # lazy import

        logger.info("Fetching FRED series '%s' (%s)...", name, series_id)
        fred = Fred(api_key=api_key)
        series = fred.get_series(
            series_id,
            observation_start=start.strftime("%Y-%m-%d"),
            observation_end=end.strftime("%Y-%m-%d"),
        )
        if series is None or len(series) == 0:
            logger.warning("FRED returned no data for '%s' (%s).", name, series_id)
            return None

        df = series.to_frame(name=name)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        if use_cache:
            _write_cache(f"fred_{name}", df)
        return df

    except Exception as exc:
        logger.warning("Failed to fetch FRED '%s' (%s): %s", name, series_id, exc)
        return None


# =============================================================================
# Orchestration
# =============================================================================
def fetch_all(
    history_years: int | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch every configured price and macro series.

    Iterates over ``config.YF_TICKERS`` and ``config.FRED_SERIES``, fetching
    each independently. Sources that fail are skipped with a WARNING; the
    function always returns whatever succeeded.

    Args:
        history_years: How many years back to fetch. Defaults to
            ``config.HISTORY_YEARS``.
        use_cache: If True, use the local Parquet cache where fresh.

    Returns:
        Mapping of internal name -> DataFrame. The primary asset
        (``config.PRIMARY_ASSET``) is guaranteed to be present, or a
        ``RuntimeError`` is raised — without gold prices the system cannot run.

    Raises:
        RuntimeError: If the primary asset (gold) could not be fetched.
    """
    years = history_years if history_years is not None else config.HISTORY_YEARS
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=int(years * 365.25) + 5)

    logger.info(
        "Fetching ~%d years of data: %s to %s.",
        years, start.date(), end.date(),
    )

    data: dict[str, pd.DataFrame] = {}

    # --- Price series --------------------------------------------------------
    for name, ticker in config.YF_TICKERS.items():
        df = fetch_price_series(name, ticker, start, end, use_cache=use_cache)
        if df is not None and not df.empty:
            data[name] = df
        else:
            logger.warning("Skipping price series '%s' — no data.", name)

    # --- Macro series --------------------------------------------------------
    for name, series_id in config.FRED_SERIES.items():
        df = fetch_fred_series(name, series_id, start, end, use_cache=use_cache)
        if df is not None and not df.empty:
            data[name] = df
        else:
            logger.warning("Skipping macro series '%s' — no data.", name)

    # --- Sanity gate ---------------------------------------------------------
    if config.PRIMARY_ASSET not in data:
        raise RuntimeError(
            f"Primary asset '{config.PRIMARY_ASSET}' could not be fetched. "
            "Check your internet connection and that the ticker "
            f"'{config.YF_TICKERS[config.PRIMARY_ASSET]}' is valid on Yahoo "
            "Finance. The system cannot continue without gold prices."
        )

    logger.info(
        "Fetch complete: %d/%d series available (%s).",
        len(data),
        len(config.YF_TICKERS) + len(config.FRED_SERIES),
        ", ".join(sorted(data)),
    )
    return data


if __name__ == "__main__":
    # Manual smoke test:  python -m data.fetcher
    from dotenv import load_dotenv

    load_dotenv()
    config.configure_logging()
    out = fetch_all()
    for k, v in out.items():
        print(f"{k:12s} rows={len(v):5d}  cols={list(v.columns)}  "
              f"range={v.index.min().date()}..{v.index.max().date()}")

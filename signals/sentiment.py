"""
signals/sentiment.py
===================
Sentiment signals for gold, with an important honesty caveat baked into the
design (read this — it affects how you interpret the backtest).

The free-data sentiment problem
-------------------------------
Your spec wants a daily news-sentiment feature across ~7 years of history.
**That cannot be built from free APIs.** NewsAPI's free tier only returns the
last ~30 days, and RSS feeds only expose current headlines. There is no free
way to reconstruct what the gold news flow said on, say, 2019-03-14. So:

  * **Backtestable sentiment** comes from the **VIX fear proxy**, which *is*
    available for the full history. Spikes in VIX reflect safe-haven demand,
    a genuine, historically-available sentiment-like driver for gold.
  * **Live news sentiment** (NewsAPI -> RSS fallback, scored by FinBERT ->
    TextBlob fallback) is computed for *today only* and is meant for the live
    signal / dashboard, NOT for the historical backtest.

A consequence worth stating plainly: because the historical proxy is
VIX-based, the backtest's "sentiment" component can flag *fear-driven
bullishness* but has little power to express *bearish* sentiment. Don't read
more into the sentiment backtest contribution than that.

Output contract
---------------
Same as the other signal modules: ``compute_sentiment_signals`` returns a
full-history frame with per-signal ``signal``/``strength`` columns and an
aggregate ``sentiment_score`` in [-1, +1]. ``live_news_sentiment`` returns a
single ``SignalResult`` for the current news flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalResult:
    """A single sentiment reading for one point in time."""
    name: str
    signal: int
    strength: float
    reason: str


# Curated finance/gold lexicon to compensate for TextBlob's blindness to
# market jargon (it scores "gold plunges" as neutral on its own).
_GOLD_BULLISH_TERMS: frozenset[str] = frozenset({
    "surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps",
    "climb", "climbs", "gain", "gains", "rise", "rises", "record high", "haven",
    "safe haven", "inflation hedge", "dovish", "rate cut", "rate cuts", "easing",
    "weaker dollar", "bullish", "demand", "uptrend", "breakout", "support",
})
_GOLD_BEARISH_TERMS: frozenset[str] = frozenset({
    "plunge", "plunges", "drop", "drops", "fall", "falls", "slump", "slumps",
    "tumble", "tumbles", "sink", "sinks", "slide", "slides", "decline", "declines",
    "selloff", "sell-off", "hawkish", "rate hike", "rate hikes", "tightening",
    "stronger dollar", "bearish", "downtrend", "pressure", "resistance", "outflows",
})


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
# Text scoring — FinBERT (optional) with TextBlob + lexicon fallback
# =============================================================================
_finbert_scorer = None  # module-level cache for the (heavy) FinBERT pipeline
_finbert_checked = False


def _get_finbert_scorer():
    """Lazily build a FinBERT scoring callable, or return ``None`` if unavailable.

    Returns a function ``str -> float in [-1, 1]`` if transformers + torch are
    installed and ``SENTIMENT_USE_FINBERT_IF_AVAILABLE`` is True; otherwise
    ``None`` (caller falls back to TextBlob). The pipeline is cached after the
    first successful load because model construction is expensive.
    """
    global _finbert_scorer, _finbert_checked
    if _finbert_checked:
        return _finbert_scorer
    _finbert_checked = True

    if not config.SENTIMENT_USE_FINBERT_IF_AVAILABLE:
        return None
    try:
        from transformers import pipeline  # heavy import (pulls torch)

        logger.info("Loading FinBERT sentiment model (first run may download)...")
        clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")

        def _score(text: str) -> float:
            res = clf(text[:512])[0]  # truncate to model max length
            label = res["label"].lower()
            conf = float(res["score"])
            if label == "positive":
                return conf
            if label == "negative":
                return -conf
            return 0.0

        _finbert_scorer = _score
        logger.info("FinBERT loaded.")
    except Exception as exc:
        logger.info("FinBERT unavailable (%s) — using TextBlob fallback.", type(exc).__name__)
        _finbert_scorer = None
    return _finbert_scorer


def _score_text_textblob(text: str) -> float:
    """Score one headline with TextBlob polarity plus a gold-lexicon overlay.

    TextBlob alone misses finance jargon, so we blend its general polarity with
    a count of bullish/bearish gold terms. Result is clipped to [-1, 1].
    """
    try:
        from textblob import TextBlob
        base = float(TextBlob(text).sentiment.polarity)
    except Exception:
        base = 0.0

    low = text.lower()
    bull = sum(1 for t in _GOLD_BULLISH_TERMS if t in low)
    bear = sum(1 for t in _GOLD_BEARISH_TERMS if t in low)
    lexicon = 0.0
    if bull or bear:
        lexicon = (bull - bear) / max(bull + bear, 1)

    # Weight the domain lexicon more heavily than TextBlob's generic read.
    score = 0.4 * base + 0.6 * lexicon
    return float(np.clip(score, -1.0, 1.0))


def score_headlines(headlines: list[str]) -> list[float]:
    """Score a list of headlines to polarities in [-1, 1].

    Uses FinBERT if available, otherwise the TextBlob + gold-lexicon fallback.

    Args:
        headlines: Raw headline strings.

    Returns:
        List of polarity scores aligned to ``headlines`` (empty if input empty).
    """
    if not headlines:
        return []
    finbert = _get_finbert_scorer()
    scorer = finbert if finbert is not None else _score_text_textblob
    return [scorer(h) for h in headlines]


# =============================================================================
# Live news fetching (current headlines only — NOT historical)
# =============================================================================
def fetch_news_headlines(max_headlines: int | None = None) -> list[tuple[datetime, str]]:
    """Fetch *current* gold headlines: NewsAPI first, RSS feeds as fallback.

    Gracefully degrades: if the NewsAPI key is missing or its quota is exceeded,
    falls back to RSS. If everything fails, returns an empty list and the
    sentiment layer simply contributes nothing live.

    Args:
        max_headlines: Cap on headlines returned. Defaults to
            ``config.SENTIMENT_MAX_HEADLINES``.

    Returns:
        List of ``(published_datetime, title)`` tuples, newest first.
    """
    cap = max_headlines or config.SENTIMENT_MAX_HEADLINES
    headlines = _fetch_newsapi(cap)
    if headlines:
        return headlines[:cap]
    logger.info("Falling back to RSS feeds for news headlines.")
    return _fetch_rss(cap)[:cap]


def _fetch_newsapi(cap: int) -> list[tuple[datetime, str]]:
    """Fetch headlines from NewsAPI's /v2/everything endpoint (free tier)."""
    api_key = config.get_newsapi_key()
    if not api_key:
        logger.info("NEWSAPI_KEY not set — skipping NewsAPI.")
        return []
    try:
        import requests

        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": config.SENTIMENT_QUERY,
                "language": config.SENTIMENT_LANGUAGE,
                "sortBy": "publishedAt",
                "pageSize": min(cap, 100),
                "apiKey": api_key,
            },
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("NewsAPI quota exceeded (429) — using RSS fallback.")
            return []
        resp.raise_for_status()
        payload = resp.json()
        out: list[tuple[datetime, str]] = []
        for art in payload.get("articles", []):
            title = (art.get("title") or "").strip()
            if not title:
                continue
            ts_raw = art.get("publishedAt", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)
            out.append((ts, title))
        logger.info("Fetched %d headlines from NewsAPI.", len(out))
        return out
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s — using RSS fallback.", exc)
        return []


def _fetch_rss(cap: int) -> list[tuple[datetime, str]]:
    """Fetch headlines from the configured RSS feeds via feedparser."""
    try:
        import feedparser
    except Exception as exc:
        logger.warning("feedparser unavailable: %s", exc)
        return []

    out: list[tuple[datetime, str]] = []
    for url in config.SENTIMENT_RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:cap]:
                title = getattr(entry, "title", "").strip()
                if not title:
                    continue
                if getattr(entry, "published_parsed", None):
                    ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)
                out.append((ts, title))
        except Exception as exc:
            logger.warning("RSS feed failed (%s): %s", url, exc)
    out.sort(key=lambda x: x[0], reverse=True)
    logger.info("Fetched %d headlines from RSS.", len(out))
    return out


def live_news_sentiment() -> SignalResult:
    """Compute a single sentiment reading from the current news flow.

    For live / dashboard use only. Returns neutral if no headlines are
    available (e.g. offline, no keys, all feeds down).

    Returns:
        A ``SignalResult`` summarising current gold news mood.
    """
    items = fetch_news_headlines()
    if not items:
        return SignalResult("News", 0, 0.0, "No live headlines available — neutral")

    titles = [t for _, t in items]
    scores = score_headlines(titles)
    avg = float(np.mean(scores)) if scores else 0.0
    signal = 1 if avg > 0.05 else (-1 if avg < -0.05 else 0)
    strength = _clip01(abs(avg))
    mood = {1: "bullish", -1: "bearish", 0: "neutral"}[signal]
    reason = (f"News mood {mood} from {len(titles)} headlines "
              f"(avg polarity {avg:+.2f})")
    return SignalResult("News", signal, round(strength, 3), reason)


# =============================================================================
# Backtestable sentiment: VIX fear proxy
# =============================================================================
def vix_fear_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """VIX-based safe-haven demand: elevated/spiking VIX is bullish gold.

    This is the historically-available sentiment proxy used in the backtest.
    Two components combine: (a) the VIX *level* above the fear threshold, and
    (b) a VIX *spike* (5-day jump) capturing acute fear events.
    """
    out = _empty_history(panel.index)
    if "vix_close" not in panel.columns:
        logger.warning("Sentiment VIX proxy skipped — 'vix_close' missing.")
        return out

    vix = panel["vix_close"]
    level_excess = (vix - config.VIX_FEAR_THRESHOLD) / 25.0
    spike = vix.pct_change(5)  # acute jumps in fear

    fear = (vix > config.VIX_FEAR_THRESHOLD) | (spike > 0.20)
    out.loc[fear, "signal"] = 1
    out.loc[fear, "strength"] = _clip01(
        np.maximum(level_excess[fear].fillna(0.0), spike[fear].fillna(0.0))
    )
    return out


def _proxy_sentiment_score(panel: pd.DataFrame) -> pd.Series:
    """Continuous [-1,1] sentiment proxy from VIX (one-sided toward bullish).

    Low VIX maps to ~0 (neutral), high VIX toward +1 (safe-haven bullish). It
    does not go meaningfully negative — see the module docstring caveat.
    """
    if "vix_close" not in panel.columns:
        return pd.Series(0.0, index=panel.index)
    vix = panel["vix_close"]
    score = (vix - config.VIX_FEAR_THRESHOLD) / 25.0
    return _clip01(score).clip(-1.0, 1.0)


def sentiment_divergence_signal(
    panel: pd.DataFrame, sentiment: pd.Series
) -> pd.DataFrame:
    """Reversal flag: gold price falling while the sentiment proxy improves.

    Args:
        panel: Master panel (needs ``gold_close``).
        sentiment: A sentiment proxy series aligned to ``panel.index``.

    Returns:
        Full-history signal frame; +1 where price is down over 5 days but
        sentiment is rising (potential bullish reversal).
    """
    out = _empty_history(panel.index)
    if "gold_close" not in panel.columns:
        return out
    price_5d = panel["gold_close"].pct_change(5)
    sent_change = sentiment.diff(5)
    diverge = (price_5d < 0) & (sent_change > 0)
    out.loc[diverge, "signal"] = 1
    out.loc[diverge, "strength"] = _clip01(sent_change[diverge].fillna(0.0) * 2.0)
    return out


# =============================================================================
# Aggregation
# =============================================================================
def compute_sentiment_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Run all backtestable sentiment signals over the full history.

    Note: this uses the VIX-based proxy and 7-day momentum — *not* historical
    news (which free APIs cannot provide). Live news is handled separately by
    ``live_news_sentiment`` for the current bar.

    Args:
        panel: The aligned master panel.

    Returns:
        DataFrame indexed like ``panel`` with per-signal columns plus an
        aggregate ``sentiment_score`` in [-1, +1] and a ``sentiment_momentum``
        column (7-day change in the proxy).
    """
    proxy = _proxy_sentiment_score(panel)
    vix_fear = vix_fear_signal(panel).rename(
        columns={"signal": "VIX_fear_signal", "strength": "VIX_fear_strength"}
    )
    divergence = sentiment_divergence_signal(panel, proxy).rename(
        columns={"signal": "Divergence_signal", "strength": "Divergence_strength"}
    )

    out = pd.concat([vix_fear, divergence], axis=1)
    out["sentiment_momentum"] = proxy.diff(config.SENTIMENT_ROLLING_DAYS).fillna(0.0)

    # Aggregate: strength-weighted mean of the component signals, in [-1,1].
    weighted = (
        vix_fear["VIX_fear_signal"] * vix_fear["VIX_fear_strength"]
        + divergence["Divergence_signal"] * divergence["Divergence_strength"]
    )
    strength_sum = (
        vix_fear["VIX_fear_strength"] + divergence["Divergence_strength"]
    ).replace(0, np.nan)
    out["sentiment_score"] = (weighted / strength_sum).fillna(0.0)
    return out


def latest_reasons(panel: pd.DataFrame, include_live_news: bool = True) -> dict[str, SignalResult]:
    """Build human-readable reasons for the latest bar (and live news if asked).

    Args:
        panel: Master panel.
        include_live_news: If True, also fetch and score current news. Set False
            for fast offline runs.

    Returns:
        Mapping of component name -> ``SignalResult`` for the most recent bar.
    """
    results: dict[str, SignalResult] = {}

    vf = vix_fear_signal(panel)
    sig = int(vf["signal"].iloc[-1])
    strength = float(vf["strength"].iloc[-1])
    vix_now = float(panel["vix_close"].iloc[-1]) if "vix_close" in panel.columns else float("nan")
    mood = {1: "bullish", -1: "bearish", 0: "neutral"}[sig]
    results["VIX_fear"] = SignalResult(
        "VIX_fear", sig, round(strength, 3),
        f"VIX {vix_now:.1f} -> safe-haven sentiment {mood} (strength {strength:.2f})",
    )

    if include_live_news:
        results["News"] = live_news_sentiment()

    return results


if __name__ == "__main__":
    # Manual smoke test:  python -m signals.sentiment
    config.configure_logging()

    # 1) Text scoring (offline, TextBlob + lexicon).
    print("--- headline scoring (TextBlob + gold lexicon) ---")
    samples = [
        "Gold surges to record high on safe-haven demand",
        "Gold plunges as dollar strengthens and yields climb",
        "Gold steady ahead of Fed decision",
    ]
    for h, s in zip(samples, score_headlines(samples)):
        print(f"  {s:+.2f} | {h}")

    # 2) VIX proxy on synthetic panel.
    print("\n--- VIX fear proxy on synthetic panel ---")
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n)
    vix = np.abs(np.cumsum(rng.normal(0, 1.5, n))) + 14
    vix[150:160] += 20  # simulate a fear spike
    panel = pd.DataFrame(
        {"gold_close": 1800 + np.cumsum(rng.normal(0, 10, n)), "vix_close": vix},
        index=idx,
    )
    sig = compute_sentiment_signals(panel)
    print(sig[["sentiment_score", "sentiment_momentum"]].iloc[148:162].round(3))

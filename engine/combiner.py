"""
engine/combiner.py
=================
Blend the four signal sources into a single composite score in [-1, +1] and map
it to a recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL).

Components (each already in [-1, +1] except ML, which we map):
  * technical_score   — from signals/technical.py
  * macro_score       — from signals/macro.py
  * sentiment_score   — from signals/sentiment.py
  * ml_classifier     — mapped from P(up):   (prob_up - 0.5) * 2
  * ml_regressor      — mapped from expected return, saturating at ±SCALE

Dynamic re-weighting (your spec's rules), applied per day:
  1. **ML abstain:** if classifier confidence < ``ML_MIN_CONFIDENCE``, the ML
     contribution is set to zero and the remaining weights are renormalised.
  2. **Crisis:** if VIX > ``COMBINER_CRISIS_VIX``, macro weight is raised to
     ``COMBINER_CRISIS_MACRO_WEIGHT`` (fundamentals dominate), others shrink.
  3. **ML boost:** if the *realised* out-of-sample AUC over the trailing window
     exceeds ``COMBINER_ML_AUC_BOOST_THRESHOLD``, total ML weight is raised to
     ``COMBINER_ML_BOOSTED_WEIGHT``.

No lookahead in the AUC gate
----------------------------
The AUC boost is the one rule that could peek at the future, so it is computed
carefully: at day ``t`` we score AUC only on past predictions whose 5-day
outcome was already known (prediction date <= t - horizon). The gate therefore
uses nothing from the future. It also requires a minimum sample and both
classes present, otherwise it stays off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    """A single combined recommendation for one date."""
    date: pd.Timestamp
    score: float
    action: str
    components: dict[str, float]
    weights: dict[str, float]


# =============================================================================
# Score mapping helpers
# =============================================================================
def _ml_classifier_score(prob_up: pd.Series) -> pd.Series:
    """Map P(up) in [0,1] to a directional score in [-1,+1]."""
    return ((prob_up - 0.5) * 2.0).clip(-1.0, 1.0)


def _ml_regressor_score(pred_return: pd.Series) -> pd.Series:
    """Map an expected forward return to [-1,+1], saturating at ±SCALE."""
    return (pred_return / config.ML_REGRESSOR_RETURN_SCALE).clip(-1.0, 1.0)


def _ml_confidence(prob_up: pd.Series) -> pd.Series:
    """Confidence = probability of the predicted class, in [0.5, 1.0]."""
    return pd.concat([prob_up, 1.0 - prob_up], axis=1).max(axis=1)


def action_for_score(score: float) -> str:
    """Map a composite score to a recommendation label using config thresholds."""
    if score > config.STRONG_BUY_THRESHOLD:
        return "STRONG BUY"
    if score > config.BUY_THRESHOLD:
        return "BUY"
    if score < config.STRONG_SELL_THRESHOLD:
        return "STRONG SELL"
    if score < config.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


# =============================================================================
# Leak-free realised-AUC gate
# =============================================================================
def _rolling_realised_auc_flag(
    prob_up: pd.Series, y_true: pd.Series, index: pd.DatetimeIndex
) -> pd.Series:
    """Return a per-date boolean: is trailing realised OOS AUC above threshold?

    Only predictions whose outcome was known before each date contribute, so the
    flag never uses future information.

    Args:
        prob_up: OOS P(up) indexed by prediction date.
        y_true: Realised binary outcomes aligned to ``prob_up``.
        index: Dates to produce the flag for.

    Returns:
        Boolean Series aligned to ``index`` (True = boost ML this day).
    """
    flag = pd.Series(False, index=index)
    if prob_up.empty:
        return flag
    horizon = config.ML_PREDICTION_HORIZON
    window = config.COMBINER_AUC_WINDOW
    pred_dates = prob_up.index

    for t in index:
        # Outcomes are only known `horizon` trading days after prediction.
        known_cutoff_pos = pred_dates.get_indexer([t])[0] - horizon
        if known_cutoff_pos < window:  # not enough settled history yet
            continue
        lo = max(0, known_cutoff_pos - window)
        sl = slice(lo, known_cutoff_pos)
        p, y = prob_up.iloc[sl], y_true.iloc[sl]
        if y.nunique() < 2:
            continue
        try:
            if roc_auc_score(y, p) > config.COMBINER_ML_AUC_BOOST_THRESHOLD:
                flag.loc[t] = True
        except ValueError:
            continue
    return flag


# =============================================================================
# Per-day weight adjustment
# =============================================================================
def _adjust_weights(vix: float, ml_conf: float, ml_boost: bool) -> dict[str, float]:
    """Compute the per-day weight vector after applying the dynamic rules.

    Order of application: start from base weights, optionally boost ML, optionally
    force crisis (macro-dominant), optionally zero ML on abstain — then renormalise
    so the active weights sum to 1.

    Args:
        vix: VIX level for the day (NaN tolerated -> treated as non-crisis).
        ml_conf: Classifier confidence for the day (NaN -> treated as abstain).
        ml_boost: Whether the realised-AUC gate is active for the day.

    Returns:
        Dict of weights for the five components summing to 1.0.
    """
    w = dict(config.SIGNAL_WEIGHTS)

    # 3. ML boost: scale total ML weight up to the boosted target.
    if ml_boost:
        ml_total = w["ml_classifier"] + w["ml_regressor"]
        target = config.COMBINER_ML_BOOSTED_WEIGHT
        if ml_total > 0:
            ratio = target / ml_total
            w["ml_classifier"] *= ratio
            w["ml_regressor"] *= ratio

    # 2. Crisis: macro dominates.
    if pd.notna(vix) and vix > config.COMBINER_CRISIS_VIX:
        w["macro"] = config.COMBINER_CRISIS_MACRO_WEIGHT

    # 1. ML abstain: zero ML contribution when not confident enough.
    if pd.isna(ml_conf) or ml_conf < config.ML_MIN_CONFIDENCE:
        w["ml_classifier"] = 0.0
        w["ml_regressor"] = 0.0

    total = sum(w.values())
    if total <= 0:
        return {k: 0.0 for k in w}
    return {k: v / total for k, v in w.items()}


# =============================================================================
# Main entry point
# =============================================================================
def combine_signals(
    technical_score: pd.Series,
    macro_score: pd.Series,
    sentiment_score: pd.Series,
    ml_prob_up: pd.Series,
    ml_pred_return: pd.Series,
    panel: pd.DataFrame,
    ml_y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Combine all signals into a per-day composite score and recommendation.

    The output index is restricted to dates where the ML out-of-sample
    predictions exist (intersected with the available rule-based signals), since
    the backtest can only act where legitimate OOS ML output is present.

    Args:
        technical_score: Full-history technical aggregate.
        macro_score: Full-history macro aggregate.
        sentiment_score: Full-history sentiment aggregate.
        ml_prob_up: OOS classifier P(up), indexed by prediction date.
        ml_pred_return: OOS regressor expected return, same index.
        panel: Master panel (used for ``vix_close`` crisis detection).
        ml_y_true: Realised binary outcomes for the AUC gate (optional). If
            omitted, the ML-boost rule is disabled.

    Returns:
        DataFrame indexed by date with columns: ``composite_score``, the five
        component scores, the five applied weights, ``ml_confidence``, ``vix``,
        and ``action``.
    """
    # Align on the dates the ML predictions cover.
    idx = ml_prob_up.index
    idx = idx.intersection(technical_score.index).intersection(macro_score.index)
    idx = idx.intersection(sentiment_score.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates between ML predictions and signals.")

    tech = technical_score.reindex(idx).fillna(0.0)
    macro = macro_score.reindex(idx).fillna(0.0)
    senti = sentiment_score.reindex(idx).fillna(0.0)
    clf_score = _ml_classifier_score(ml_prob_up.reindex(idx))
    reg_score = _ml_regressor_score(ml_pred_return.reindex(idx))
    ml_conf = _ml_confidence(ml_prob_up.reindex(idx))
    vix = (panel["vix_close"].reindex(idx) if "vix_close" in panel.columns
           else pd.Series(np.nan, index=idx))

    # Leak-free ML-boost flag (only if outcomes were supplied).
    if ml_y_true is not None:
        boost = _rolling_realised_auc_flag(ml_prob_up, ml_y_true.reindex(ml_prob_up.index), idx)
    else:
        boost = pd.Series(False, index=idx)

    rows: list[dict] = []
    for t in idx:
        weights = _adjust_weights(vix.loc[t], ml_conf.loc[t], bool(boost.loc[t]))
        comps = {
            "technical": float(tech.loc[t]),
            "macro": float(macro.loc[t]),
            "sentiment": float(senti.loc[t]),
            "ml_classifier": float(clf_score.loc[t]),
            "ml_regressor": float(reg_score.loc[t]),
        }
        score = float(np.clip(sum(weights[k] * comps[k] for k in comps), -1.0, 1.0))
        rows.append({
            "composite_score": score,
            **{f"score_{k}": comps[k] for k in comps},
            **{f"w_{k}": weights[k] for k in weights},
            "ml_confidence": float(ml_conf.loc[t]),
            "vix": float(vix.loc[t]) if pd.notna(vix.loc[t]) else np.nan,
            "action": action_for_score(score),
        })

    out = pd.DataFrame(rows, index=idx)
    logger.info(
        "Combined signals over %d days (%s..%s). Action mix: %s",
        len(out), idx.min().date(), idx.max().date(),
        out["action"].value_counts().to_dict(),
    )
    return out


def latest_recommendation(combined: pd.DataFrame) -> Recommendation:
    """Build a ``Recommendation`` for the most recent combined date."""
    t = combined.index[-1]
    row = combined.loc[t]
    comps = {k.replace("score_", ""): float(row[k]) for k in combined.columns
             if k.startswith("score_")}
    weights = {k.replace("w_", ""): float(row[k]) for k in combined.columns
               if k.startswith("w_")}
    return Recommendation(
        date=t, score=float(row["composite_score"]), action=str(row["action"]),
        components=comps, weights=weights,
    )


if __name__ == "__main__":
    # Smoke test on synthetic series:  python -m engine.combiner
    config.configure_logging()
    rng = np.random.default_rng(4)
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n)
    tech = pd.Series(rng.uniform(-1, 1, n), index=idx)
    macro = pd.Series(rng.uniform(-1, 1, n), index=idx)
    senti = pd.Series(rng.uniform(0, 1, n), index=idx)
    prob = pd.Series(rng.uniform(0.3, 0.8, n), index=idx)
    pred_ret = pd.Series(rng.normal(0, 0.02, n), index=idx)
    y_true = (pred_ret.shift(-5) > 0).astype(int)
    panel = pd.DataFrame({"vix_close": np.abs(rng.normal(20, 6, n))}, index=idx)

    combined = combine_signals(tech, macro, senti, prob, pred_ret, panel, ml_y_true=y_true)
    print(combined[["composite_score", "action", "ml_confidence", "vix"]].tail())
    rec = latest_recommendation(combined)
    print(f"\nLatest: {rec.action} (score {rec.score:+.3f})")
    print(f"  components: { {k: round(v,2) for k,v in rec.components.items()} }")
    print(f"  weights:    { {k: round(v,2) for k,v in rec.weights.items()} }")

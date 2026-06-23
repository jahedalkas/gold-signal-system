"""
ml/predictor.py
==============
Load the trained models and generate the current prediction for gold.

Output (a ``Prediction``):
    direction        : "UP" or "DOWN" over the horizon
    confidence       : probability of the predicted class (0.5..1.0)
    prob_up          : raw P(up) from the classifier
    expected_return  : regressor's expected H-day forward return (fraction)
    abstain          : True if confidence < ``config.ML_MIN_CONFIDENCE``
    top_features     : the 3 features that most drove *this* prediction, with
                       their signed SHAP contributions

The abstain flag implements your spec's "low confidence = sit out" rule: when
the classifier is not confident enough, the combiner should treat the ML
contribution as zero rather than forcing a weak call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

import config
from ml import features as features_mod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """A single model prediction for one date."""
    date: pd.Timestamp
    direction: str
    confidence: float
    prob_up: float
    expected_return: float
    abstain: bool
    top_features: list[tuple[str, float]]

    def describe(self) -> str:
        """Return a human-readable one-block summary of the prediction."""
        drivers = ", ".join(f"{name} ({val:+.3f})" for name, val in self.top_features)
        flag = " [ABSTAIN — low confidence]" if self.abstain else ""
        return (
            f"{self.direction} with {self.confidence:.0%} confidence{flag}\n"
            f"  Expected {config.ML_PREDICTION_HORIZON}-day return: "
            f"{self.expected_return:+.2%}\n"
            f"  Top drivers (SHAP): {drivers}"
        )


def load_models(path=None) -> dict:
    """Load the persisted model bundle saved by ``trainer.walk_forward``.

    Args:
        path: Optional override path. Defaults to ``ml/_models/xgb_models.joblib``.

    Returns:
        The model bundle dict (classifier, regressor, feature_names, params...).

    Raises:
        FileNotFoundError: If no saved model bundle exists yet.
    """
    p = path or (config.MODELS_DIR / "xgb_models.joblib")
    if not p.exists():
        raise FileNotFoundError(
            f"No trained models found at {p}. Run the trainer first "
            "(it is invoked automatically by main.py)."
        )
    bundle = joblib.load(p)
    logger.info("Loaded models trained with horizon=%dd, lag=%dd, %d features.",
                bundle["horizon"], bundle["lag"], len(bundle["feature_names"]))
    return bundle


def predict_latest(panel: pd.DataFrame, bundle: dict | None = None) -> Prediction:
    """Generate the prediction for the most recent available date.

    Args:
        panel: Aligned master panel (its newest rows supply the live features).
        bundle: Optional pre-loaded model bundle; loaded from disk if omitted.

    Returns:
        A ``Prediction`` for the latest tradeable date.

    Raises:
        RuntimeError: If no complete feature row is available.
    """
    if bundle is None:
        bundle = load_models()

    feature_names = bundle["feature_names"]
    live = features_mod.build_live_features(panel, lag=bundle["lag"])
    if live.empty:
        raise RuntimeError("No complete feature row available for prediction.")

    # Align columns to the training order; missing cols (e.g. a source dropped
    # out today) are filled with 0.0 so prediction still proceeds.
    row = live.iloc[[-1]].reindex(columns=feature_names, fill_value=0.0)
    date = live.index[-1]

    clf = bundle["classifier"]
    reg = bundle["regressor"]
    prob_up = float(clf.predict_proba(row)[0, 1])
    expected_return = float(reg.predict(row)[0])

    direction = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = prob_up if prob_up >= 0.5 else (1.0 - prob_up)
    abstain = confidence < config.ML_MIN_CONFIDENCE
    top_features = _instance_drivers(clf, row, feature_names)

    pred = Prediction(
        date=date, direction=direction, confidence=round(confidence, 4),
        prob_up=round(prob_up, 4), expected_return=expected_return,
        abstain=abstain, top_features=top_features,
    )
    logger.info("Prediction for %s: %s (conf %.0f%%, exp %.2f%%)%s",
                date.date(), direction, confidence * 100, expected_return * 100,
                " [abstain]" if abstain else "")
    return pred


def _instance_drivers(model, row: pd.DataFrame, feature_names: list[str],
                      k: int = 3) -> list[tuple[str, float]]:
    """Return the top-``k`` features driving this single prediction via SHAP.

    Falls back to global gain importance (unsigned) if SHAP is unavailable.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(row)
        if isinstance(vals, list):
            vals = vals[-1]
        contrib = np.asarray(vals).reshape(-1)
        order = np.argsort(np.abs(contrib))[::-1][:k]
        return [(feature_names[i], float(contrib[i])) for i in order]
    except Exception as exc:
        logger.warning("Per-instance SHAP unavailable (%s) — using gain importance.",
                       type(exc).__name__)
        imp = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
        order = np.argsort(imp)[::-1][:k]
        return [(feature_names[i], float(imp[i])) for i in order]


if __name__ == "__main__":
    # Smoke test: train on synthetic data then predict the latest bar.
    from ml.features import build_feature_matrix
    from ml.trainer import walk_forward

    config.configure_logging()
    rng = np.random.default_rng(9)
    n = 1500
    idx = pd.bdate_range("2019-01-01", periods=n)

    def walk(s, v):
        return s + np.cumsum(rng.normal(0, v, n))

    base = walk(1500, 10)
    dxy = walk(95, 0.35)
    panel = pd.DataFrame({
        "gold_open": base, "gold_high": base + 5, "gold_low": base - 5,
        "gold_close": base - 0.4 * (dxy - 95),
        "gold_volume": rng.integers(1e4, 1e5, n).astype(float),
        "silver_close": walk(20, 0.25), "dxy_close": dxy,
        "vix_close": np.abs(walk(18, 1.0)) + 12, "spx_close": walk(3000, 30),
        "oil_close": walk(60, 1.2), "tnx_close": np.abs(walk(25, 0.4)),
        "real_yield": walk(0.5, 0.025), "gld_volume": rng.integers(5e6, 2e7, n).astype(float),
    }, index=idx)

    fm = build_feature_matrix(panel)
    result = walk_forward(fm)
    bundle = {
        "classifier": result.final_classifier, "regressor": result.final_regressor,
        "feature_names": result.feature_names, "horizon": config.ML_PREDICTION_HORIZON,
        "lag": config.ML_FEATURE_LAG,
    }
    pred = predict_latest(panel, bundle=bundle)
    print("\n--- Latest prediction ---")
    print(pred.describe())

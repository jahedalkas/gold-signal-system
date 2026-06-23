"""
ml/evaluator.py
==============
Evaluate the walk-forward model's out-of-sample predictions honestly.

The two metrics that matter most for a financial classifier
----------------------------------------------------------
* **AUC-ROC** — can the model rank up-days above down-days better than chance?
  0.50 is a coin flip; on noisy daily gold, anything durably above ~0.55
  out-of-sample is genuinely notable.
* **Information Coefficient (IC)** — the Spearman rank correlation between
  predicted and realised forward returns. This is the industry-standard measure
  of predictive signal. Real-world equity ICs of 0.03–0.05 are considered good;
  do not expect more, and be suspicious of much higher (it usually means
  leakage).

Accuracy alone is misleading when classes are imbalanced, so we always report
it next to a buy-and-hold baseline (the naive "always predict up" rate) and a
random baseline (0.50). Beating accuracy by predicting the majority class is
not skill.

Charts saved to ``reports/charts``: ROC curve, confusion matrix, calibration
curve, predicted-vs-actual scatter, and the SHAP feature-importance bar.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless backend — render to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

import config  # noqa: E402
from ml.trainer import WalkForwardResult  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate(result: WalkForwardResult, save_charts: bool = True) -> dict:
    """Compute out-of-sample ML metrics and (optionally) save diagnostic charts.

    Args:
        result: The walk-forward result from ``trainer.walk_forward``.
        save_charts: If True, write diagnostic PNGs to ``reports/charts``.

    Returns:
        A dict of headline metrics (accuracy, precision, recall, f1, auc, ic,
        directional accuracy, and the two baselines).
    """
    oos = result.oos_predictions
    y_true = oos["y_true_binary"].astype(int)
    prob_up = oos["prob_up"]
    y_pred = (prob_up > 0.5).astype(int)

    metrics = {
        "n_predictions": int(len(oos)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, prob_up),
        "information_coefficient": float(
            oos["pred_return"].corr(oos["y_true_return"], method="spearman")
        ),
        "directional_accuracy": float(
            (np.sign(oos["pred_return"]) == np.sign(oos["y_true_return"])).mean()
        ),
        "buy_and_hold_baseline": float(y_true.mean()),  # "always up" accuracy
        "random_baseline": 0.5,
    }

    _log_summary(metrics)

    if save_charts:
        _plot_roc(y_true, prob_up)
        _plot_confusion(y_true, y_pred)
        _plot_calibration(y_true, prob_up)
        _plot_pred_vs_actual(oos)
        _plot_shap_importance(result.feature_importance)
        logger.info("Saved ML evaluation charts to %s", config.CHARTS_DIR)

    return metrics


def _safe_auc(y_true: pd.Series, prob: pd.Series) -> float:
    """ROC-AUC that returns NaN if only one class is present."""
    try:
        return float(roc_auc_score(y_true, prob)) if y_true.nunique() > 1 else float("nan")
    except ValueError:
        return float("nan")


def _log_summary(m: dict) -> None:
    """Log a compact, honest metrics summary with baseline comparisons."""
    edge = m["accuracy"] - m["buy_and_hold_baseline"]
    logger.info(
        "ML OOS evaluation (n=%d):\n"
        "  Accuracy=%.3f | Buy&Hold(always-up)=%.3f | edge vs B&H=%+.3f\n"
        "  AUC=%.3f (0.50=chance) | IC=%+.3f (0.03-0.05 is good)\n"
        "  Precision=%.3f | Recall=%.3f | F1=%.3f | Directional acc=%.3f",
        m["n_predictions"], m["accuracy"], m["buy_and_hold_baseline"], edge,
        m["auc"], m["information_coefficient"],
        m["precision"], m["recall"], m["f1"], m["directional_accuracy"],
    )
    if not np.isnan(m["auc"]) and m["auc"] < 0.52:
        logger.warning(
            "AUC is at/near chance (%.3f). Treat this model as having no reliable "
            "directional edge — do not over-weight it in the combiner.", m["auc"]
        )


# =============================================================================
# Charts
# =============================================================================
def _save(fig, name: str) -> None:
    """Save and close a figure under the charts directory."""
    fig.tight_layout()
    fig.savefig(config.CHARTS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(y_true: pd.Series, prob: pd.Series) -> None:
    """ROC curve with the AUC annotated."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if y_true.nunique() > 1:
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax.plot(fpr, tpr, label=f"Model (AUC={roc_auc_score(y_true, prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance (0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — out-of-sample direction")
    ax.legend(loc="lower right")
    _save(fig, "ml_roc_curve.png")


def _plot_confusion(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred Down", "Pred Up"])
    ax.set_yticks([0, 1], ["True Down", "True Up"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "ml_confusion_matrix.png")


def _plot_calibration(y_true: pd.Series, prob: pd.Series) -> None:
    """Calibration curve — is a stated 70% confidence right ~70% of the time?"""
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label="Model")
    except Exception as exc:
        logger.warning("Calibration curve skipped: %s", exc)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted P(up)")
    ax.set_ylabel("Observed fraction up")
    ax.set_title("Calibration curve")
    ax.legend(loc="upper left")
    _save(fig, "ml_calibration_curve.png")


def _plot_pred_vs_actual(oos: pd.DataFrame) -> None:
    """Scatter of predicted vs realised forward return, with IC in the title."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(oos["pred_return"], oos["y_true_return"], s=8, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ic = oos["pred_return"].corr(oos["y_true_return"], method="spearman")
    ax.set_xlabel("Predicted forward return")
    ax.set_ylabel("Actual forward return")
    ax.set_title(f"Predicted vs actual (IC={ic:+.3f})")
    _save(fig, "ml_pred_vs_actual.png")


def _plot_shap_importance(importance: pd.Series, top_n: int = 15) -> None:
    """Horizontal bar chart of the top mean-|SHAP| features."""
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#3b7dd8")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature importance (top {len(top)})")
    _save(fig, "ml_shap_importance.png")


if __name__ == "__main__":
    # Smoke test: train on synthetic data, then evaluate.
    from ml.features import build_feature_matrix
    from ml.trainer import walk_forward

    config.configure_logging()
    rng = np.random.default_rng(13)
    n = 1600
    idx = pd.bdate_range("2018-06-01", periods=n)

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
    metrics = evaluate(result, save_charts=True)
    print("\n--- Metrics dict ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

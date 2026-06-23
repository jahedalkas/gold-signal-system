"""
ml/trainer.py
=============
Train and walk-forward-validate the two XGBoost models:

  * **Classifier** — will gold be higher in ``H`` days? (binary direction)
  * **Regressor**  — what is the expected ``H``-day forward return? (magnitude)

Walk-forward, not random split (this is non-negotiable for time series)
----------------------------------------------------------------------
A random train/test split leaks the future into the past and produces
gorgeous, meaningless backtests. Instead we:

  1. Tune hyperparameters ONCE on the initial training window (all past data)
     using ``RandomizedSearchCV`` with ``TimeSeriesSplit`` — folds always train
     on earlier data and validate on later data.
  2. Step forward through the out-of-sample period in
     ``ML_RETRAIN_MONTHS``-month windows. At each step we **refit** the models
     (with the tuned params) on all data up to that point — an expanding window
     that simulates periodically retraining a live system — and predict the
     next window. Those predictions are stored as genuine out-of-sample output.

Tuning once and refitting (rather than re-searching every step) is the main
reason this runs in seconds on a laptop. Tuning uses only initial-window data,
so it introduces no leakage.

What you get back
-----------------
A ``WalkForwardResult`` with: the stitched out-of-sample predictions, per-window
metrics (accuracy / AUC / directional accuracy), the final models refit on all
data (for live prediction), the chosen hyperparameters, and SHAP-based feature
importances. The final models are also persisted to ``ml/_models``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

import config
from ml.features import FeatureMatrix

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Outputs of the walk-forward training run.

    Attributes:
        oos_predictions: DataFrame indexed by date with columns
            ``prob_up, pred_return, y_true_binary, y_true_return`` — the
            stitched out-of-sample predictions across all walk-forward windows.
        window_metrics: One row per walk-forward window (train/test sizes and
            accuracy / AUC / directional accuracy).
        final_classifier: Classifier refit on ALL available data (for live use).
        final_regressor: Regressor refit on ALL available data (for live use).
        best_clf_params: Tuned classifier hyperparameters.
        best_reg_params: Tuned regressor hyperparameters.
        feature_importance: Mean |SHAP| per feature (descending), if available.
        feature_names: Feature column order used by the models.
    """
    oos_predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    final_classifier: XGBClassifier
    final_regressor: XGBRegressor
    best_clf_params: dict
    best_reg_params: dict
    feature_importance: pd.Series
    feature_names: list[str] = field(default_factory=list)


# =============================================================================
# Model factories
# =============================================================================
def _make_classifier(params: dict, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Construct an XGBoost classifier with laptop-friendly, reproducible settings."""
    return XGBClassifier(
        tree_method="hist",
        n_jobs=-1,
        random_state=config.ML_RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def _make_regressor(params: dict) -> XGBRegressor:
    """Construct an XGBoost regressor with laptop-friendly, reproducible settings."""
    return XGBRegressor(
        tree_method="hist",
        n_jobs=-1,
        random_state=config.ML_RANDOM_STATE,
        eval_metric="rmse",
        **params,
    )


def _pos_weight(y: pd.Series) -> float:
    """Class-imbalance weight = (#negatives / #positives), clamped to be finite."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


# =============================================================================
# One-time hyperparameter search (on the initial training window only)
# =============================================================================
def _tune(estimator, X: pd.DataFrame, y: pd.Series, scoring: str) -> dict:
    """Randomised hyperparameter search with time-series CV; returns best params.

    Args:
        estimator: An unfitted XGBoost estimator to clone during search.
        X: Initial-window features (past data only — no leakage).
        y: Matching target.
        scoring: sklearn scoring string (e.g. ``"roc_auc"``).

    Returns:
        The best hyperparameter dict found.
    """
    n_splits = min(config.ML_CV_SPLITS, max(2, len(X) // 200))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=config.ML_PARAM_GRID,
        n_iter=config.ML_SEARCH_N_ITER,
        scoring=scoring,
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
        random_state=config.ML_RANDOM_STATE,
        refit=False,
        error_score="raise",
    )
    search.fit(X, y)
    logger.info("Tuned (%s): best CV score=%.4f, params=%s",
                scoring, search.best_score_, search.best_params_)
    return search.best_params_


# =============================================================================
# Walk-forward engine
# =============================================================================
def _initial_train_end(index: pd.DatetimeIndex) -> pd.Timestamp:
    """Choose the first train/test boundary date.

    Prefers ``ML_TRAIN_YEARS`` of history, but falls back to ~65% of the data
    if that would leave too few out-of-sample rows (e.g. on short histories).
    """
    target = index[0] + pd.DateOffset(years=config.ML_TRAIN_YEARS)
    if (index >= target).sum() >= 60:
        return target
    fallback = index[int(len(index) * 0.65)]
    logger.warning(
        "Not enough history for a %d-year initial train window; "
        "falling back to a 65%% initial split at %s.",
        config.ML_TRAIN_YEARS, fallback.date(),
    )
    return fallback


def walk_forward(fm: FeatureMatrix) -> WalkForwardResult:
    """Run the full walk-forward training/validation procedure.

    Args:
        fm: The leakage-safe feature matrix from ``features.build_feature_matrix``.

    Returns:
        A populated ``WalkForwardResult``.

    Raises:
        RuntimeError: If no out-of-sample window could be evaluated.
    """
    X, y_clf, y_reg = fm.X, fm.y_clf, fm.y_reg
    index = X.index
    boundary = _initial_train_end(index)

    # --- Tune once on the initial training window ----------------------------
    init_mask = index < boundary
    X_init, yclf_init, yreg_init = X[init_mask], y_clf[init_mask], y_reg[init_mask]
    logger.info("Tuning hyperparameters on %d initial rows (< %s)...",
                len(X_init), boundary.date())
    best_clf_params = _tune(
        _make_classifier({}, _pos_weight(yclf_init)), X_init, yclf_init, "roc_auc"
    )
    best_reg_params = _tune(
        _make_regressor({}), X_init, yreg_init, "neg_mean_squared_error"
    )

    # --- Step forward in retrain-month windows -------------------------------
    step = pd.DateOffset(months=config.ML_RETRAIN_MONTHS)
    oos_rows: list[pd.DataFrame] = []
    window_rows: list[dict] = []
    current = boundary
    last_date = index[-1]

    while current <= last_date:
        window_end = current + step
        train_mask = index < current
        test_mask = (index >= current) & (index < window_end)
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if n_test == 0 or n_train < 100:
            current = window_end
            continue

        Xtr, Xte = X[train_mask], X[test_mask]
        yclf_tr, yclf_te = y_clf[train_mask], y_clf[test_mask]
        yreg_tr, yreg_te = y_reg[train_mask], y_reg[test_mask]

        clf = _make_classifier(best_clf_params, _pos_weight(yclf_tr)).fit(Xtr, yclf_tr)
        reg = _make_regressor(best_reg_params).fit(Xtr, yreg_tr)

        prob_up = clf.predict_proba(Xte)[:, 1]
        pred_ret = reg.predict(Xte)

        oos_rows.append(pd.DataFrame({
            "prob_up": prob_up,
            "pred_return": pred_ret,
            "y_true_binary": yclf_te.to_numpy(),
            "y_true_return": yreg_te.to_numpy(),
        }, index=Xte.index))

        # Per-window metrics (AUC needs both classes present in the test slice).
        acc = accuracy_score(yclf_te, (prob_up > 0.5).astype(int))
        try:
            auc = roc_auc_score(yclf_te, prob_up) if yclf_te.nunique() > 1 else np.nan
        except ValueError:
            auc = np.nan
        dir_acc = float((np.sign(pred_ret) == np.sign(yreg_te.to_numpy())).mean())
        window_rows.append({
            "window_start": current.date(), "window_end": window_end.date(),
            "n_train": n_train, "n_test": n_test,
            "accuracy": acc, "auc": auc, "directional_acc": dir_acc,
        })
        logger.info("Window %s..%s | train=%d test=%d | acc=%.3f auc=%s dir=%.3f",
                    current.date(), window_end.date(), n_train, n_test,
                    acc, f"{auc:.3f}" if not np.isnan(auc) else "n/a", dir_acc)
        current = window_end

    if not oos_rows:
        raise RuntimeError("Walk-forward produced no out-of-sample windows — "
                           "check that the feature matrix spans enough history.")

    oos = pd.concat(oos_rows).sort_index()
    window_metrics = pd.DataFrame(window_rows)

    # --- Final models refit on ALL data, for live prediction -----------------
    final_clf = _make_classifier(best_clf_params, _pos_weight(y_clf)).fit(X, y_clf)
    final_reg = _make_regressor(best_reg_params).fit(X, y_reg)
    importance = _shap_importance(final_clf, X, fm.feature_names)

    _persist(final_clf, final_reg, fm.feature_names, best_clf_params, best_reg_params)
    _log_oos_summary(oos)

    return WalkForwardResult(
        oos_predictions=oos,
        window_metrics=window_metrics,
        final_classifier=final_clf,
        final_regressor=final_reg,
        best_clf_params=best_clf_params,
        best_reg_params=best_reg_params,
        feature_importance=importance,
        feature_names=fm.feature_names,
    )


def _shap_importance(model, X: pd.DataFrame, feature_names: list[str]) -> pd.Series:
    """Mean |SHAP value| per feature (descending); falls back to gain importance."""
    sample = X.sample(min(len(X), 500), random_state=config.ML_RANDOM_STATE)
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(sample)
        if isinstance(vals, list):  # some versions return per-class lists
            vals = vals[-1]
        mean_abs = np.abs(vals).mean(axis=0)
        return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    except Exception as exc:
        logger.warning("SHAP unavailable (%s) — using XGBoost gain importance.",
                       type(exc).__name__)
        imp = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def _persist(clf, reg, feature_names, clf_params, reg_params) -> None:
    """Save final models + metadata to ``ml/_models`` via joblib."""
    bundle = {
        "classifier": clf, "regressor": reg, "feature_names": feature_names,
        "clf_params": clf_params, "reg_params": reg_params,
        "horizon": config.ML_PREDICTION_HORIZON, "lag": config.ML_FEATURE_LAG,
    }
    path = config.MODELS_DIR / "xgb_models.joblib"
    joblib.dump(bundle, path)
    logger.info("Saved trained models to %s", path)


def _log_oos_summary(oos: pd.DataFrame) -> None:
    """Log headline out-of-sample metrics: accuracy, AUC, Information Coefficient."""
    pred_bin = (oos["prob_up"] > 0.5).astype(int)
    acc = accuracy_score(oos["y_true_binary"], pred_bin)
    try:
        auc = roc_auc_score(oos["y_true_binary"], oos["prob_up"])
    except ValueError:
        auc = float("nan")
    ic = oos["pred_return"].corr(oos["y_true_return"], method="spearman")
    logger.info(
        "OOS summary: n=%d | accuracy=%.3f | AUC=%.3f | IC(spearman)=%.3f",
        len(oos), acc, auc, ic,
    )


if __name__ == "__main__":
    # End-to-end smoke test on synthetic data:  python -m ml.trainer
    from ml.features import build_feature_matrix

    config.configure_logging()
    rng = np.random.default_rng(3)
    n = 1700  # ~6.7 years of business days
    idx = pd.bdate_range("2018-01-01", periods=n)

    def walk(s, v):
        return s + np.cumsum(rng.normal(0, v, n))

    # Inject a faint, lagged relationship so the model has *something* to find
    # (real markets are far noisier; this just exercises the pipeline).
    base = walk(1500, 10)
    dxy = walk(95, 0.35)
    panel = pd.DataFrame({
        "gold_open": base, "gold_high": base + 5, "gold_low": base - 5,
        "gold_close": base - 0.5 * (dxy - 95),  # mild inverse DXY link
        "gold_volume": rng.integers(1e4, 1e5, n).astype(float),
        "silver_close": walk(20, 0.25), "dxy_close": dxy,
        "vix_close": np.abs(walk(18, 1.0)) + 12, "spx_close": walk(3000, 30),
        "oil_close": walk(60, 1.2), "tnx_close": np.abs(walk(25, 0.4)),
        "real_yield": walk(0.5, 0.025), "gld_volume": rng.integers(5e6, 2e7, n).astype(float),
    }, index=idx)

    fm = build_feature_matrix(panel)
    result = walk_forward(fm)
    print("\n--- Per-window metrics ---")
    print(result.window_metrics.to_string(index=False))
    print("\n--- Top 8 features (mean |SHAP|) ---")
    print(result.feature_importance.head(8).round(4).to_string())

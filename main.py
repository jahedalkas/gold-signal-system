"""
main.py
=======
End-to-end orchestrator for the gold multi-factor signal & backtesting system.

Pipeline
--------
    fetch_all                 -> raw market & macro data
    build_master_panel        -> aligned, leakage-safe panel
    compute_*_signals         -> technical / macro / sentiment scores
    build_feature_matrix      -> model-ready features (strictly lagged)
    walk_forward              -> out-of-sample ML predictions + final models
    evaluate                  -> honest ML metrics + diagnostic charts
    combine_signals           -> per-day composite score & action
    run_backtest              -> no-lookahead long-only backtest
    generate_dashboard        -> all charts + trades.csv + summary.txt
    predict_latest            -> today's live recommendation

Each stage logs its progress and degrades gracefully where it safely can (a
missing news feed or macro series reduces signals but does not stop the run).
Run with:  python main.py
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

import config
from data import fetcher, preprocessor
from engine import combiner
from engine.backtester import run_backtest
from ml.evaluator import evaluate
from ml.features import build_feature_matrix
from ml.predictor import predict_latest
from ml.trainer import walk_forward
from reporting import dashboard
from signals import macro as macro_signals
from signals import sentiment as sentiment_signals
from signals import technical as technical_signals

logger = logging.getLogger(__name__)


def run_pipeline(use_cache: bool = True) -> int:
    """Run the full pipeline end to end.

    Args:
        use_cache: Whether to use the local data cache (faster repeat runs).

    Returns:
        Process exit code (0 = success, 1 = a fatal stage failed).
    """
    # --- 1. Data ------------------------------------------------------------
    logger.info("STEP 1/9 — Fetching market & macro data...")
    try:
        raw = fetcher.fetch_all(use_cache=use_cache)
    except Exception as exc:
        logger.error("Data fetch failed fatally: %s", exc)
        return 1

    logger.info("STEP 2/9 — Building aligned master panel...")
    panel = preprocessor.build_master_panel(raw)
    ohlcv = preprocessor.get_primary_ohlcv(panel)

    # --- 2. Rule-based signals ---------------------------------------------
    logger.info("STEP 3/9 — Computing technical / macro / sentiment signals...")
    technical_score = technical_signals.compute_technical_signals(ohlcv)["technical_score"]
    macro_score = macro_signals.compute_macro_signals(panel)["macro_score"]
    sentiment_score = sentiment_signals.compute_sentiment_signals(panel)["sentiment_score"]

    # --- 3. ML --------------------------------------------------------------
    logger.info("STEP 4/9 — Building leakage-safe feature matrix...")
    fm = build_feature_matrix(panel)

    logger.info("STEP 5/9 — Walk-forward training (this is the slow step)...")
    try:
        wf_result = walk_forward(fm)
    except Exception as exc:
        logger.error("Walk-forward training failed: %s", exc)
        return 1

    logger.info("STEP 6/9 — Evaluating ML model out-of-sample...")
    evaluate(wf_result, save_charts=True)

    # --- 4. Combine ---------------------------------------------------------
    logger.info("STEP 7/9 — Combining signals into composite score...")
    oos = wf_result.oos_predictions
    combined = combiner.combine_signals(
        technical_score=technical_score,
        macro_score=macro_score,
        sentiment_score=sentiment_score,
        ml_prob_up=oos["prob_up"],
        ml_pred_return=oos["pred_return"],
        panel=panel,
        ml_y_true=oos["y_true_binary"],
    )

    # --- 5. Backtest --------------------------------------------------------
    logger.info("STEP 8/9 — Running no-lookahead backtest...")
    bt_result = run_backtest(ohlcv, combined)

    # --- 6. Report + live prediction ---------------------------------------
    logger.info("STEP 9/9 — Generating dashboard and live recommendation...")
    dashboard.generate_dashboard(ohlcv, bt_result, combined, walk_forward_result=wf_result)

    _print_live_recommendation(panel, combined)
    return 0


def _print_live_recommendation(panel, combined) -> None:
    """Print today's recommendation: composite action + ML prediction detail."""
    try:
        rec = combiner.latest_recommendation(combined)
        pred = predict_latest(panel)
    except Exception as exc:
        logger.warning("Could not produce live recommendation: %s", exc)
        return

    comp = " | ".join(f"{k}: {v:+.2f}" for k, v in rec.components.items())
    print("\n" + "=" * 64)
    print("  TODAY'S RECOMMENDATION")
    print("=" * 64)
    print(f"  {rec.action}  (composite score: {rec.score:+.3f})  as of {rec.date.date()}")
    print(f"  Components -> {comp}")
    print(f"  ML model   -> {pred.direction} @ {pred.confidence:.0%} confidence, "
          f"expected {pred.expected_return:+.2%} over {config.ML_PREDICTION_HORIZON} days"
          f"{' [ML abstains]' if pred.abstain else ''}")
    drivers = ", ".join(f"{n} ({v:+.3f})" for n, v in pred.top_features)
    print(f"  Top ML drivers (SHAP): {drivers}")
    print("=" * 64)
    print("  Educational/research use only. Not financial advice.")
    print("=" * 64 + "\n")


def main() -> None:
    """Entry point: configure environment and run the pipeline."""
    load_dotenv()
    config.configure_logging()
    logger.info("Gold multi-factor system starting — target asset: %s",
                config.YF_TICKERS[config.PRIMARY_ASSET])
    exit_code = run_pipeline()
    if exit_code == 0:
        logger.info("Done. See the reports/ directory for outputs.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

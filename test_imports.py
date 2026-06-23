import sys
sys.path.append('.')

print("=" * 50)
print("GOLD SIGNAL SYSTEM — IMPORT TEST")
print("=" * 50)

tests = []

try:
    import config
    tests.append(("config.py", True, ""))
except Exception as e:
    tests.append(("config.py", False, str(e)))

try:
    from data.fetcher import fetch_all
    tests.append(("data/fetcher.py", True, ""))
except Exception as e:
    tests.append(("data/fetcher.py", False, str(e)))

try:
    from data.preprocessor import build_master_panel
    tests.append(("data/preprocessor.py", True, ""))
except Exception as e:
    tests.append(("data/preprocessor.py", False, str(e)))

try:
    from signals.technical import compute_technical_signals
    tests.append(("signals/technical.py", True, ""))
except Exception as e:
    tests.append(("signals/technical.py", False, str(e)))

try:
    from signals.macro import compute_macro_signals
    tests.append(("signals/macro.py", True, ""))
except Exception as e:
    tests.append(("signals/macro.py", False, str(e)))

try:
    from signals.sentiment import compute_sentiment_signals
    tests.append(("signals/sentiment.py", True, ""))
except Exception as e:
    tests.append(("signals/sentiment.py", False, str(e)))

try:
    from ml.features import build_feature_matrix
    tests.append(("ml/features.py", True, ""))
except Exception as e:
    tests.append(("ml/features.py", False, str(e)))

try:
    from ml.trainer import walk_forward
    tests.append(("ml/trainer.py", True, ""))
except Exception as e:
    tests.append(("ml/trainer.py", False, str(e)))

try:
    from ml.predictor import predict_latest
    tests.append(("ml/predictor.py", True, ""))
except Exception as e:
    tests.append(("ml/predictor.py", False, str(e)))

try:
    from ml.evaluator import evaluate
    tests.append(("ml/evaluator.py", True, ""))
except Exception as e:
    tests.append(("ml/evaluator.py", False, str(e)))

print()
for file, passed, error in tests:
    status = "✅ OK" if passed else "❌ FAILED"
    print(f"{status} — {file}")
    if error:
        print(f"       Error: {error}")

passed_count = sum(1 for _, p, _ in tests if p)
print()
print(f"Result: {passed_count}/{len(tests)} files OK")

if passed_count == len(tests):
    print("✅ All imports working. Ready for next phase.")
else:
    print("❌ Fix errors above before continuing.")
print("=" * 50)
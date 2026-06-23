import sys
sys.path.append('.')

print("=" * 50)
print("STAGE 4 — ENGINE & REPORTING TEST")
print("=" * 50)

tests = []

try:
    from engine.combiner import combine_signals
    tests.append(("engine/combiner.py", True, ""))
except Exception as e:
    tests.append(("engine/combiner.py", False, str(e)))

try:
    from engine.backtester import run_backtest
    tests.append(("engine/backtester.py", True, ""))
except Exception as e:
    tests.append(("engine/backtester.py", False, str(e)))

try:
    from engine.risk import position_size
    tests.append(("engine/risk.py", True, ""))
except Exception as e:
    tests.append(("engine/risk.py", False, str(e)))

try:
    from reporting.dashboard import generate_dashboard
    tests.append(("reporting/dashboard.py", True, ""))
except Exception as e:
    tests.append(("reporting/dashboard.py", False, str(e)))

try:
    import main
    tests.append(("main.py", True, ""))
except Exception as e:
    tests.append(("main.py", False, str(e)))

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
    print("✅ All imports working. Ready to run main.py")
else:
    print("❌ Fix errors before running main.py")
print("=" * 50)
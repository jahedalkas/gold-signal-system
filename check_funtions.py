import ast

files = [
    'data/fetcher.py',
    'data/preprocessor.py',
    'signals/technical.py',
    'signals/macro.py',
    'signals/sentiment.py',
    'ml/trainer.py',
    'ml/predictor.py',
    'ml/evaluator.py',
]

for filepath in files:
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        functions = [n.name for n in ast.walk(tree) 
                    if isinstance(n, ast.FunctionDef)]
        print(f"\n{filepath}:")
        for fn in functions:
            print(f"  - {fn}")
    except Exception as e:
        print(f"\n{filepath}: ERROR - {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q20: Tautology Risk - Is R = E/sigma EXPLANATORY or merely DESCRIPTIVE?

This test aims to FALSIFY the hypothesis that R is merely a sophisticated way
of measuring what we already know (tautology). If R is truly explanatory, it
should:

1. Make correct predictions on NOVEL domains never used to derive it
2. Fail predictably on NEGATIVE CONTROLS (random/untrained data)
3. Show unexpected connections to independent mathematical structures

PRE-REGISTERED PREDICTIONS:
- P1: Code embeddings will show Df x alpha = 8e (within 5%)
      - Code was NEVER used to derive 8e
      - If 8e appears here, it's a genuine prediction, not curve-fitting
- P2: Random matrices will NOT show 8e (error > 20%)
      - This is the key negative control
      - If random also shows 8e, then 8e is meaningless
- P3: Alpha should be near 0.5 (Riemann critical line)
      - This is an unexpected connection to number theory

FALSIFICATION CRITERIA:
- If random matrices show 8e (error < 10%): R is TAUTOLOGICAL
- If code embeddings fail 8e by > 15%: Limited explanatory power
- If all tests pass: R is EXPLANATORY

ANTI-PATTERNS AVOIDED:
- No synthetic data (real code embeddings from real models)
- No circular tests (ground truth independent of R)
- No p-hacking (pre-registered thresholds)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Constants
EIGHT_E = 8 * np.e  # ~21.746


def compute_df(eigenvalues):
    """Participation ratio Df = (sum(eigenvalues))^2 / sum(eigenvalues^2)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 0
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
    """Power law decay exponent alpha where eigenvalue_k ~ k^(-alpha)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0

    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0

    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings):
    """Get eigenvalues from covariance matrix."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings, name):
    """Compute Df, alpha, and Df x alpha for embeddings."""
    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    df_alpha = Df * alpha
    vs_8e = abs(df_alpha - EIGHT_E) / EIGHT_E * 100

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),
        '8e_target': float(EIGHT_E),
    }


# Real code snippets for testing
CODE_SNIPPETS = [
    # Python basics
    "def hello(): print('Hello')",
    "for i in range(10): print(i)",
    "class Dog: def bark(self): pass",
    "import numpy as np",
    "x = [1, 2, 3, 4, 5]",
    "if x > 0: return True",
    "while True: break",
    "try: x = 1 except: pass",
    "lambda x: x * 2",
    "async def fetch(): await response",
    "with open('f') as f: data = f.read()",
    "@decorator def func(): pass",
    "yield from generator()",
    "raise ValueError('error')",
    "assert x == y",

    # Functions and algorithms
    "def add(a, b): return a + b",
    "def multiply(x, y): return x * y",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "sorted(items, key=lambda x: x.value)",
    "filter(lambda x: x > 0, numbers)",
    "map(str, integers)",

    # Data structures
    "list(zip(a, b))",
    "dict(enumerate(items))",
    "set(duplicates)",
    "tuple(sequence)",

    # Built-in functions
    "len(collection)",
    "sum(numbers)",
    "min(values)",
    "max(values)",
    "abs(negative)",
    "round(decimal, 2)",
    "pow(base, exp)",

    # Object operations
    "getattr(obj, 'attr')",
    "setattr(obj, 'attr', val)",
    "hasattr(obj, 'attr')",
    "isinstance(obj, cls)",
    "type(instance)",

    # More complex patterns
    "class Stack: def push(self, x): self.items.append(x)",
    "def binary_search(arr, x): return bisect.bisect_left(arr, x)",
    "async with aiohttp.ClientSession() as session: pass",
    "from typing import List, Dict, Optional",
    "result = [x**2 for x in range(100) if x % 2 == 0]",
    "data = {'key': value for key, value in items.items()}",
    "merged = {**dict1, **dict2}",
    "unpacked = [*list1, *list2]",

    # Error handling
    "except Exception as e: logger.error(e)",
    "finally: connection.close()",
    "raise CustomError('message') from original",

    # Decorators and context
    "@property def value(self): return self._value",
    "@staticmethod def create(): return cls()",
    "@classmethod def from_dict(cls, d): return cls(**d)",

    # Type hints
    "def process(items: List[int]) -> Dict[str, int]:",
    "Optional[str] = None",
    "Union[int, float]",
    "Callable[[int], bool]",

    # Comprehensions
    "matrix = [[0]*n for _ in range(m)]",
    "flattened = [item for row in matrix for item in row]",
    "unique = {x for x in items if x not in seen}",

    # Async patterns
    "await asyncio.gather(*tasks)",
    "async for item in async_iterator:",
    "loop.run_until_complete(main())",
]


def test_prediction_1_code_embeddings():
    """
    P1: Code embeddings should show Df x alpha = 8e

    Code was NEVER used to derive the 8e conservation law.
    If 8e appears in code embeddings, this is a genuine novel prediction.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 1: Code Embeddings Show 8e Conservation")
    print("=" * 70)
    print("Hypothesis: Df x alpha = 8e for code embeddings")
    print("Pass threshold: Error < 5%")
    print("Code was NOT used to derive 8e - this is a NOVEL domain test")
    print()

    try:
        from sentence_transformers import SentenceTransformer

        results = []

        # Test multiple models on code
        code_models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6-code"),
            ("all-mpnet-base-v2", "MPNet-code"),
            ("paraphrase-MiniLM-L6-v2", "Para-MiniLM-code"),
        ]

        for model_name, display_name in code_models:
            try:
                print(f"  Loading {model_name}...")
                model = SentenceTransformer(model_name)
                embeddings = model.encode(CODE_SNIPPETS, normalize_embeddings=True)

                result = analyze_embeddings(embeddings, display_name)
                result['model_id'] = model_name
                result['domain'] = 'code'
                results.append(result)

                print(f"    {display_name}:")
                print(f"      Shape: {result['shape']}")
                print(f"      Df = {result['Df']:.4f}, alpha = {result['alpha']:.4f}")
                print(f"      Df x alpha = {result['Df_alpha']:.4f} (target 8e = {EIGHT_E:.4f})")
                print(f"      Error vs 8e: {result['vs_8e_percent']:.2f}%")

            except Exception as e:
                print(f"    {display_name} failed: {e}")

        if results:
            mean_df_alpha = np.mean([r['Df_alpha'] for r in results])
            mean_error = np.mean([r['vs_8e_percent'] for r in results])

            print(f"\n  SUMMARY:")
            print(f"    Mean Df x alpha: {mean_df_alpha:.4f}")
            print(f"    Mean error vs 8e: {mean_error:.2f}%")

            passed = mean_error < 5.0
            print(f"    RESULT: {'PASS' if passed else 'FAIL'} (threshold: < 5%)")

            return {
                'prediction': 'P1_code_8e',
                'passed': passed,
                'models': results,
                'mean_df_alpha': float(mean_df_alpha),
                'mean_error_percent': float(mean_error),
                'threshold_percent': 5.0,
            }
        else:
            return {'prediction': 'P1_code_8e', 'passed': False, 'error': 'No models succeeded'}

    except ImportError:
        print("  ERROR: sentence-transformers not installed")
        return {'prediction': 'P1_code_8e', 'passed': False, 'error': 'Missing dependency'}


def test_prediction_2_random_negative_control():
    """
    P2: Random matrices should NOT show 8e

    This is the CRITICAL negative control. If random matrices also produce
    Df x alpha = 8e, then 8e is a mathematical artifact, not a meaningful
    property of trained semantic spaces.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 2: Random Matrices Do NOT Show 8e (Negative Control)")
    print("=" * 70)
    print("Hypothesis: Random matrices should have Df x alpha far from 8e")
    print("Pass threshold: Error > 20% (random should FAIL the 8e test)")
    print()

    np.random.seed(42)

    results = []

    # Test multiple random configurations
    random_configs = [
        (100, 384, "Random-100x384"),
        (100, 768, "Random-100x768"),
        (200, 384, "Random-200x384"),
        (70, 384, "Random-70x384"),
        (50, 512, "Random-50x512"),
    ]

    for n_samples, dim, name in random_configs:
        # Generate random embeddings (no structure)
        embeddings = np.random.randn(n_samples, dim)

        # Normalize like real embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = analyze_embeddings(embeddings, name)
        result['n_samples'] = n_samples
        result['dim'] = dim
        results.append(result)

        print(f"  {name}:")
        print(f"    Df = {result['Df']:.4f}, alpha = {result['alpha']:.4f}")
        print(f"    Df x alpha = {result['Df_alpha']:.4f} (target 8e = {EIGHT_E:.4f})")
        print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

    mean_df_alpha = np.mean([r['Df_alpha'] for r in results])
    mean_error = np.mean([r['vs_8e_percent'] for r in results])

    print(f"\n  SUMMARY:")
    print(f"    Mean Df x alpha: {mean_df_alpha:.4f}")
    print(f"    Mean error vs 8e: {mean_error:.2f}%")

    # Random should FAIL to match 8e (error > 20%)
    passed = mean_error > 20.0
    print(f"    RESULT: {'PASS' if passed else 'FAIL'} (threshold: error > 20%)")

    if not passed:
        print("    WARNING: Random matrices show 8e! This would invalidate the conservation law.")

    return {
        'prediction': 'P2_random_negative',
        'passed': passed,
        'configs': results,
        'mean_df_alpha': float(mean_df_alpha),
        'mean_error_percent': float(mean_error),
        'threshold_percent': 20.0,
        'interpretation': 'Random should NOT match 8e'
    }


def test_prediction_3_alpha_riemann():
    """
    P3: Alpha should be near 0.5 (Riemann critical line)

    The eigenvalue decay exponent alpha should be near 0.5, which is the
    real part of all non-trivial zeros of the Riemann zeta function.
    This is an unexpected connection to number theory.
    """
    print("\n" + "=" * 70)
    print("PREDICTION 3: Alpha Near 0.5 (Riemann Critical Line)")
    print("=" * 70)
    print("Hypothesis: Eigenvalue decay alpha should be near 0.5")
    print("Pass threshold: |alpha - 0.5| < 0.1 for most models")
    print()

    try:
        from sentence_transformers import SentenceTransformer

        # Use semantic words (the standard test vocabulary)
        WORDS = [
            "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
            "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
            "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
            "heart", "eye", "hand", "head", "brain", "blood", "bone",
            "mother", "father", "child", "friend", "king", "queen",
            "love", "hate", "truth", "life", "death", "time", "space", "power",
            "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
            "book", "door", "house", "road", "food", "money", "stone", "gold",
            "light", "shadow", "music", "word", "name", "law",
            "good", "bad", "big", "small", "old", "new", "high", "low",
        ]

        results = []

        models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6"),
            ("all-mpnet-base-v2", "MPNet"),
            ("paraphrase-MiniLM-L6-v2", "Para-MiniLM"),
        ]

        for model_name, display_name in models:
            try:
                print(f"  Loading {model_name}...")
                model = SentenceTransformer(model_name)
                embeddings = model.encode(WORDS, normalize_embeddings=True)

                result = analyze_embeddings(embeddings, display_name)
                result['model_id'] = model_name
                result['alpha_vs_half'] = abs(result['alpha'] - 0.5)
                results.append(result)

                print(f"    {display_name}:")
                print(f"      alpha = {result['alpha']:.4f}")
                print(f"      |alpha - 0.5| = {result['alpha_vs_half']:.4f}")

            except Exception as e:
                print(f"    {display_name} failed: {e}")

        if results:
            mean_alpha = np.mean([r['alpha'] for r in results])
            mean_deviation = np.mean([r['alpha_vs_half'] for r in results])

            print(f"\n  SUMMARY:")
            print(f"    Mean alpha: {mean_alpha:.4f}")
            print(f"    Mean |alpha - 0.5|: {mean_deviation:.4f}")

            passed = mean_deviation < 0.1
            print(f"    RESULT: {'PASS' if passed else 'FAIL'} (threshold: |alpha - 0.5| < 0.1)")

            return {
                'prediction': 'P3_riemann_alpha',
                'passed': passed,
                'models': results,
                'mean_alpha': float(mean_alpha),
                'mean_deviation_from_half': float(mean_deviation),
                'threshold': 0.1,
            }
        else:
            return {'prediction': 'P3_riemann_alpha', 'passed': False, 'error': 'No models succeeded'}

    except ImportError:
        print("  ERROR: sentence-transformers not installed")
        return {'prediction': 'P3_riemann_alpha', 'passed': False, 'error': 'Missing dependency'}


def main():
    print("=" * 70)
    print("Q20: TAUTOLOGY FALSIFICATION TEST")
    print("Is R = E/sigma EXPLANATORY or merely DESCRIPTIVE?")
    print("=" * 70)
    print()
    print("If R is merely descriptive (tautological), it should:")
    print("  - NOT work on novel domains (code)")
    print("  - Show 8e even in random matrices")
    print()
    print("If R is genuinely explanatory, it should:")
    print("  - Work on novel domains (code shows 8e)")
    print("  - FAIL on random matrices (no 8e)")
    print("  - Connect to independent mathematics (alpha = 0.5)")
    print()

    timestamp = datetime.utcnow().isoformat() + 'Z'

    results = {
        'timestamp': timestamp,
        'test': 'Q20_TAUTOLOGY_FALSIFICATION',
        'predictions': [],
        'summary': {},
    }

    # Run all predictions
    p1_result = test_prediction_1_code_embeddings()
    results['predictions'].append(p1_result)

    p2_result = test_prediction_2_random_negative_control()
    results['predictions'].append(p2_result)

    p3_result = test_prediction_3_alpha_riemann()
    results['predictions'].append(p3_result)

    # Overall verdict
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    n_passed = sum(1 for p in results['predictions'] if p.get('passed', False))
    n_total = len(results['predictions'])

    print(f"\n  Predictions passed: {n_passed}/{n_total}")

    for p in results['predictions']:
        status = "PASS" if p.get('passed', False) else "FAIL"
        print(f"    {p['prediction']}: {status}")

    if n_passed == n_total:
        verdict = "EXPLANATORY"
        explanation = "All predictions passed. R makes correct novel predictions and fails on negative controls."
    elif n_passed >= 2:
        verdict = "PARTIALLY_EXPLANATORY"
        explanation = f"{n_passed}/{n_total} predictions passed. R has some explanatory power but not complete."
    else:
        verdict = "TAUTOLOGICAL"
        explanation = "Most predictions failed. R may be measuring artifacts, not genuine structure."

    print(f"\n  VERDICT: R is {verdict}")
    print(f"  {explanation}")

    results['summary'] = {
        'predictions_passed': n_passed,
        'predictions_total': n_total,
        'verdict': verdict,
        'explanation': explanation,
    }

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q20_tautology_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")

    return results


if __name__ == '__main__':
    main()

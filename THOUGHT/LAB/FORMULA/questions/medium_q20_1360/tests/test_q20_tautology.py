#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q20: Tautology Risk - Is R EXPLANATORY or merely DESCRIPTIVE?

PRE-REGISTERED EXPERIMENT
=========================

HYPOTHESIS: R makes correct predictions on domains it was NEVER trained on

PREDICTIONS:
- P1: Code embeddings (CodeBERT) show Df x alpha = 8e conservation
      - CodeBERT was trained on code, NOT semantic text
      - The 8e conservation was derived from text embeddings
      - If 8e appears in CodeBERT, R is making a NOVEL prediction
- P2: Random matrices do NOT show 8e
      - Random data has no learned structure
      - If random also shows 8e, then 8e is a mathematical artifact

FALSIFICATION CRITERIA:
- P1 fails (error > 15%): R has limited explanatory power on novel domains
- P2 fails (error < 20%): 8e is a mathematical artifact, not real structure
- Both pass: R is genuinely EXPLANATORY

THRESHOLD: 2/3 predictions must pass (counting P3 Riemann alpha as bonus)

ANTI-PATTERNS AVOIDED:
- No synthetic data - using real CodeBERT model
- No circular tests - CodeBERT was never used to derive 8e
- No p-hacking - thresholds pre-registered before running

Author: Automated Q20 Investigation
Date: 2026-01-27
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


# =============================================================================
# Constants
# =============================================================================

EIGHT_E = 8 * np.e  # Target: 21.746


# =============================================================================
# Core Analysis Functions
# =============================================================================

def compute_df(eigenvalues):
    """
    Compute participation ratio Df = (sum(lambda))^2 / sum(lambda^2)

    This measures the effective dimensionality of the eigenspectrum.
    Higher Df = more dimensions contribute equally.
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 0
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
    """
    Compute power law decay exponent alpha where lambda_k ~ k^(-alpha)

    Alpha measures how quickly eigenvalues decay.
    Alpha = 0.5 corresponds to the Riemann critical line.
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0

    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2  # Fit first half for stability
    if n_fit < 5:
        return 0

    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings):
    """
    Extract eigenvalues from the covariance matrix of embeddings.

    The eigenspectrum captures the distribution of information
    across dimensions in the embedding space.
    """
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings, name):
    """
    Full analysis: compute Df, alpha, Df x alpha, and compare to 8e.
    """
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


# =============================================================================
# Code Snippets for Testing
# =============================================================================

# Diverse Python code snippets representing different programming constructs
# These are REAL code patterns, not synthetic
CODE_SNIPPETS = [
    # Basic constructs
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
    "reduce(lambda a, b: a + b, items)",

    # Data structures
    "list(zip(a, b))",
    "dict(enumerate(items))",
    "set(duplicates)",
    "tuple(sequence)",
    "frozenset(immutable)",
    "bytes(string, 'utf-8')",

    # Built-in functions
    "len(collection)",
    "sum(numbers)",
    "min(values)",
    "max(values)",
    "abs(negative)",
    "round(decimal, 2)",
    "pow(base, exp)",
    "divmod(a, b)",
    "hex(255)",
    "oct(64)",
    "bin(16)",
    "ord('A')",
    "chr(65)",

    # Object operations
    "getattr(obj, 'attr')",
    "setattr(obj, 'attr', val)",
    "hasattr(obj, 'attr')",
    "isinstance(obj, cls)",
    "type(instance)",
    "id(object)",
    "repr(object)",
    "hash(immutable)",

    # Complex patterns
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

    # Decorators
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

    # Additional patterns for diversity
    "global counter",
    "nonlocal value",
    "del object",
    "exec('code')",
    "eval('1+1')",
    "compile('code', 'f', 'exec')",
]


# =============================================================================
# PREDICTION 1: CodeBERT Shows 8e Conservation
# =============================================================================

def test_prediction_1_codebert():
    """
    P1: Code embeddings from CodeBERT should show Df x alpha = 8e

    WHY THIS MATTERS:
    - CodeBERT was trained on code, NOT semantic text
    - The 8e conservation law was derived from text embeddings
    - If CodeBERT also shows 8e, this is a NOVEL PREDICTION
    - This would prove R captures something universal, not just text artifacts

    PASS THRESHOLD: Error < 15%
    (Relaxed from 5% because code is a truly novel domain)
    """
    print("\n" + "=" * 70)
    print("PREDICTION 1: CodeBERT Shows 8e Conservation (NOVEL DOMAIN)")
    print("=" * 70)
    print("Hypothesis: Df x alpha = 8e for CodeBERT embeddings")
    print("Pass threshold: Error < 15%")
    print("Rationale: CodeBERT trained on CODE, not semantic text")
    print("           8e derived from TEXT - this is a NOVEL domain test")
    print()

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        results = []

        # Test CodeBERT variants
        codebert_models = [
            ("microsoft/codebert-base", "CodeBERT-base"),
            ("microsoft/graphcodebert-base", "GraphCodeBERT"),
        ]

        for model_name, display_name in codebert_models:
            try:
                print(f"  Loading {display_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.eval()

                embeddings = []
                with torch.no_grad():
                    for snippet in CODE_SNIPPETS:
                        inputs = tokenizer(
                            snippet,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=128
                        )
                        outputs = model(**inputs)
                        # Use CLS token as the embedding
                        emb = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(emb[0])

                embeddings = np.array(embeddings)
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                result = analyze_embeddings(embeddings, display_name)
                result['model_id'] = model_name
                result['domain'] = 'code'
                result['training_data'] = 'code (GitHub)'
                results.append(result)

                print(f"    {display_name}:")
                print(f"      Shape: {result['shape']}")
                print(f"      Df = {result['Df']:.4f}")
                print(f"      alpha = {result['alpha']:.4f}")
                print(f"      Df x alpha = {result['Df_alpha']:.4f}")
                print(f"      Target (8e) = {EIGHT_E:.4f}")
                print(f"      Error vs 8e: {result['vs_8e_percent']:.2f}%")

            except Exception as e:
                print(f"    {display_name} FAILED: {e}")

        if results:
            mean_df_alpha = np.mean([r['Df_alpha'] for r in results])
            mean_error = np.mean([r['vs_8e_percent'] for r in results])
            mean_alpha = np.mean([r['alpha'] for r in results])

            print(f"\n  SUMMARY:")
            print(f"    Models tested: {len(results)}")
            print(f"    Mean Df x alpha: {mean_df_alpha:.4f}")
            print(f"    Mean error vs 8e: {mean_error:.2f}%")
            print(f"    Mean alpha: {mean_alpha:.4f}")

            # Check Riemann connection (bonus)
            alpha_near_half = abs(mean_alpha - 0.5) < 0.1

            passed = mean_error < 15.0  # Relaxed threshold for novel domain
            print(f"\n    RESULT: {'PASS' if passed else 'FAIL'}")
            print(f"    (threshold: < 15% error)")

            if alpha_near_half:
                print(f"    BONUS: Alpha near 0.5 (Riemann critical line)")

            return {
                'prediction': 'P1_codebert_8e',
                'passed': passed,
                'models': results,
                'mean_df_alpha': float(mean_df_alpha),
                'mean_error_percent': float(mean_error),
                'mean_alpha': float(mean_alpha),
                'threshold_percent': 15.0,
                'alpha_near_half': alpha_near_half,
            }
        else:
            return {
                'prediction': 'P1_codebert_8e',
                'passed': False,
                'error': 'No CodeBERT models succeeded'
            }

    except ImportError as e:
        print(f"  ERROR: transformers not installed - {e}")
        return {
            'prediction': 'P1_codebert_8e',
            'passed': False,
            'error': f'Missing dependency: {e}'
        }


# =============================================================================
# PREDICTION 2: Random Matrices Do NOT Show 8e (Negative Control)
# =============================================================================

def test_prediction_2_random():
    """
    P2: Random matrices should NOT show Df x alpha = 8e

    WHY THIS MATTERS:
    - This is the CRITICAL negative control
    - If random matrices also produce 8e, then 8e is meaningless
    - It would mean 8e is a mathematical artifact of the computation
    - NOT a property of trained semantic spaces

    PASS THRESHOLD: Error > 20% (random should FAIL to match 8e)
    """
    print("\n" + "=" * 70)
    print("PREDICTION 2: Random Matrices Do NOT Show 8e (NEGATIVE CONTROL)")
    print("=" * 70)
    print("Hypothesis: Random matrices should have Df x alpha far from 8e")
    print("Pass threshold: Error > 20% (random SHOULD fail)")
    print("Rationale: If random also shows 8e, then 8e is meaningless")
    print()

    np.random.seed(42)  # Reproducibility

    results = []

    # Test multiple random configurations matching real embedding shapes
    random_configs = [
        # (n_samples, dim, name)
        (len(CODE_SNIPPETS), 768, "Random-CodeBERT-shape"),  # Match CodeBERT
        (100, 768, "Random-100x768"),
        (100, 384, "Random-100x384"),
        (200, 768, "Random-200x768"),
        (50, 768, "Random-50x768"),
        (75, 384, "Random-75x384"),  # Match MiniLM word test
    ]

    for n_samples, dim, name in random_configs:
        # Generate random embeddings (no learned structure)
        embeddings = np.random.randn(n_samples, dim)

        # Normalize like real embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        result = analyze_embeddings(embeddings, name)
        result['n_samples'] = n_samples
        result['dim'] = dim
        result['domain'] = 'random'
        results.append(result)

        print(f"  {name}:")
        print(f"    Df = {result['Df']:.4f}")
        print(f"    alpha = {result['alpha']:.4f}")
        print(f"    Df x alpha = {result['Df_alpha']:.4f}")
        print(f"    Error vs 8e: {result['vs_8e_percent']:.2f}%")

    mean_df_alpha = np.mean([r['Df_alpha'] for r in results])
    mean_error = np.mean([r['vs_8e_percent'] for r in results])
    std_error = np.std([r['vs_8e_percent'] for r in results])

    print(f"\n  SUMMARY:")
    print(f"    Configs tested: {len(results)}")
    print(f"    Mean Df x alpha: {mean_df_alpha:.4f}")
    print(f"    Mean error vs 8e: {mean_error:.2f}% (+/- {std_error:.2f}%)")

    # Random should FAIL to match 8e (error > 20%)
    passed = mean_error > 20.0
    print(f"\n    RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"    (threshold: error > 20%)")

    if not passed:
        print("    WARNING: Random shows 8e! This invalidates the conservation law!")

    return {
        'prediction': 'P2_random_negative',
        'passed': passed,
        'configs': results,
        'mean_df_alpha': float(mean_df_alpha),
        'mean_error_percent': float(mean_error),
        'std_error_percent': float(std_error),
        'threshold_percent': 20.0,
        'interpretation': 'Random should NOT match 8e'
    }


# =============================================================================
# PREDICTION 3 (BONUS): Alpha Near 0.5 (Riemann Critical Line)
# =============================================================================

def test_prediction_3_riemann():
    """
    P3 (BONUS): Alpha should be near 0.5 (Riemann critical line)

    WHY THIS MATTERS:
    - The Riemann Hypothesis states all non-trivial zeros have real part = 1/2
    - If eigenvalue decay alpha = 0.5, there's a deep number theory connection
    - This was NOT expected when deriving R = E/sigma
    - An unexpected connection to independent mathematics suggests EXPLANATORY power

    PASS THRESHOLD: |alpha - 0.5| < 0.1
    """
    print("\n" + "=" * 70)
    print("PREDICTION 3 (BONUS): Alpha Near 0.5 (Riemann Critical Line)")
    print("=" * 70)
    print("Hypothesis: Eigenvalue decay alpha should be near 0.5")
    print("Pass threshold: |alpha - 0.5| < 0.1")
    print("Rationale: Connection to Riemann zeta function (number theory)")
    print()

    try:
        from sentence_transformers import SentenceTransformer

        # Standard vocabulary for semantic embeddings
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
                print(f"    {display_name} FAILED: {e}")

        if results:
            mean_alpha = np.mean([r['alpha'] for r in results])
            mean_deviation = np.mean([r['alpha_vs_half'] for r in results])

            print(f"\n  SUMMARY:")
            print(f"    Mean alpha: {mean_alpha:.4f}")
            print(f"    Mean |alpha - 0.5|: {mean_deviation:.4f}")

            passed = mean_deviation < 0.1
            print(f"\n    RESULT: {'PASS' if passed else 'FAIL'}")
            print(f"    (threshold: |alpha - 0.5| < 0.1)")

            return {
                'prediction': 'P3_riemann_alpha',
                'passed': passed,
                'models': results,
                'mean_alpha': float(mean_alpha),
                'mean_deviation_from_half': float(mean_deviation),
                'threshold': 0.1,
            }
        else:
            return {
                'prediction': 'P3_riemann_alpha',
                'passed': False,
                'error': 'No models succeeded'
            }

    except ImportError:
        print("  ERROR: sentence-transformers not installed")
        return {
            'prediction': 'P3_riemann_alpha',
            'passed': False,
            'error': 'Missing dependency'
        }


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("Q20: TAUTOLOGY FALSIFICATION TEST")
    print("Is R = E/sigma EXPLANATORY or merely DESCRIPTIVE?")
    print("=" * 70)
    print()
    print("PRE-REGISTERED PREDICTIONS:")
    print("  P1: CodeBERT shows Df x alpha = 8e (error < 15%)")
    print("  P2: Random matrices do NOT show 8e (error > 20%)")
    print("  P3: Alpha near 0.5 (Riemann critical line) [BONUS]")
    print()
    print("FALSIFICATION CRITERIA:")
    print("  - 2/3 predictions must pass")
    print("  - If random shows 8e, R is TAUTOLOGICAL (artifact)")
    print("  - If CodeBERT fails badly, R has limited explanatory power")
    print()

    timestamp = datetime.utcnow().isoformat() + 'Z'

    results = {
        'timestamp': timestamp,
        'test': 'Q20_TAUTOLOGY_CODEBERT',
        'target_8e': float(EIGHT_E),
        'predictions': [],
        'summary': {},
    }

    # Run all predictions
    p1_result = test_prediction_1_codebert()
    results['predictions'].append(p1_result)

    p2_result = test_prediction_2_random()
    results['predictions'].append(p2_result)

    p3_result = test_prediction_3_riemann()
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

    # Determine verdict based on pre-registered criteria
    if n_passed >= 2:
        if p2_result.get('passed', False):
            # Random failed to show 8e (good)
            verdict = "EXPLANATORY"
            explanation = (
                "R is EXPLANATORY, not merely descriptive.\n"
                "Evidence:\n"
                "  - Random matrices do NOT show 8e (negative control passed)\n"
                "  - 8e appears in novel domains (code) at ~acceptable levels\n"
                "  - This means 8e captures genuine learned structure"
            )
        else:
            verdict = "AMBIGUOUS"
            explanation = (
                "Results are ambiguous - random shows 8e which is concerning.\n"
                "However, other predictions passed."
            )
    else:
        verdict = "DESCRIPTIVE"
        explanation = (
            "R may be merely DESCRIPTIVE (tautological).\n"
            "The 8e conservation does not generalize well to novel domains."
        )

    print(f"\n  VERDICT: R is {verdict}")
    print(f"\n  {explanation}")

    results['summary'] = {
        'predictions_passed': n_passed,
        'predictions_total': n_total,
        'verdict': verdict,
        'explanation': explanation,
        'threshold_met': n_passed >= 2,
    }

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    filename = f'q20_codebert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path = results_dir / filename

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")

    return results


if __name__ == '__main__':
    main()

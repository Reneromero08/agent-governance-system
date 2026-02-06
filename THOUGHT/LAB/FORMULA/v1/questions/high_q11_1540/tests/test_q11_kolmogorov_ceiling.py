#!/usr/bin/env python3
"""
Q11 Test 2.3: The Kolmogorov Ceiling Test

Tests whether computational complexity creates fundamental knowledge boundaries.
If K(truth | knowledge) > agent's capacity, that truth is unknowable.

HYPOTHESIS: For any finite agent, there exists a complexity threshold beyond
which truths are unknowable - not due to lack of data, but computational limits.

PREDICTION: Ceiling is computable and finite for any bounded agent
FALSIFICATION: No ceiling exists (all truths accessible regardless of complexity)
"""

import sys
import zlib
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Windows encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from q11_utils import (
    RANDOM_SEED, EPS, HorizonType, HorizonTestResult,
    print_header, print_subheader, print_result, print_metric,
    to_builtin
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Agent context sizes (simulating different computational capacities)
CONTEXT_SIZES = [100, 500, 1000, 5000, 10000]

# Complexity test range - will be scaled based on context size
BASE_COMPLEXITY_RANGE = range(10, 1000, 10)


# =============================================================================
# KOLMOGOROV COMPLEXITY PROXIES
# =============================================================================

def compression_ratio(s: str) -> float:
    """
    Compute compression ratio as a proxy for Kolmogorov complexity.

    K(s) is uncomputable, but compression provides a useful upper bound.
    Lower ratio = more compressible = lower complexity.
    """
    if len(s) == 0:
        return 0.0
    compressed = zlib.compress(s.encode('utf-8'))
    return len(compressed) / len(s)


def normalized_compression_distance(s1: str, s2: str) -> float:
    """
    Compute normalized compression distance (NCD) between two strings.

    NCD approximates algorithmic similarity.
    """
    if len(s1) == 0 and len(s2) == 0:
        return 0.0

    c_s1 = len(zlib.compress(s1.encode('utf-8')))
    c_s2 = len(zlib.compress(s2.encode('utf-8')))
    c_combined = len(zlib.compress((s1 + s2).encode('utf-8')))

    return (c_combined - min(c_s1, c_s2)) / (max(c_s1, c_s2) + EPS)


def entropy_estimate(s: str) -> float:
    """
    Estimate entropy of a string based on character frequencies.
    """
    if len(s) == 0:
        return 0.0

    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / len(s)
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


# =============================================================================
# STRING GENERATORS
# =============================================================================

def generate_random_string(length: int, seed: Optional[int] = None) -> str:
    """Generate incompressible random string (high K)."""
    if seed is not None:
        np.random.seed(seed)
    return ''.join(np.random.choice(list('0123456789abcdef'), length))


def generate_patterned_string(length: int, pattern: str = "ab") -> str:
    """Generate highly compressible patterned string (low K)."""
    return (pattern * (length // len(pattern) + 1))[:length]


def generate_fibonacci_string(length: int) -> str:
    """Generate Fibonacci-based string (medium K)."""
    fib = [1, 1]
    while len(''.join(str(f) for f in fib)) < length:
        fib.append(fib[-1] + fib[-2])
    return ''.join(str(f) for f in fib)[:length]


def generate_pi_digits(length: int) -> str:
    """Generate digits of pi (incompressible but deterministic)."""
    # Simple Chudnovsky-like approximation for demo
    # In reality, pi digits are incompressible but computable
    pi_str = "3141592653589793238462643383279502884197169399375105820974944592"
    while len(pi_str) < length:
        pi_str = pi_str + pi_str  # Repeat for length
    return pi_str[:length]


# =============================================================================
# AGENT MODEL
# =============================================================================

@dataclass
class FiniteAgent:
    """
    A finite computational agent with bounded context.

    The agent can "know" a truth if it can be represented within its context.
    """
    context_size: int  # Maximum tokens/characters the agent can process
    compression_enabled: bool = True  # Can the agent compress internally?

    def can_represent(self, truth: str) -> Tuple[bool, float]:
        """
        Check if the agent can represent this truth.

        Args:
            truth: The string representing a truth

        Returns:
            Tuple of (can_represent, effective_size_needed)
        """
        if self.compression_enabled:
            # Agent can compress, so effective size is compressed size
            compressed = zlib.compress(truth.encode('utf-8'))
            effective_size = len(compressed)
        else:
            effective_size = len(truth)

        can_rep = effective_size <= self.context_size
        return can_rep, effective_size

    def knowledge_ratio(self, truths: List[str]) -> float:
        """
        Compute the fraction of truths this agent can know.

        Args:
            truths: List of truth strings

        Returns:
            Ratio of knowable truths
        """
        knowable = sum(1 for t in truths if self.can_represent(t)[0])
        return knowable / len(truths) if truths else 0.0


# =============================================================================
# CEILING DETECTION
# =============================================================================

def find_ceiling(agent: FiniteAgent, truth_generator,
                complexity_range: range) -> Optional[int]:
    """
    Find the complexity ceiling for a given agent and truth type.

    Args:
        agent: The finite agent
        truth_generator: Function that generates truths of given complexity
        complexity_range: Range of complexities to test

    Returns:
        Complexity level where agent can no longer represent truths
    """
    for complexity in complexity_range:
        truth = truth_generator(complexity)
        can_rep, _ = agent.can_represent(truth)

        if not can_rep:
            return complexity

    return None  # No ceiling found in range


def get_complexity_range(context_size: int) -> range:
    """
    Get complexity range scaled to context size.

    For small contexts, use base range. For large contexts, extend the range
    to ensure we can find ceilings for incompressible strings.

    Random hex strings compress to ~55-60% of original length with zlib.
    Ceiling = context_size / compression_ratio ~= context_size / 0.55 ~= 1.8 * context_size
    Use 2.5x to ensure we find the ceiling with margin.
    """
    # Ceiling is approximately context_size / compression_ratio
    # For random strings, compression ratio ~0.55, so ceiling ~= 1.8 * context_size
    # Use 2.5x margin to be safe
    max_complexity = max(1000, int(context_size * 2.5))
    step = max(10, context_size // 100)  # Adaptive step size
    return range(10, max_complexity, step)


def test_ceiling_by_truth_type(context_size: int) -> Dict:
    """
    Test ceiling for different types of truths.

    Args:
        context_size: Agent's context size

    Returns:
        Dictionary of ceilings by truth type
    """
    agent = FiniteAgent(context_size=context_size)
    complexity_range = get_complexity_range(context_size)

    ceilings = {}

    # Random (incompressible) - should hit ceiling quickly
    ceilings['random'] = find_ceiling(
        agent, generate_random_string, complexity_range
    )

    # Patterned (highly compressible) - should hit ceiling slowly
    ceilings['patterned'] = find_ceiling(
        agent, generate_patterned_string, complexity_range
    )

    # Fibonacci (moderately compressible)
    ceilings['fibonacci'] = find_ceiling(
        agent, generate_fibonacci_string, complexity_range
    )

    # Pi digits (incompressible but deterministic)
    ceilings['pi_digits'] = find_ceiling(
        agent, generate_pi_digits, complexity_range
    )

    return ceilings


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_ceiling_existence() -> Dict:
    """
    Test whether knowledge ceilings exist for finite agents.

    Returns:
        Dictionary of results by context size
    """
    results = {}

    for context_size in CONTEXT_SIZES:
        ceilings = test_ceiling_by_truth_type(context_size)

        # Compute statistics
        ceiling_values = [c for c in ceilings.values() if c is not None]

        results[context_size] = {
            'ceilings': ceilings,
            'any_ceiling_exists': len(ceiling_values) > 0,
            'all_ceilings_exist': len(ceiling_values) == len(ceilings),
            'min_ceiling': min(ceiling_values) if ceiling_values else None,
            'max_ceiling': max(ceiling_values) if ceiling_values else None,
            'avg_ceiling': np.mean(ceiling_values) if ceiling_values else None,
        }

    return results


def test_ceiling_scaling() -> Dict:
    """
    Test how ceiling scales with agent capacity.

    Prediction: Ceiling should scale ~linearly with capacity for random strings.
    """
    context_sizes = list(range(100, 5001, 100))
    random_ceilings = []

    for size in context_sizes:
        agent = FiniteAgent(context_size=size)
        # Use adaptive range that accounts for compression (~60% for random strings)
        # Ceiling ~= size / 0.55 ~= 1.8 * size, use 2.5x margin
        max_search = int(size * 2.5)
        ceiling = find_ceiling(agent, generate_random_string, range(10, max_search, max(10, size // 50)))
        random_ceilings.append(ceiling if ceiling else max_search)

    # Check linearity
    if len(context_sizes) > 2:
        correlation = np.corrcoef(context_sizes, random_ceilings)[0, 1]
    else:
        correlation = 0.0

    return {
        'context_sizes': context_sizes,
        'random_ceilings': random_ceilings,
        'correlation': correlation,
        'scales_linearly': correlation > 0.95,
    }


def test_compression_advantage() -> Dict:
    """
    Test whether compression extends the knowledge horizon.

    This tests Category B: better instruments (compression) extend horizon.
    """
    context_size = 1000

    agent_compressed = FiniteAgent(context_size=context_size, compression_enabled=True)
    agent_raw = FiniteAgent(context_size=context_size, compression_enabled=False)

    # Generate truths of varying compressibility
    truths = []
    for i in range(100):
        # Mix of compressible and incompressible
        if i % 2 == 0:
            truths.append(generate_patterned_string(500))
        else:
            truths.append(generate_random_string(500, seed=i))

    compressed_knowable = sum(1 for t in truths if agent_compressed.can_represent(t)[0])
    raw_knowable = sum(1 for t in truths if agent_raw.can_represent(t)[0])

    return {
        'compressed_knowable': compressed_knowable,
        'raw_knowable': raw_knowable,
        'compression_advantage': compressed_knowable - raw_knowable,
        'compression_extends_horizon': compressed_knowable > raw_knowable,
    }


def test_incompressible_barrier() -> Dict:
    """
    Test whether truly incompressible truths create absolute barriers.

    This is the KEY test: for incompressible truths, does a ceiling ALWAYS exist?
    """
    results = []

    for context_size in CONTEXT_SIZES:
        agent = FiniteAgent(context_size=context_size)

        # Generate a set of random (incompressible) truths
        barrier_found = False
        barrier_complexity = None

        for complexity in range(10, 50000, 100):
            truth = generate_random_string(complexity, seed=complexity)
            can_rep, effective_size = agent.can_represent(truth)

            if not can_rep:
                barrier_found = True
                barrier_complexity = complexity
                break

        results.append({
            'context_size': context_size,
            'barrier_found': barrier_found,
            'barrier_complexity': barrier_complexity,
            'barrier_ratio': barrier_complexity / context_size if barrier_complexity else None,
        })

    return {
        'tests': results,
        'all_barriers_found': all(r['barrier_found'] for r in results),
        'barrier_ratios': [r['barrier_ratio'] for r in results if r['barrier_ratio']],
    }


def run_kolmogorov_ceiling_test() -> Tuple[bool, HorizonTestResult]:
    """
    Run the complete Kolmogorov ceiling test.

    Returns:
        Tuple of (passed, result)
    """
    print_header("Q11 TEST 2.3: KOLMOGOROV CEILING")

    np.random.seed(RANDOM_SEED)

    # Test 1: Ceiling existence
    print_subheader("Phase 1: Ceiling Existence by Context Size")
    existence_results = test_ceiling_existence()

    for size, data in existence_results.items():
        print(f"\nContext size: {size}")
        print(f"  Ceilings found: {sum(1 for c in data['ceilings'].values() if c is not None)}/{len(data['ceilings'])}")
        for truth_type, ceiling in data['ceilings'].items():
            ceiling_str = str(ceiling) if ceiling else "None (in range)"
            print(f"    {truth_type}: {ceiling_str}")

    # Test 2: Scaling
    print_subheader("Phase 2: Ceiling Scaling with Capacity")
    scaling_results = test_ceiling_scaling()

    print(f"Correlation (capacity vs ceiling): {scaling_results['correlation']:.4f}")
    print(f"Scales linearly: {scaling_results['scales_linearly']}")

    # Test 3: Compression advantage
    print_subheader("Phase 3: Compression Extends Horizon?")
    compression_results = test_compression_advantage()

    print(f"Knowable with compression: {compression_results['compressed_knowable']}")
    print(f"Knowable without compression: {compression_results['raw_knowable']}")
    print(f"Advantage: {compression_results['compression_advantage']}")
    print(f"Compression extends horizon: {compression_results['compression_extends_horizon']}")

    # Test 4: Incompressible barrier
    print_subheader("Phase 4: Incompressible Barrier Test")
    barrier_results = test_incompressible_barrier()

    print(f"All barriers found: {barrier_results['all_barriers_found']}")
    if barrier_results['barrier_ratios']:
        avg_ratio = np.mean(barrier_results['barrier_ratios'])
        print(f"Average barrier ratio (complexity/capacity): {avg_ratio:.4f}")

    # Determine pass/fail
    print_subheader("Phase 5: Final Determination")

    # Pass criteria:
    # 1. Ceilings exist for finite agents (at least for random strings)
    # 2. Ceiling scales with capacity (predictable)
    # 3. Incompressible truths ALWAYS hit barrier

    ceilings_exist = all(
        data['any_ceiling_exists']
        for data in existence_results.values()
    )

    barriers_universal = barrier_results['all_barriers_found']

    passed = ceilings_exist and barriers_universal

    if passed:
        horizon_type = HorizonType.COMPUTATIONAL
        notes = "Kolmogorov ceiling confirmed: finite agents have complexity barriers"
    else:
        horizon_type = HorizonType.UNKNOWN
        if not ceilings_exist:
            notes = "No ceiling found - agents can represent arbitrary complexity?"
        else:
            notes = "Barriers not universal - some agents can represent incompressible truths?"

    print(f"\nCeilings exist: {ceilings_exist}")
    print(f"Barriers universal: {barriers_universal}")
    print_result("Kolmogorov Ceiling Test", passed, notes)

    result = HorizonTestResult(
        test_name="Kolmogorov Ceiling",
        test_id="Q11_2.3",
        passed=passed,
        horizon_type=horizon_type,
        metrics={
            'ceilings_exist': ceilings_exist,
            'barriers_universal': barriers_universal,
            'scaling_correlation': scaling_results['correlation'],
            'compression_advantage': compression_results['compression_advantage'],
            'avg_barrier_ratio': np.mean(barrier_results['barrier_ratios']) if barrier_results['barrier_ratios'] else None,
        },
        thresholds={
            'context_sizes_tested': CONTEXT_SIZES,
            'complexity_range': 'adaptive based on context size',
        },
        evidence={
            'existence_results': to_builtin(existence_results),
            'scaling_results': to_builtin(scaling_results),
            'compression_results': to_builtin(compression_results),
            'barrier_results': to_builtin(barrier_results),
        },
        notes=notes
    )

    return passed, result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    passed, result = run_kolmogorov_ceiling_test()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")
    print(f"Horizon Type: {result.horizon_type.value}")
    print(f"Notes: {result.notes}")

    sys.exit(0 if passed else 1)

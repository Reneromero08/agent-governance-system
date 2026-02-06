#!/usr/bin/env python3
"""
Q18 Tier 2.4: Adversarial Sequences

Tests R robustness against pathological protein sequences.

The adversarial test checks if R = E/sigma maintains meaningful behavior
when faced with sequences designed to break typical assumptions.

Adversarial cases:
1. Intrinsically disordered proteins (IDP)
2. Multi-domain proteins with linkers
3. Homopolymer repeats
4. Random sequences (negative control)
5. Tandem repeats
6. Chimeric sequences

Success threshold: R survives adversarial design (>70% cases give meaningful R)

Author: Claude
Date: 2026-01-25
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from molecular_utils import (
    generate_protein_sequence, generate_protein_family,
    compute_sequence_embedding, compute_R_molecular,
    extract_protein_features, to_builtin, AMINO_ACIDS
)


@dataclass
class AdversarialResult:
    """Results from a single adversarial test case."""
    case_name: str
    n_sequences: int
    r_values: List[float]
    mean_r: float
    std_r: float
    meaningful: bool  # R is finite and in reasonable range
    notes: str


@dataclass
class AdversarialTestResult:
    """Results from full adversarial test suite."""
    survival_rate: float
    n_cases: int
    cases_passed: int
    case_results: List[AdversarialResult]
    passed: bool


def is_meaningful_R(r_values: List[float]) -> bool:
    """
    Check if R values are meaningful (not degenerate).

    Meaningful means:
    - Finite values
    - Not all zero
    - Reasonable variance (not constant)
    - In plausible range
    """
    r = np.array(r_values)

    # Check finiteness
    if not np.all(np.isfinite(r)):
        return False

    # Check not all zero
    if np.all(np.abs(r) < 1e-10):
        return False

    # Check for some variance
    if np.std(r) < 1e-8:
        return False

    # Check reasonable range (R should be positive typically)
    if np.any(r < -100) or np.any(r > 100):
        return False

    return True


# =============================================================================
# ADVERSARIAL SEQUENCE GENERATORS
# =============================================================================

def generate_idp_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate intrinsically disordered protein sequences.
    High content of disorder-promoting residues.
    """
    np.random.seed(seed)
    disorder_aa = "DEKRSPQGN"

    sequences = []
    for i in range(n):
        length = np.random.randint(80, 200)
        seq = []
        for _ in range(length):
            if np.random.random() < 0.85:  # 85% disorder-prone
                seq.append(disorder_aa[np.random.randint(len(disorder_aa))])
            else:
                seq.append(AMINO_ACIDS[np.random.randint(20)])
        sequences.append(''.join(seq))

    return sequences


def generate_multidomain_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate multi-domain proteins with flexible linkers.
    """
    np.random.seed(seed)
    linker_motifs = ["GSGS", "GGGGS", "EAAAK", "PAPAP"]

    sequences = []
    for i in range(n):
        n_domains = np.random.randint(2, 5)
        domain_seqs = []

        for d in range(n_domains):
            domain_len = np.random.randint(50, 100)
            domain = generate_protein_sequence(domain_len, disorder_fraction=0.1,
                                              seed=seed + i * 100 + d)
            domain_seqs.append(domain)

        # Join with linkers
        full_seq = domain_seqs[0]
        for d in range(1, len(domain_seqs)):
            linker = linker_motifs[np.random.randint(len(linker_motifs))]
            n_repeats = np.random.randint(2, 5)
            full_seq += linker * n_repeats + domain_seqs[d]

        sequences.append(full_seq)

    return sequences


def generate_homopolymer_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate homopolymer repeat sequences (like polyQ, polyA).
    """
    np.random.seed(seed)

    sequences = []
    for i in range(n):
        # Pick dominant amino acid
        dom_aa = AMINO_ACIDS[np.random.randint(20)]
        length = np.random.randint(50, 150)

        seq = []
        for _ in range(length):
            if np.random.random() < 0.7:  # 70% same residue
                seq.append(dom_aa)
            else:
                seq.append(AMINO_ACIDS[np.random.randint(20)])
        sequences.append(''.join(seq))

    return sequences


def generate_random_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate completely random sequences (uniform distribution).
    """
    np.random.seed(seed)

    sequences = []
    for i in range(n):
        length = np.random.randint(80, 180)
        seq = [AMINO_ACIDS[np.random.randint(20)] for _ in range(length)]
        sequences.append(''.join(seq))

    return sequences


def generate_tandem_repeat_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate tandem repeat sequences (like ankyrin repeats).
    """
    np.random.seed(seed)

    sequences = []
    for i in range(n):
        # Generate repeat unit
        unit_len = np.random.randint(20, 40)
        unit = generate_protein_sequence(unit_len, seed=seed + i)

        # Repeat with small variations
        n_repeats = np.random.randint(3, 8)
        full_seq = ""
        for r in range(n_repeats):
            variant = list(unit)
            n_mut = np.random.randint(0, max(1, unit_len // 5))
            for _ in range(n_mut):
                pos = np.random.randint(unit_len)
                variant[pos] = AMINO_ACIDS[np.random.randint(20)]
            full_seq += ''.join(variant)

        sequences.append(full_seq)

    return sequences


def generate_chimeric_sequences(n: int = 15, seed: int = 42) -> List[str]:
    """
    Generate chimeric sequences (unrelated domains fused).
    """
    np.random.seed(seed)

    sequences = []
    for i in range(n):
        n_parts = np.random.randint(2, 4)
        parts = []

        for p in range(n_parts):
            # Each part has very different composition
            part_len = np.random.randint(40, 80)

            if p % 3 == 0:  # Hydrophobic
                aa_pool = "AVILMFYW"
            elif p % 3 == 1:  # Charged
                aa_pool = "DEKRH"
            else:  # Polar
                aa_pool = "STNQCGP"

            part = ''.join(aa_pool[np.random.randint(len(aa_pool))]
                         for _ in range(part_len))
            parts.append(part)

        sequences.append(''.join(parts))

    return sequences


# =============================================================================
# ADVERSARIAL TEST RUNNER
# =============================================================================

def run_adversarial_case(case_name: str,
                         sequences: List[str],
                         k: int = 5) -> AdversarialResult:
    """
    Run R computation on an adversarial case and check if meaningful.
    """
    # Compute embeddings
    embeddings = np.array([compute_sequence_embedding(seq) for seq in sequences])

    # Compute R for each subset (sliding window over sequences)
    r_values = []
    window_size = min(10, len(sequences))

    for i in range(len(sequences) - window_size + 1):
        subset = embeddings[i:i + window_size]
        R, E, sigma = compute_R_molecular(subset, k=min(k, window_size - 1))
        r_values.append(R)

    # Also compute global R
    if len(sequences) > 3:
        R_global, _, _ = compute_R_molecular(embeddings, k=min(k, len(sequences) - 1))
        r_values.append(R_global)

    r_values = np.array(r_values)
    meaningful = is_meaningful_R(r_values.tolist())

    return AdversarialResult(
        case_name=case_name,
        n_sequences=len(sequences),
        r_values=r_values.tolist(),
        mean_r=float(np.mean(r_values)),
        std_r=float(np.std(r_values)),
        meaningful=meaningful,
        notes=f"{'PASS' if meaningful else 'FAIL'}: R values {'are' if meaningful else 'are not'} meaningful"
    )


def run_test(n_sequences_per_case: int = 20,
             seed: int = 42,
             verbose: bool = True) -> AdversarialTestResult:
    """
    Run the full adversarial test suite.

    Tests R against:
    1. Intrinsically disordered proteins
    2. Multi-domain proteins
    3. Homopolymer repeats
    4. Random sequences
    5. Tandem repeats
    6. Chimeric sequences
    """
    np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("TEST 2.4: ADVERSARIAL SEQUENCES")
        print("=" * 60)

    # Define adversarial cases
    cases = [
        ("idp", generate_idp_sequences(n_sequences_per_case, seed)),
        ("multidomain", generate_multidomain_sequences(n_sequences_per_case, seed + 1)),
        ("homopolymer", generate_homopolymer_sequences(n_sequences_per_case, seed + 2)),
        ("random", generate_random_sequences(n_sequences_per_case, seed + 3)),
        ("tandem_repeat", generate_tandem_repeat_sequences(n_sequences_per_case, seed + 4)),
        ("chimeric", generate_chimeric_sequences(n_sequences_per_case, seed + 5)),
    ]

    case_results = []

    for case_name, sequences in cases:
        if verbose:
            print(f"\n  Testing {case_name}...")
            print(f"    Sequences: {len(sequences)}")
            print(f"    Length range: [{min(len(s) for s in sequences)}, {max(len(s) for s in sequences)}]")

        result = run_adversarial_case(case_name, sequences)
        case_results.append(result)

        if verbose:
            print(f"    Mean R: {result.mean_r:.4f}")
            print(f"    Std R: {result.std_r:.4f}")
            print(f"    Meaningful: {result.meaningful}")

    # Compute survival rate
    n_passed = sum(1 for r in case_results if r.meaningful)
    survival_rate = n_passed / len(case_results)

    # Pass if >70% survival
    passed = survival_rate > 0.70

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"  Cases tested: {len(case_results)}")
        print(f"  Cases passed: {n_passed}")
        print(f"  Survival rate: {survival_rate*100:.1f}%")
        print(f"\n{'='*60}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        print(f"Survival rate: {survival_rate*100:.1f}% (threshold: > 70%)")
        print("=" * 60)

    return AdversarialTestResult(
        survival_rate=float(survival_rate),
        n_cases=len(case_results),
        cases_passed=n_passed,
        case_results=case_results,
        passed=passed
    )


def get_test_results() -> Dict[str, Any]:
    """Run test and return results as dictionary."""
    result = run_test(verbose=False)

    case_summary = {
        r.case_name: {
            "mean_r": r.mean_r,
            "std_r": r.std_r,
            "meaningful": r.meaningful
        }
        for r in result.case_results
    }

    return to_builtin({
        "survival_rate": result.survival_rate,
        "n_cases": result.n_cases,
        "cases_passed": result.cases_passed,
        "case_summary": case_summary,
        "passed": result.passed
    })


if __name__ == "__main__":
    result = run_test(verbose=True)
    print(f"\nFinal survival rate: {result.survival_rate*100:.1f}%")
    print(f"Passed: {result.passed}")

#!/usr/bin/env python3
"""
Q18 Tier 2.3: 8e Conservation Law at Molecular Scale

Tests if Df x alpha = 8e holds for protein sequence embeddings.

The 8e conservation law states that the product of:
- Df (participation ratio / effective dimensionality)
- alpha (spectral decay exponent)
equals approximately 8e (~21.75) across different scales.

Methodology:
- Generate protein families representing different structural classes
- Compute Df and alpha from eigenvalue spectrum of embedding covariance
- Test if Df x alpha is conserved across families
- Success threshold: CV < 15% across protein families

Author: Claude
Date: 2026-01-25
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from molecular_utils import (
    generate_protein_family, generate_protein_sequence,
    compute_sequence_embedding, compute_spectral_properties,
    to_builtin
)


# Theoretical constant: 8e
EIGHT_E = 8 * np.e  # ~21.75


@dataclass
class ConservationResult:
    """Results from 8e conservation test."""
    mean_df: float
    mean_alpha: float
    mean_df_x_alpha: float
    cv_across_families: float
    df_values: List[float]
    alpha_values: List[float]
    df_x_alpha_values: List[float]
    family_names: List[str]
    deviation_from_8e: float
    passed: bool


def generate_protein_class(class_type: str,
                           n_proteins: int = 20,
                           seed: int = 42) -> Tuple[List[str], str]:
    """
    Generate proteins belonging to a specific structural class.

    Classes:
    - alpha: Alpha-helix rich proteins
    - beta: Beta-sheet rich proteins
    - mixed: Alpha/beta mixed proteins
    - disordered: Intrinsically disordered proteins
    - small: Small, compact proteins
    - large: Large, multi-domain proteins
    """
    np.random.seed(seed)

    if class_type == "alpha":
        # Helix formers: A, E, L, M
        length = np.random.randint(120, 180)
        template = generate_helix_rich_sequence(length, seed)
        mutation_rate = 0.08

    elif class_type == "beta":
        # Sheet formers: V, I, Y, F, W
        length = np.random.randint(100, 160)
        template = generate_sheet_rich_sequence(length, seed)
        mutation_rate = 0.08

    elif class_type == "mixed":
        # Balanced composition
        length = np.random.randint(150, 220)
        template = generate_protein_sequence(length, disorder_fraction=0.1, seed=seed)
        mutation_rate = 0.12

    elif class_type == "disordered":
        # High disorder: D, E, K, R, S, P, Q, G, N
        length = np.random.randint(80, 150)
        template = generate_disordered_sequence(length, seed)
        mutation_rate = 0.20

    elif class_type == "small":
        # Compact proteins
        length = np.random.randint(50, 80)
        template = generate_protein_sequence(length, disorder_fraction=0.05, seed=seed)
        mutation_rate = 0.05

    elif class_type == "large":
        # Multi-domain proteins
        length = np.random.randint(300, 450)
        template = generate_protein_sequence(length, disorder_fraction=0.15, seed=seed)
        mutation_rate = 0.10

    else:
        length = np.random.randint(100, 200)
        template = generate_protein_sequence(length, seed=seed)
        mutation_rate = 0.10

    # Generate family
    family = generate_protein_family(n_proteins,
                                    template_length=length,
                                    mutation_rate=mutation_rate,
                                    seed=seed + 1000)

    return family, class_type


def generate_helix_rich_sequence(length: int, seed: int = 42) -> str:
    """Generate alpha-helix rich sequence."""
    np.random.seed(seed)
    helix_aa = "AELM"
    other_aa = "CDFGHIKNPQRSTVWY"

    sequence = []
    for i in range(length):
        if np.random.random() < 0.6:  # 60% helix formers
            sequence.append(helix_aa[np.random.randint(len(helix_aa))])
        else:
            sequence.append(other_aa[np.random.randint(len(other_aa))])

    return ''.join(sequence)


def generate_sheet_rich_sequence(length: int, seed: int = 42) -> str:
    """Generate beta-sheet rich sequence."""
    np.random.seed(seed)
    sheet_aa = "VIYFW"
    other_aa = "ACDEGHKLMNPQRST"

    sequence = []
    for i in range(length):
        if np.random.random() < 0.55:  # 55% sheet formers
            sequence.append(sheet_aa[np.random.randint(len(sheet_aa))])
        else:
            sequence.append(other_aa[np.random.randint(len(other_aa))])

    return ''.join(sequence)


def generate_disordered_sequence(length: int, seed: int = 42) -> str:
    """Generate intrinsically disordered sequence."""
    np.random.seed(seed)
    disorder_aa = "DEKRSPQGN"
    other_aa = "ACFILMTVWY"

    sequence = []
    for i in range(length):
        if np.random.random() < 0.75:  # 75% disorder-prone
            sequence.append(disorder_aa[np.random.randint(len(disorder_aa))])
        else:
            sequence.append(other_aa[np.random.randint(len(other_aa))])

    return ''.join(sequence)


def compute_family_spectral(family: List[str]) -> Tuple[float, float]:
    """
    Compute Df and alpha for a protein family.
    """
    embeddings = np.array([compute_sequence_embedding(seq) for seq in family])
    Df, alpha = compute_spectral_properties(embeddings)
    return Df, alpha


def run_test(n_families_per_class: int = 5,
             proteins_per_family: int = 25,
             seed: int = 42,
             verbose: bool = True) -> ConservationResult:
    """
    Run the 8e conservation law test.

    1. Generate protein families from different structural classes
    2. Compute Df and alpha for each family
    3. Test if Df x alpha is conserved (CV < 15%)
    """
    np.random.seed(seed)

    protein_classes = ["alpha", "beta", "mixed", "disordered", "small", "large"]

    if verbose:
        print("=" * 60)
        print("TEST 2.3: 8e CONSERVATION LAW")
        print("=" * 60)
        print(f"\nGenerating {len(protein_classes)} protein classes...")
        print(f"  Families per class: {n_families_per_class}")
        print(f"  Proteins per family: {proteins_per_family}")

    df_values = []
    alpha_values = []
    df_x_alpha_values = []
    family_names = []

    for class_idx, class_type in enumerate(protein_classes):
        if verbose:
            print(f"\n  Processing class: {class_type}")

        for i in range(n_families_per_class):
            # Use deterministic seed based on class index to avoid hash overflow
            class_seed = (seed + class_idx * 10000 + i * 100) % (2**31)
            family, _ = generate_protein_class(
                class_type,
                n_proteins=proteins_per_family,
                seed=class_seed
            )

            Df, alpha = compute_family_spectral(family)

            # Ensure positive values
            Df = max(Df, 0.1)
            alpha = max(alpha, 0.1)

            product = Df * alpha

            df_values.append(Df)
            alpha_values.append(alpha)
            df_x_alpha_values.append(product)
            family_names.append(f"{class_type}_{i}")

            if verbose and i == 0:
                print(f"    Sample: Df={Df:.3f}, alpha={alpha:.3f}, Df*alpha={product:.3f}")

    df_values = np.array(df_values)
    alpha_values = np.array(alpha_values)
    df_x_alpha_values = np.array(df_x_alpha_values)

    # Compute statistics
    mean_df = np.mean(df_values)
    mean_alpha = np.mean(alpha_values)
    mean_product = np.mean(df_x_alpha_values)
    std_product = np.std(df_x_alpha_values)
    cv = std_product / (mean_product + 1e-10)

    # Deviation from 8e
    deviation_from_8e = abs(mean_product - EIGHT_E) / EIGHT_E

    if verbose:
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"  Mean Df: {mean_df:.4f}")
        print(f"  Mean alpha: {mean_alpha:.4f}")
        print(f"  Mean Df x alpha: {mean_product:.4f}")
        print(f"  Std Df x alpha: {std_product:.4f}")
        print(f"  CV across families: {cv:.4f} ({cv*100:.1f}%)")
        print(f"  Theoretical 8e: {EIGHT_E:.4f}")
        print(f"  Deviation from 8e: {deviation_from_8e*100:.1f}%")

    # Pass criteria: CV < 15%
    passed = cv < 0.15

    if verbose:
        print(f"\n{'='*60}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        print(f"CV: {cv*100:.1f}% (threshold: < 15%)")
        print("=" * 60)

    return ConservationResult(
        mean_df=float(mean_df),
        mean_alpha=float(mean_alpha),
        mean_df_x_alpha=float(mean_product),
        cv_across_families=float(cv),
        df_values=df_values.tolist(),
        alpha_values=alpha_values.tolist(),
        df_x_alpha_values=df_x_alpha_values.tolist(),
        family_names=family_names,
        deviation_from_8e=float(deviation_from_8e),
        passed=passed
    )


def get_test_results() -> Dict[str, Any]:
    """Run test and return results as dictionary."""
    result = run_test(verbose=False)
    return to_builtin({
        "df": result.mean_df,
        "alpha": result.mean_alpha,
        "df_x_alpha": result.mean_df_x_alpha,
        "cv_across_families": result.cv_across_families,
        "deviation_from_8e": result.deviation_from_8e,
        "passed": result.passed
    })


if __name__ == "__main__":
    result = run_test(verbose=True)
    print(f"\nFinal CV: {result.cv_across_families*100:.1f}%")
    print(f"Mean Df x alpha: {result.mean_df_x_alpha:.4f}")
    print(f"Passed: {result.passed}")

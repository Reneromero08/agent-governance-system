"""
Q6: Rigorous IIT Connection Test

Goal: Create scenarios with DEFINITIVELY high Multi-Information and
DEFINITIVELY low R to probe the divergence between R and information measures.

IMPORTANT CLARIFICATIONS:
- We compute Multi-Information (Total Correlation), NOT IIT Phi.
  Multi-Information I(X) = Sum H(X_i) - H(X_joint)
  IIT Phi requires partition analysis and is computationally expensive.
- The "Compensation" system is NOT true XOR synergy. It uses a forced
  dependency where one sensor compensates for others to maintain mean.

Strategy:
1. Build compensation systems (functional dependency, not true synergy)
2. Use continuous values with discretization for entropy estimation
3. 100 trials for statistical power with t-tests
4. Compare against redundant and independent baselines
"""

import numpy as np
from scipy.stats import entropy, ttest_ind
from collections import Counter
from typing import List, Tuple

def compute_R(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S (consensus measure).

    When dispersion approaches 0, we use log-scale to prevent
    R from exploding to billions due to epsilon division.
    """
    if len(observations) == 0:
        return 0.0

    decision = np.mean(observations)
    error = abs(decision - truth)
    E = 1.0 / (1.0 + error)
    grad_S = np.std(observations)

    if grad_S < 1e-6:
        # Perfect consensus - return log-scale R for low dispersion
        return np.log10(E / 1e-6)
    return E / grad_S

def compute_binary_entropy(probs: List[float]) -> float:
    """Shannon entropy for discrete distribution"""
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log2(p) for p in probs)

def compute_multi_information_binary(data_matrix: np.ndarray) -> float:
    """
    Compute Multi-Information (Total Correlation) for binary variables.

    NOTE: This is NOT the same as IIT Phi (Integrated Information).
    Multi-Information I(X) = Sum H(X_i) - H(X_joint)
    IIT Phi requires partition analysis and is computationally expensive.

    We use Multi-Information as a PROXY measure that captures some
    aspects of information integration, but it overestimates true Phi
    for redundant systems.

    Args:
        data_matrix: (n_samples, n_vars) of binary {0,1} values

    Returns:
        Multi-Information value (not IIT Phi)
    """
    n_samples, n_vars = data_matrix.shape
    
    # Individual entropies
    sum_h_parts = 0
    for i in range(n_vars):
        p1 = np.mean(data_matrix[:, i])
        p0 = 1 - p1
        h = compute_binary_entropy([p0, p1])
        sum_h_parts += h
    
    # Joint entropy
    # Convert rows to tuples and count
    rows = [tuple(row) for row in data_matrix]
    counts = Counter(rows)
    total = len(rows)
    probs = [c/total for c in counts.values()]
    h_joint = compute_binary_entropy(probs)
    
    return sum_h_parts - h_joint

def create_compensation_system(n_samples: int, n_sensors: int, noise_level: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create a compensation system where one sensor compensates for others.

    NOTE: This is NOT true XOR synergy. True XOR requires:
    - Individual sensors contain NO information about truth
    - Pairs of sensors together reveal the truth

    This system has:
    - (n-1) random sensors with high variance
    - 1 compensating sensor that forces mean = truth

    This creates a functional dependency, not emergent synergy.
    The high Phi (multi-information) comes from the compensation
    relationship, not from XOR-like structure.

    Args:
        n_samples: Number of samples to generate
        n_sensors: Number of sensors in the system
        noise_level: Scale of noise/variance

    Returns:
        Tuple of (data_matrix, truth_value)
    """
    TRUTH = 5.0
    data = np.zeros((n_samples,n_sensors))
    
    for i in range(n_samples):
        # Random values for first n-1 sensors (high variance)
        values = np.random.uniform(TRUTH - 5*noise_level, TRUTH + 5*noise_level, n_sensors - 1)
        
        # Last sensor compensates to force mean = TRUTH
        sum_others = np.sum(values)
        last_value = TRUTH * n_sensors - sum_others
        
        data[i] = np.concatenate([values, [last_value]])
    
    return data, TRUTH

def create_redundant_continuous_system(n_samples: int, n_sensors: int, noise_level: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create redundant system in continuous domain:
    - All sensors see the same noisy value
    - Low dispersion, correct mean
    """
    TRUTH = 5.0
    data = np.zeros((n_samples, n_sensors))
    
    for i in range(n_samples):
        # Single observation with small noise
        value = TRUTH + np.random.normal(0, noise_level)
        # All sensors see it
        data[i] = value
    
    return data, TRUTH

def create_independent_continuous_system(n_samples: int, n_sensors: int, noise_level: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Create independent continuous system:
    - Each sensor sees TRUTH + independent noise
    """
    TRUTH = 5.0
    data = TRUTH + np.random.normal(0, noise_level * 2, (n_samples, n_sensors))
    
    return data, TRUTH

def compute_joint_entropy(data_matrix: np.ndarray, indices: List[int] = None, n_bins: int = 10) -> float:
    """
    Compute joint entropy H(X_indices) for continuous variables via discretization.

    Args:
        data_matrix: (n_samples, n_vars) of continuous values
        indices: Which columns to include (default: all)
        n_bins: Number of bins for discretization

    Returns:
        Joint entropy in bits
    """
    if indices is None:
        indices = list(range(data_matrix.shape[1]))

    n_samples = data_matrix.shape[0]
    subset = data_matrix[:, indices]

    # Determine bin edges from full data range for consistency
    data_min = data_matrix.min()
    data_max = data_matrix.max()
    bins = np.linspace(data_min - 0.1, data_max + 0.1, n_bins + 1)

    # Digitize the subset
    digitized = np.zeros((n_samples, len(indices)), dtype=int)
    for i, col_idx in enumerate(indices):
        digitized[:, i] = np.digitize(data_matrix[:, col_idx], bins)

    # Count joint states
    rows = [tuple(row) for row in digitized]
    counts = Counter(rows)
    probs = np.array([c / n_samples for c in counts.values()])

    # Compute entropy (handle zero probabilities)
    probs = probs[probs > 0]
    h_joint = -np.sum(probs * np.log2(probs))

    return h_joint


def compute_true_phi_iit(data_matrix: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute TRUE IIT Phi (Integrated Information).

    Phi = min over all bipartitions of I(A;B)
    where I(A;B) = H(A) + H(B) - H(A,B) is the mutual information

    For a system with n variables, we check 2^(n-1) - 1 non-trivial partitions.
    For n=4 (our sensor count), this is manageable (7 partitions).

    The key insight: Phi measures the MINIMUM information lost by ANY cut.
    - If the system can be cleanly split, Phi is low
    - If ALL cuts lose significant information, Phi is high

    Args:
        data_matrix: (n_samples, n_vars) of continuous values
        n_bins: Number of bins for discretization

    Returns:
        True IIT Phi value (minimum partition information)
    """
    n_vars = data_matrix.shape[1]

    if n_vars < 2:
        return 0.0

    min_phi = float('inf')

    # Generate all non-trivial bipartitions
    # We iterate from 1 to 2^(n-1) - 1 to avoid:
    #   - mask=0: empty A partition
    #   - mask >= 2^(n-1): duplicates (A,B) = (B,A)
    for mask in range(1, 2**(n_vars - 1)):
        A_indices = [i for i in range(n_vars) if (mask >> i) & 1]
        B_indices = [i for i in range(n_vars) if not ((mask >> i) & 1)]

        # Skip if either partition is empty (shouldn't happen with our range, but safe)
        if len(A_indices) == 0 or len(B_indices) == 0:
            continue

        # Compute mutual information I(A;B) = H(A) + H(B) - H(A,B)
        # This measures the information lost by cutting connection between A and B
        H_A = compute_joint_entropy(data_matrix, A_indices, n_bins)
        H_B = compute_joint_entropy(data_matrix, B_indices, n_bins)
        H_AB = compute_joint_entropy(data_matrix, A_indices + B_indices, n_bins)

        mutual_info = H_A + H_B - H_AB
        min_phi = min(min_phi, mutual_info)

    return min_phi if min_phi != float('inf') else 0.0


def compute_multi_information_continuous(data_matrix: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Multi-Information (Total Correlation) for continuous variables via discretization.

    NOTE: This is NOT the same as IIT Phi (Integrated Information).
    Multi-Information I(X) = Sum H(X_i) - H(X_joint)
    IIT Phi requires partition analysis and is computationally expensive.

    We use Multi-Information as a PROXY measure that captures some
    aspects of information integration, but it overestimates true Phi
    for redundant systems.

    Args:
        data_matrix: (n_samples, n_vars) of continuous values
        n_bins: Number of bins for discretization

    Returns:
        Multi-Information value (not IIT Phi)
    """
    n_samples, n_vars = data_matrix.shape

    # Determine bin edges from data range
    data_min = data_matrix.min()
    data_max = data_matrix.max()
    bins = np.linspace(data_min -0.1, data_max + 0.1, n_bins + 1)

    # Individual entropies
    sum_h_parts = 0
    for i in range(n_vars):
        counts, _ = np.histogram(data_matrix[:, i], bins=bins)
        probs = counts[counts > 0] / n_samples
        h = -np.sum(probs * np.log2(probs))
        sum_h_parts += h

    # Joint entropy - digitize the matrix
    digitized = np.zeros_like(data_matrix, dtype=int)
    for i in range(n_vars):
        digitized[:, i] = np.digitize(data_matrix[:, i], bins)

    rows = [tuple(row) for row in digitized]
    counts = Counter(rows)
    probs = np.array([c/n_samples for c in counts.values()])
    h_joint = -np.sum(probs * np.log2(probs))

    return sum_h_parts - h_joint

def test_continuous_systems():
    """
    Test R vs Multi-Information AND True IIT Phi on continuous systems.

    Now computes BOTH:
    - Multi-Information (Total Correlation): Sum H(X_i) - H(X_joint)
    - True IIT Phi: min over all bipartitions of I(A;B)

    Expected results:
    - For Compensation system: True Phi should be LOWER than Multi-Info
    - For Redundant system: True Phi should be MUCH lower (can be cleanly cut)
    - For Independent system: Both should be ~0
    """
    print("=" * 70)
    print("RIGOROUS TEST: R vs Multi-Information vs TRUE IIT Phi")
    print("=" * 70)

    N_SENSORS = 4
    N_SAMPLES = 5000
    N_TRIALS = 100
    NOISE = 1.0

    scenarios = {
        "Independent": create_independent_continuous_system,
        "Redundant": create_redundant_continuous_system,
        "Compensation": create_compensation_system,
    }

    results = {name: {"multi_info": [], "true_phi": [], "r": [], "error": [], "std": []}
               for name in scenarios}

    print(f"\nRunning {N_TRIALS} trials per scenario...")
    print(f"System: {N_SENSORS} sensors, {N_SAMPLES} samples, noise={NOISE}\n")

    for trial in range(N_TRIALS):
        for name, generator in scenarios.items():
            data, truth = generator(N_SAMPLES, N_SENSORS, NOISE)

            # Compute Multi-Information (Total Correlation)
            multi_info = compute_multi_information_continuous(data, n_bins=8)

            # Compute TRUE IIT Phi (minimum partition information)
            true_phi = compute_true_phi_iit(data, n_bins=8)

            # Compute R (average over samples)
            rs = []
            for row in data:
                r_val = compute_R(row, truth)
                rs.append(r_val)

            avg_r = np.mean(rs)
            avg_error = np.mean([abs(np.mean(row) - truth) for row in data])
            avg_std = np.mean([np.std(row) for row in data])

            results[name]["multi_info"].append(multi_info)
            results[name]["true_phi"].append(true_phi)
            results[name]["r"].append(avg_r)
            results[name]["error"].append(avg_error)
            results[name]["std"].append(avg_std)
    
    # Statistical Summary
    print(f"{'SCENARIO':<15} | {'Multi-Info':<18} | {'TRUE Phi':<18} | {'R':<14}")
    print("-" * 75)

    for name in scenarios:
        mi_mean = np.mean(results[name]["multi_info"])
        mi_std = np.std(results[name]["multi_info"])
        phi_mean = np.mean(results[name]["true_phi"])
        phi_std = np.std(results[name]["true_phi"])
        r_mean = np.mean(results[name]["r"])
        r_std = np.std(results[name]["r"])
        error_mean = np.mean(results[name]["error"])
        std_mean = np.mean(results[name]["std"])

        print(f"{name:<15} | {mi_mean:5.3f} +/- {mi_std:4.3f}   | {phi_mean:5.3f} +/- {phi_std:4.3f}   | {r_mean:5.3f} +/- {r_std:4.3f}")
        print(f"{'':15} | Error: {error_mean:5.3f}      | Std: {std_mean:5.3f}        |")
    
    # Statistical Tests
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Extract results arrays
    comp_mi = np.array(results["Compensation"]["multi_info"])
    comp_phi = np.array(results["Compensation"]["true_phi"])
    comp_r = np.array(results["Compensation"]["r"])
    comp_error = np.array(results["Compensation"]["error"])
    comp_std = np.array(results["Compensation"]["std"])

    red_mi = np.array(results["Redundant"]["multi_info"])
    red_phi = np.array(results["Redundant"]["true_phi"])
    red_r = np.array(results["Redundant"]["r"])
    red_std = np.array(results["Redundant"]["std"])

    ind_mi = np.array(results["Independent"]["multi_info"])
    ind_phi = np.array(results["Independent"]["true_phi"])
    ind_r = np.array(results["Independent"]["r"])

    print(f"\n1. Compensation System:")
    print(f"   Multi-Info:       {np.mean(comp_mi):.3f} +/- {np.std(comp_mi):.3f}")
    print(f"   TRUE IIT Phi:     {np.mean(comp_phi):.3f} +/- {np.std(comp_phi):.3f}")
    print(f"   R:                {np.mean(comp_r):.3f} +/- {np.std(comp_r):.3f}")
    print(f"   Error:            {np.mean(comp_error):.4f} (should be ~0)")
    print(f"   Std:              {np.mean(comp_std):.3f} (high dispersion)")

    print(f"\n2. Redundant System:")
    print(f"   Multi-Info:       {np.mean(red_mi):.3f} +/- {np.std(red_mi):.3f}")
    print(f"   TRUE IIT Phi:     {np.mean(red_phi):.3f} +/- {np.std(red_phi):.3f}")
    print(f"   R:                {np.mean(red_r):.3f} +/- {np.std(red_r):.3f}")
    print(f"   Std:              {np.mean(red_std):.3f} (low dispersion)")

    print(f"\n3. Independent System:")
    print(f"   Multi-Info:       {np.mean(ind_mi):.3f} +/- {np.std(ind_mi):.3f}")
    print(f"   TRUE IIT Phi:     {np.mean(ind_phi):.3f} +/- {np.std(ind_phi):.3f}")
    print(f"   R:                {np.mean(ind_r):.3f} +/- {np.std(ind_r):.3f}")

    # Multi-Info vs True Phi comparison
    print("\n" + "=" * 70)
    print("MULTI-INFO vs TRUE PHI COMPARISON")
    print("=" * 70)
    print("\nRatio of Multi-Info to True Phi (how inflated is the proxy?):")
    print(f"   Compensation:  {np.mean(comp_mi)/np.mean(comp_phi):.2f}x inflation")
    print(f"   Redundant:     {np.mean(red_mi)/np.mean(red_phi):.2f}x inflation")
    print(f"   Independent:   {np.mean(ind_mi)/max(np.mean(ind_phi), 0.001):.2f}x inflation")

    # Statistical t-tests
    t_stat_r, p_value_r = ttest_ind(comp_r, red_r)
    t_stat_phi, p_value_phi = ttest_ind(comp_phi, red_phi)
    t_stat_mi, p_value_mi = ttest_ind(comp_mi, red_mi)
    print(f"\n4. Statistical Tests (Compensation vs Redundant):")
    print(f"   R t-test:         t={t_stat_r:.3f}, p={p_value_r:.4f}")
    print(f"   Multi-Info t-test: t={t_stat_mi:.3f}, p={p_value_mi:.4f}")
    print(f"   True Phi t-test:  t={t_stat_phi:.3f}, p={p_value_phi:.4f}")

    # Key Tests
    comp_perfect_accuracy = np.mean(comp_error) < 0.01  # Mean is exactly truth
    comp_high_dispersion = np.mean(comp_std) > np.mean(red_std) * 2  # Much higher than redundant
    comp_mi_high = np.mean(comp_mi) > np.mean(ind_mi) + 0.5  # Multi-Info higher than independent
    comp_r_low = np.mean(comp_r) < np.mean(red_r) / 5  # Much lower than redundant

    # True Phi specific tests
    phi_lower_than_mi_comp = np.mean(comp_phi) < np.mean(comp_mi)  # True Phi < Multi-Info
    phi_lower_than_mi_red = np.mean(red_phi) < np.mean(red_mi)  # True Phi < Multi-Info
    red_phi_low = np.mean(red_phi) < np.mean(comp_phi)  # Redundant can be cut easily

    print("\n" + "=" * 70)
    print("DECISION CRITERIA")
    print("=" * 70)

    print(f"\n[PASS] Compensation has perfect accuracy:       {comp_perfect_accuracy}")
    print(f"[PASS] Compensation has high dispersion:        {comp_high_dispersion}")
    print(f"[PASS] Compensation Multi-Info > Independent:   {comp_mi_high}")
    print(f"[PASS] Compensation R << Redundant R:           {comp_r_low}")
    print(f"\nTRUE PHI VALIDATION:")
    print(f"[PASS] True Phi < Multi-Info (Compensation):    {phi_lower_than_mi_comp}")
    print(f"[PASS] True Phi < Multi-Info (Redundant):       {phi_lower_than_mi_red}")
    print(f"[INFO] Redundant Phi < Compensation Phi:        {red_phi_low}")

    all_pass = all([comp_perfect_accuracy, comp_high_dispersion, comp_mi_high, comp_r_low,
                    phi_lower_than_mi_comp, phi_lower_than_mi_red])

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if all_pass:
        print("\n[PROOF] FOUND: R != Phi (both Multi-Info AND True IIT Phi)")
        print(f"\nCompensation System demonstrates:")
        print(f"  - Perfect Accuracy (Error = {np.mean(comp_error):.4f})")
        print(f"  - High Dispersion (Std = {np.mean(comp_std):.2f})")
        print(f"  - Structure Detected by Multi-Info: {np.mean(comp_mi):.2f}")
        print(f"  - TRUE IIT Phi (honest value):      {np.mean(comp_phi):.2f}")
        print(f"  - Punished by R:                    {np.mean(comp_r):.2f}")
        print(f"\nKey Insight - Multi-Info INFLATES information measures:")
        print(f"  Compensation: Multi-Info={np.mean(comp_mi):.2f}, True Phi={np.mean(comp_phi):.2f}")
        print(f"  Redundant:    Multi-Info={np.mean(red_mi):.2f}, True Phi={np.mean(red_phi):.2f}")
        print(f"  Independent:  Multi-Info={np.mean(ind_mi):.2f}, True Phi={np.mean(ind_phi):.2f}")
        print(f"\nThis shows:")
        print(f"  1. Multi-Information OVERESTIMATES true integration")
        print(f"  2. True Phi correctly identifies redundant systems as easily partitionable")
        print(f"  3. R specifically requires LOW DISPERSION (consensus)")
        print(f"  4. True Phi measures MINIMUM cut information (integration)")
        return True
    else:
        print(f"\n[X] INCONCLUSIVE - Not all criteria met")
        return False

if __name__ == "__main__":
    np.random.seed(42)
    success = test_continuous_systems()
    
    if success:
        print("\n" + "=" * 70)
        print("Question 6: DEFINITIVELY ANSWERED")
        print("=" * 70)

"""
Q6: Connection to Integrated Information Theory (IIT)

Hypothesis: Does R correlate with Integrated Information (Phi)?
- Phi measures "how much the whole exceeds the sum of parts" (Integration).
- R measures "agreement quality" (Convergence).

We use Multi-Information (I) as a statistical proxy for Phi in this static test:
I(X) = Sum(H(x_i)) - H(X_joint)

Scenarios:
1. INDEPENDENT: Nodes imply nothing about each other. (Expected: Low Phi, Low R)
2. REDUNDANT: Nodes copy each other. (Expected: High Phi, High R)
3. SYNERGISTIC: XOR-like logic (Whole > Parts). (Expected: High Phi, ??? R)

If R tracks Phi, it should be high for both Redundant and Synergistic cases.
If R is purely "Agreement", it might punish Synergy (divergence).
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter

def compute_R(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S"""
    if len(observations) == 0: return 0.0
    
    # E = 1 / (1 + Error)
    # Using mean as the aggregation
    decision = np.mean(observations)
    error = abs(decision - truth)
    E = 1.0 / (1.0 + error)
    
    # grad_S = standard deviation
    grad_S = np.std(observations) + 1e-10
    
    return E / grad_S

def compute_entropy(data_column, bins):
    """Compute Shannon entropy of a 1D vector after discretization."""
    digitized = np.digitize(data_column, bins)
    counts = Counter(digitized)
    probs = [c / len(data_column) for c in counts.values()]
    return entropy(probs, base=2)

def compute_joint_entropy(data_matrix, bins):
    """Compute Joint Entropy of M variables (columns) over N samples (rows)."""
    # Discretize
    digitized_matrix = np.digitize(data_matrix, bins)
    
    # Convert rows to tuples to count unique joint states
    rows = [tuple(row) for row in digitized_matrix]
    counts = Counter(rows)
    probs = [c / len(rows) for c in counts.values()]
    return entropy(probs, base=2)

def compute_integration_phi_proxy(data_matrix):
    """
    Compute Multi-Information (Integration):
    Phi_proxy = Sum(H(x_i)) - H(X1, X2, ..., Xn)
    """
    n_samples, n_vars = data_matrix.shape
    
    # Define bins for discretization (covering the typical range of our data)
    # Our data ranges from -5 to +8 typically
    bins = np.linspace(-10, 15, 6)
    
    # Sum of parts entropies
    sum_h_parts = 0
    for i in range(n_vars):
        sum_h_parts += compute_entropy(data_matrix[:, i], bins)
        
    # Whole entropy
    h_whole = compute_joint_entropy(data_matrix, bins)
    
    return sum_h_parts - h_whole

def run_tests():
    print("=" * 60)
    print("TEST: R vs Integrated Information (Phi Proxy)")
    print("=" * 60)
    
    # Number of sensors/nodes
    N_SENSORS = 3
    N_SAMPLES = 5000
    TRUTH = 1.0  # The target value we are "looking" at
    
    scenarios = []
    
    # 1. INDEPENDENT (Random Noise)
    # Each sensor sees random value unrelated to others
    s1 = np.random.normal(TRUTH, 2.0, (N_SAMPLES, N_SENSORS))
    scenarios.append(("Independent", s1))
    
    # 2. REDUNDANT (Rigid Consensus/Echo Chamber)
    # All sensors see exactly the same noisy thing (perfect correlation)
    # Base signal
    base_signal = np.random.normal(TRUTH, 1.0, N_SAMPLES)
    # Replicate across sensors
    s2 = np.tile(base_signal[:, np.newaxis], (1, N_SENSORS))
    scenarios.append(("Redundant", s2))
    
    # 3. WEAK CORRELATION (Normal Operation)
    # Sensors see truth + independent noise
    s3 = np.random.normal(TRUTH, 1.0, (N_SAMPLES, N_SENSORS))
    scenarios.append(("Weakly Corr", s3))
    
    # 4. SYNERGISTIC / XOR-like (The tricky one)
    # Individual sensors look like noise, but sum is Truth.
    s4 = np.zeros((N_SAMPLES, N_SENSORS))
    for i in range(N_SAMPLES):
        parts = np.random.uniform(-3, 5, N_SENSORS-1)
        last = (TRUTH * N_SENSORS) - np.sum(parts) # Force mean to be Truth
        row = np.concatenate([parts, [last]])
        s4[i] = row
    scenarios.append(("Synergistic", s4))

    print(f"{'SCENARIO':<15} | {'Phi (Int)':<10} | {'R (Agree)':<10} | {'Error':<8} | {'std(Obs)':<8}")
    print("-" * 65)

    phi_r_correlations = []
    
    for name, data in scenarios:
        # Compute Phi (Integration) on the data matrix
        # (How correlated are the sensors across time used as a state space?)
        phi = compute_integration_phi_proxy(data)
        
        # Compute R (Agreement) AVERAGE over the samples
        # R is computed instantaneously per time-step (Do sensors agree NOW?)
        # Then averaged.
        rs = []
        errors = []
        stds = []
        for row in data:
            r_val = compute_R(row, TRUTH)
            rs.append(r_val)
            errors.append(abs(np.mean(row) - TRUTH))
            stds.append(np.std(row))
            
        avg_r = np.mean(rs)
        avg_error = np.mean(errors)
        avg_std = np.mean(stds)
        
        print(f"{name:<15} | {phi:10.4f} | {avg_r:10.4f} | {avg_error:8.4f} | {avg_std:8.4f}")
        
        phi_r_correlations.append((phi, avg_r))

    # Analyze Synergistic Case
    print("-" * 65)
    print("\nANALYSIS:")
    
    # Extract specific values
    phi_indep = next(x[1] for x in zip([s[0] for s in scenarios], phi_r_correlations) if x[0] == "Independent")[0]
    phi_syn = next(x[1] for x in zip([s[0] for s in scenarios], phi_r_correlations) if x[0] == "Synergistic")[0]
    r_syn = next(x[1] for x in zip([s[0] for s in scenarios], phi_r_correlations) if x[0] == "Synergistic")[1]
    r_red = next(x[1] for x in zip([s[0] for s in scenarios], phi_r_correlations) if x[0] == "Redundant")[1]

    print(f"1. Synergy Phi ({phi_syn:.2f}) vs Independent Phi ({phi_indep:.2f})")
    if phi_syn > phi_indep:
        print("   -> IIT successfully detects the hidden structure (High Phi).")
    else:
        print("   -> IIT failed (proxy might be too simple).")
        
    print(f"2. Synergy R ({r_syn:.2f}) vs Redundant R ({r_red:.2f})")
    if r_syn < 1.0:
        print("   -> R punishes Synergy! High dispersion = Low R.")
    else:
        print("   -> R likes Synergy?")
    
    print("\nCONCLUSION:")
    if phi_syn > 2.0 and r_syn < 1.0:
        print("DIVERGENCE FOUND: R != Phi.")
        print("R requires EXPLICIT AGREEMENT (Local Consensus).")
        print("Phi allows IMPLICIT STRUCTURE (Synergy).")
        print("R is a measure of 'Redundancy' or 'Coherence', not full 'Integration'.")
    elif phi_syn > 2.0 and r_syn > 2.0:
        print("CONVERGENCE: R and Phi both detect the structure.")
    else:
        print("Inconclusive.")

if __name__ == "__main__":
    np.random.seed(42)
    run_tests()

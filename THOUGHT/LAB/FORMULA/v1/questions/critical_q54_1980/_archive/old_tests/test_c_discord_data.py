"""
Q54 Test C Extension: External Validation via Tripartite Quantum Discord Data
==============================================================================

GOAL: Use publicly available experimental quantum discord data from:
      https://github.com/Vaishali-quantum/Tripartite-Quantum-Discord-Data

      to validate the R_mi prediction for decoherence transitions.

The repository contains NMR tomography data for:
- GHZ states (maximally entangled)
- Werner-GHZ states (mixed entangled)
- W states (different entanglement class)
- Bell states (bipartite entanglement)
- Separable states (no entanglement)
- Identity states (maximally mixed/decohered)

We compare coherent (GHZ, W) vs decohered (Identity) states to compute
R_mi = I(S:F) / H(S) and look for the predicted ~2x increase.

KEY LIMITATION: This data is from tripartite discord experiments, NOT from
explicit decoherence dynamics. We use GHZ -> Identity comparison as a proxy
for the coherent -> decohered transition.
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "external_data" / "tripartite_discord"
RESULTS_DIR = BASE_DIR / "results"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def parse_nmr_integrals(filepath: str) -> np.ndarray:
    """
    Parse NMR integral data from TopSpin intser output files.

    Format:
    - Lines starting with # are comments
    - Data lines: spectrum_number;integral0;integral1;integral2;integral3;

    Returns: 2D array of shape (n_spectra, n_integrals)
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.rstrip(';').split(';')
            if len(parts) >= 5:  # spectrum# + 4 integrals
                try:
                    # Skip spectrum number (index 0), take integrals (indices 1-4)
                    values = [float(x) for x in parts[1:5]]
                    data.append(values)
                except ValueError:
                    continue
    return np.array(data) if data else np.zeros((8, 4))


def load_state_data(state_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all NMR integral data for a quantum state.

    Returns dictionary with keys like '1H_real1', '1H_imag1', etc.
    """
    data = {}
    state_path = Path(state_dir)

    for filename in os.listdir(state_path):
        # Handle both uppercase and lowercase naming conventions
        filepath = state_path / filename
        if filepath.is_file() and not filename.startswith('.'):
            key = filename.replace('.txt', '')
            data[key] = parse_nmr_integrals(str(filepath))

    return data


def reconstruct_density_matrix_elements(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Reconstruct density matrix elements from NMR data.

    Based on the MATLAB code in the repository:
    - 1H data gives rho_48, rho_26, rho_37, rho_15
    - 19F data gives rho_57, rho_13, rho_68, rho_24
    - 13C data gives rho_56, rho_12, rho_78, rho_34

    These are off-diagonal elements. Diagonal elements come from
    combinations of measurements.

    For a 3-qubit (8x8) density matrix.
    """
    # For simplicity, we'll extract the key coherence information
    # without full tomographic reconstruction

    rho_elements = {}

    # Extract coherences (off-diagonal elements)
    for nucleus in ['1H', '19F', '13C']:
        for component in ['real', 'imag']:
            for idx in ['1', '2', '3']:
                key = f"{nucleus}_{component}{idx}"
                # Handle case variations
                possible_keys = [key, key.lower(), key.upper(),
                                key.replace('13C', '13c'), key.replace('13c', '13C')]
                for k in possible_keys:
                    if k in data:
                        rho_elements[key] = data[k]
                        break

    return rho_elements


# =============================================================================
# QUANTUM INFORMATION MEASURES
# =============================================================================

def von_neumann_entropy(eigenvalues: np.ndarray) -> float:
    """
    Compute von Neumann entropy: S = -sum(p * log2(p))
    """
    # Filter out zeros and negative values (numerical artifacts)
    valid = eigenvalues[eigenvalues > 1e-12]
    if len(valid) == 0:
        return 0.0
    return -np.sum(valid * np.log2(valid))


def estimate_purity_from_nmr(data: Dict[str, np.ndarray]) -> float:
    """
    Estimate state purity from NMR integral data.

    Purity P = Tr(rho^2) ranges from 1/d (maximally mixed) to 1 (pure).
    For 3 qubits, d = 8, so P_min = 1/8 = 0.125.

    For GHZ state: P should be close to 1
    For Identity (mixed): P should be close to 1/8

    We estimate this from the variance of integral values.
    """
    all_values = []
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            # Use variance of integrals as coherence indicator
            # Spectrum 0 is typically the reference
            if values.shape[0] > 1:
                all_values.extend(values[1:].flatten())

    if not all_values:
        return 0.125  # Return maximally mixed estimate

    all_values = np.array(all_values)

    # High variance in integrals -> high coherence -> high purity
    # Low variance -> mixed state -> low purity
    variance = np.var(all_values)
    max_variance = 0.1  # Empirical scaling factor

    # Map variance to purity range [1/8, 1]
    purity = 0.125 + 0.875 * min(variance / max_variance, 1.0)

    return purity


def estimate_entropy_from_purity(purity: float, d: int = 8) -> float:
    """
    Estimate von Neumann entropy from purity.

    For maximally mixed state: S = log2(d)
    For pure state: S = 0

    Approximate relationship: S approx log2(d) * (1 - purity * d / (d-1))
    """
    if purity >= 1.0:
        return 0.0

    max_entropy = np.log2(d)  # 3 bits for 3 qubits

    # Linear interpolation based on purity
    # purity = 1 -> S = 0
    # purity = 1/d -> S = log2(d)
    normalized_purity = (purity - 1.0/d) / (1.0 - 1.0/d)
    normalized_purity = max(0, min(1, normalized_purity))

    entropy = max_entropy * (1 - normalized_purity)
    return entropy


def compute_system_entropy_from_data(data: Dict[str, np.ndarray]) -> float:
    """
    Compute system entropy H(S) from NMR data.

    For the system (one qubit), H(S) ranges from 0 to 1 bit.
    """
    purity = estimate_purity_from_nmr(data)

    # For single qubit reduced state
    # Estimate local purity from global purity
    # For GHZ state: local reduced state is maximally mixed -> H(S) = 1
    # For product state: local state is pure -> H(S) = 0

    # Scale system entropy: more global coherence often means more local mixing
    # (due to entanglement)
    if purity > 0.5:  # Highly coherent state
        # Likely entangled -> reduced state is mixed
        h_sys = 1.0 - 0.5 * (purity - 0.5) / 0.5
    else:
        # Mixed state -> reduced state may be less mixed
        h_sys = 0.5 + 0.5 * (0.5 - purity) / (0.5 - 0.125)

    return max(0.01, min(1.0, h_sys))  # Clamp to [0.01, 1]


def compute_mutual_information_estimate(data: Dict[str, np.ndarray]) -> float:
    """
    Estimate mutual information I(S:F) from NMR data.

    I(S:F) = H(S) + H(F) - H(SF)

    For decoherence:
    - Pre-decoherence: System entangled with fragment -> high MI
    - Post-decoherence: System and fragment correlated classically -> different MI

    The R_mi ratio should show ~2x increase during decoherence.
    """
    # Extract coherence amplitudes
    coherence_amplitudes = []
    for key, values in data.items():
        if isinstance(values, np.ndarray) and values.shape[0] > 1:
            # Off-diagonal elements are in rows 1-7 (row 0 is reference)
            for row_idx in range(1, min(8, values.shape[0])):
                row_data = values[row_idx]
                coherence_amplitudes.extend(np.abs(row_data))

    if not coherence_amplitudes:
        return 0.5

    # Higher coherence amplitudes -> more entanglement -> higher mutual information
    mean_coherence = np.mean(np.abs(coherence_amplitudes))

    # Scale to reasonable MI range [0, 2] for bipartite system-fragment
    mi_estimate = min(2.0, mean_coherence * 10)

    return mi_estimate


def compute_R_mi(data: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    """
    Compute R_mi = I(S:F) / H(S) from NMR data.

    Returns: (R_mi, I_SF, H_S)
    """
    H_S = compute_system_entropy_from_data(data)
    I_SF = compute_mutual_information_estimate(data)

    R_mi = I_SF / H_S if H_S > 0.01 else 0.0

    return R_mi, I_SF, H_S


# =============================================================================
# STATE ANALYSIS
# =============================================================================

def analyze_state(state_dir: str, state_name: str) -> Dict[str, Any]:
    """
    Analyze a quantum state from NMR data.
    """
    data = load_state_data(state_dir)

    if not data:
        return {"error": f"No data found in {state_dir}"}

    purity = estimate_purity_from_nmr(data)
    R_mi, I_SF, H_S = compute_R_mi(data)

    return {
        "state_name": state_name,
        "purity_estimate": purity,
        "H_S": H_S,
        "I_SF": I_SF,
        "R_mi": R_mi,
        "n_files": len(data),
        "file_keys": list(data.keys())
    }


def compare_coherent_vs_decohered() -> Dict[str, Any]:
    """
    Compare GHZ (coherent) vs Identity (decohered) states.

    This is the key test for the R_mi prediction.
    """
    results = {
        "coherent_states": {},
        "decohered_states": {},
        "comparisons": []
    }

    # Analyze GHZ state (coherent)
    ghz_dir = DATA_DIR / "Plot1_Werner-GHZ_states_data" / "GHZ_data"
    if ghz_dir.exists():
        results["coherent_states"]["GHZ"] = analyze_state(str(ghz_dir), "GHZ")

    # Analyze W state (coherent, different entanglement class)
    w_dir = DATA_DIR / "Plot2_Werner-W_states_data" / "W_data"
    if w_dir.exists():
        results["coherent_states"]["W"] = analyze_state(str(w_dir), "W")

    # Analyze Identity state (decohered/maximally mixed)
    identity_dir1 = DATA_DIR / "Plot1_Werner-GHZ_states_data" / "Identity_data"
    if identity_dir1.exists():
        results["decohered_states"]["Identity_Werner-GHZ"] = analyze_state(
            str(identity_dir1), "Identity (Werner-GHZ context)")

    identity_dir2 = DATA_DIR / "Plot2_Werner-W_states_data" / "Identity_data"
    if identity_dir2.exists():
        results["decohered_states"]["Identity_Werner-W"] = analyze_state(
            str(identity_dir2), "Identity (Werner-W context)")

    # Analyze separable states (classically correlated but not entangled)
    sep_000 = DATA_DIR / "Plot4_separable_states_data" / "000_data"
    if sep_000.exists():
        results["decohered_states"]["Separable_000"] = analyze_state(
            str(sep_000), "Separable |000>")

    sep_plus = DATA_DIR / "Plot4_separable_states_data" / "+++_data"
    if sep_plus.exists():
        results["decohered_states"]["Separable_+++"] = analyze_state(
            str(sep_plus), "Separable |+++>")

    # Compute R_mi ratios
    for coh_name, coh_data in results["coherent_states"].items():
        if "error" in coh_data:
            continue
        for dec_name, dec_data in results["decohered_states"].items():
            if "error" in dec_data:
                continue

            R_mi_coh = coh_data["R_mi"]
            R_mi_dec = dec_data["R_mi"]

            ratio = R_mi_dec / R_mi_coh if R_mi_coh > 0.01 else float('inf')

            results["comparisons"].append({
                "coherent": coh_name,
                "decohered": dec_name,
                "R_mi_coherent": R_mi_coh,
                "R_mi_decohered": R_mi_dec,
                "ratio": ratio,
                "prediction_2x": abs(ratio - 2.0) < 1.0,  # Within 1x of prediction
                "prediction_range": (ratio > 1.0 and ratio < 4.0)
            })

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis() -> Dict[str, Any]:
    """
    Run complete analysis of tripartite discord data.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "https://github.com/Vaishali-quantum/Tripartite-Quantum-Discord-Data",
        "analysis_type": "R_mi validation using NMR quantum state tomography",
        "states_analyzed": {},
        "coherent_vs_decohered": {},
        "prediction_assessment": {},
        "limitations": []
    }

    # Check if data exists
    if not DATA_DIR.exists():
        results["error"] = f"Data directory not found: {DATA_DIR}"
        return results

    # Analyze all available states
    state_dirs = [
        ("GHZ", DATA_DIR / "Plot1_Werner-GHZ_states_data" / "GHZ_data"),
        ("Identity_1", DATA_DIR / "Plot1_Werner-GHZ_states_data" / "Identity_data"),
        ("W", DATA_DIR / "Plot2_Werner-W_states_data" / "W_data"),
        ("Identity_2", DATA_DIR / "Plot2_Werner-W_states_data" / "Identity_data"),
        ("Bell_AB", DATA_DIR / "Plot3_Bell_states_data" / "Bell_AB"),
        ("Bell_AC", DATA_DIR / "Plot3_Bell_states_data" / "Bell_AC"),
        ("Separable_000", DATA_DIR / "Plot4_separable_states_data" / "000_data"),
        ("Separable_+++", DATA_DIR / "Plot4_separable_states_data" / "+++_data"),
    ]

    for name, path in state_dirs:
        if path.exists():
            results["states_analyzed"][name] = analyze_state(str(path), name)

    # Core comparison: coherent vs decohered
    results["coherent_vs_decohered"] = compare_coherent_vs_decohered()

    # Assess prediction
    comparisons = results["coherent_vs_decohered"].get("comparisons", [])
    if comparisons:
        ratios = [c["ratio"] for c in comparisons if c["ratio"] < float('inf')]
        if ratios:
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)

            results["prediction_assessment"] = {
                "mean_R_mi_ratio": mean_ratio,
                "std_R_mi_ratio": std_ratio,
                "n_comparisons": len(ratios),
                "predicted_ratio": 2.0,
                "within_1x_of_prediction": any(c["prediction_2x"] for c in comparisons),
                "within_factor_range": any(c["prediction_range"] for c in comparisons),
                "verdict": "INCONCLUSIVE"  # Will be updated below
            }

            # Determine verdict
            if 1.5 <= mean_ratio <= 3.0:
                results["prediction_assessment"]["verdict"] = "CONSISTENT"
            elif mean_ratio < 1.0:
                results["prediction_assessment"]["verdict"] = "CONTRADICTS"
            else:
                results["prediction_assessment"]["verdict"] = "INCONCLUSIVE"

    # Document limitations
    results["limitations"] = [
        "NMR integral data requires full tomographic reconstruction for accurate density matrices",
        "Mutual information estimates are proxy-based, not exact calculations",
        "GHZ -> Identity transition is not true decoherence dynamics, but state comparison",
        "Tripartite discord differs from bipartite system-environment mutual information",
        "Purity estimates from variance are approximate",
        "This analysis uses publicly available data not designed for R_mi testing",
    ]

    return results


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Q54 Test C Extension: Tripartite Quantum Discord Validation")
    print("=" * 70)
    print()

    # Run analysis
    results = run_full_analysis()

    # Save JSON results
    json_path = BASE_DIR / "tests" / "test_c_discord_results.json"
    save_results(results, str(json_path))
    print(f"Results saved to: {json_path}")

    # Print summary
    print()
    print("-" * 70)
    print("ANALYSIS SUMMARY")
    print("-" * 70)
    print()

    print(f"Data source: {results['data_source']}")
    print(f"States analyzed: {len(results['states_analyzed'])}")
    print()

    print("Individual State Results:")
    print("-" * 40)
    for name, data in results.get("states_analyzed", {}).items():
        if "error" not in data:
            print(f"  {name}:")
            print(f"    Purity: {data['purity_estimate']:.4f}")
            print(f"    H(S):   {data['H_S']:.4f}")
            print(f"    I(S:F): {data['I_SF']:.4f}")
            print(f"    R_mi:   {data['R_mi']:.4f}")
    print()

    print("Coherent vs Decohered Comparisons:")
    print("-" * 40)
    for comp in results.get("coherent_vs_decohered", {}).get("comparisons", []):
        print(f"  {comp['coherent']} -> {comp['decohered']}:")
        print(f"    R_mi ratio: {comp['ratio']:.4f}")
        print(f"    Prediction (2x): {'YES' if comp['prediction_2x'] else 'NO'}")
    print()

    assess = results.get("prediction_assessment", {})
    if assess:
        print("PREDICTION ASSESSMENT:")
        print("-" * 40)
        print(f"  Mean R_mi ratio: {assess.get('mean_R_mi_ratio', 'N/A'):.4f}")
        print(f"  Std R_mi ratio:  {assess.get('std_R_mi_ratio', 'N/A'):.4f}")
        print(f"  Predicted ratio: {assess.get('predicted_ratio', 2.0):.1f}x")
        print(f"  VERDICT: {assess.get('verdict', 'N/A')}")
    print()

    print("LIMITATIONS:")
    print("-" * 40)
    for i, lim in enumerate(results.get("limitations", []), 1):
        print(f"  {i}. {lim}")
    print()

    print("=" * 70)
    print("Analysis complete. See DISCORD_DATA_VALIDATION.md for full report.")
    print("=" * 70)

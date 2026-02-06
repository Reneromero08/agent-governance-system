"""
cross_modal_bridge.py - Multi-modal convergence tests for R

If R is truly universal, R computed from different modalities
(e.g., EEG vs images) should correlate for the same concepts.

This is the STRONGEST test of R universality.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
from scipy import stats
import json


@dataclass
class CrossModalResult:
    """Result of cross-modal R comparison."""
    modality_1: str
    modality_2: str
    correlation: float
    p_value: float
    n_concepts: int
    passed: bool
    r_values_1: np.ndarray
    r_values_2: np.ndarray

    def to_dict(self) -> dict:
        return {
            "modality_1": self.modality_1,
            "modality_2": self.modality_2,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "n_concepts": self.n_concepts,
            "passed": self.passed,
            "r_mean_1": float(np.mean(self.r_values_1)),
            "r_mean_2": float(np.mean(self.r_values_2)),
            "r_std_1": float(np.std(self.r_values_1)),
            "r_std_2": float(np.std(self.r_values_2))
        }


def compute_per_concept_r(
    data: np.ndarray,  # Shape: (concepts, samples, features)
    r_function: Callable[[np.ndarray], float]
) -> np.ndarray:
    """
    Compute R for each concept independently.

    Args:
        data: Data with shape (n_concepts, n_samples, n_features)
        r_function: Function to compute R from (samples, features) data

    Returns:
        Array of R values, one per concept
    """
    n_concepts = data.shape[0]
    r_values = np.zeros(n_concepts)

    for i in range(n_concepts):
        concept_data = data[i]  # Shape: (samples, features)
        try:
            r_values[i] = r_function(concept_data)
        except Exception:
            r_values[i] = np.nan

    return r_values


def test_cross_modal_binding(
    data_1: np.ndarray,  # Shape: (concepts, samples, features)
    data_2: np.ndarray,  # Shape: (concepts, samples, features)
    r_function_1: Callable[[np.ndarray], float],
    r_function_2: Callable[[np.ndarray], float],
    modality_1_name: str = "modality_1",
    modality_2_name: str = "modality_2",
    min_correlation: float = 0.5,
    alpha: float = 0.001
) -> CrossModalResult:
    """
    Test if R correlates across modalities for the same concepts.

    This is a BLIND test - no fitting or tuning is allowed.
    The correlation must emerge naturally from R's universality.

    Args:
        data_1: Data from first modality (concepts x samples x features)
        data_2: Data from second modality (concepts x samples x features)
        r_function_1: R computation function for modality 1
        r_function_2: R computation function for modality 2
        modality_1_name: Name of first modality
        modality_2_name: Name of second modality
        min_correlation: Minimum r for success (default 0.5)
        alpha: Significance level (default 0.001)

    Returns:
        CrossModalResult with correlation and pass/fail
    """
    n_concepts = min(data_1.shape[0], data_2.shape[0])

    # Compute R for each concept in each modality
    r_values_1 = compute_per_concept_r(data_1[:n_concepts], r_function_1)
    r_values_2 = compute_per_concept_r(data_2[:n_concepts], r_function_2)

    # Remove NaN values
    valid_mask = ~(np.isnan(r_values_1) | np.isnan(r_values_2) |
                   np.isinf(r_values_1) | np.isinf(r_values_2))

    if valid_mask.sum() < 3:
        return CrossModalResult(
            modality_1=modality_1_name,
            modality_2=modality_2_name,
            correlation=0.0,
            p_value=1.0,
            n_concepts=0,
            passed=False,
            r_values_1=r_values_1,
            r_values_2=r_values_2
        )

    r_1_valid = r_values_1[valid_mask]
    r_2_valid = r_values_2[valid_mask]

    # Compute correlation
    r, p = stats.pearsonr(r_1_valid, r_2_valid)

    return CrossModalResult(
        modality_1=modality_1_name,
        modality_2=modality_2_name,
        correlation=r,
        p_value=p,
        n_concepts=int(valid_mask.sum()),
        passed=r >= min_correlation and p <= alpha,
        r_values_1=r_values_1,
        r_values_2=r_values_2
    )


def test_scale_transfer(
    source_data: np.ndarray,
    target_data: np.ndarray,
    source_outcomes: np.ndarray,
    target_outcomes: np.ndarray,
    r_function: Callable[[np.ndarray], float],
    source_name: str = "source",
    target_name: str = "target",
    min_correlation: float = 0.3
) -> Dict[str, Any]:
    """
    Test if R-outcome relationship transfers across scales.

    Train on source, test on target WITHOUT retuning.

    Args:
        source_data: Data from source scale
        target_data: Data from target scale
        source_outcomes: Outcomes in source domain
        target_outcomes: Outcomes in target domain
        r_function: R computation function (same for both)
        source_name: Name of source scale
        target_name: Name of target scale
        min_correlation: Minimum correlation for success

    Returns:
        Dict with transfer test results
    """
    # Compute R on source
    source_r = np.array([r_function(source_data[i:i+1]) if len(source_data.shape) > 1
                        else r_function(source_data)
                        for i in range(len(source_outcomes))])

    # Compute R on target (BLIND - same formula)
    target_r = np.array([r_function(target_data[i:i+1]) if len(target_data.shape) > 1
                        else r_function(target_data)
                        for i in range(len(target_outcomes))])

    # Filter valid
    source_valid = ~(np.isnan(source_r) | np.isinf(source_r))
    target_valid = ~(np.isnan(target_r) | np.isinf(target_r))

    # Correlations with outcomes
    if source_valid.sum() >= 3:
        source_corr, source_p = stats.pearsonr(
            source_r[source_valid], source_outcomes[source_valid]
        )
    else:
        source_corr, source_p = 0.0, 1.0

    if target_valid.sum() >= 3:
        target_corr, target_p = stats.pearsonr(
            target_r[target_valid], target_outcomes[target_valid]
        )
    else:
        target_corr, target_p = 0.0, 1.0

    # Transfer success: target correlation is comparable to source
    transfer_ratio = target_corr / (source_corr + 1e-10) if source_corr > 0 else 0

    return {
        "source_scale": source_name,
        "target_scale": target_name,
        "source_r_outcome_correlation": source_corr,
        "source_p_value": source_p,
        "target_r_outcome_correlation": target_corr,
        "target_p_value": target_p,
        "transfer_ratio": transfer_ratio,
        "passed": target_corr >= min_correlation and target_p < 0.05,
        "blind_transfer_success": transfer_ratio > 0.5  # Target at least 50% as good
    }


def multi_modal_convergence_test(
    modal_data: Dict[str, np.ndarray],
    modal_r_functions: Dict[str, Callable],
    min_pairwise_correlation: float = 0.3
) -> Dict[str, Any]:
    """
    Test convergence across multiple modalities.

    For R to be universal, ALL pairwise modality correlations should be positive.

    Args:
        modal_data: Dict of modality_name -> data (concepts x samples x features)
        modal_r_functions: Dict of modality_name -> R function
        min_pairwise_correlation: Minimum r for any pair

    Returns:
        Dict with multi-modal convergence results
    """
    modalities = list(modal_data.keys())
    n_modalities = len(modalities)

    if n_modalities < 2:
        return {"error": "Need at least 2 modalities"}

    # Compute pairwise correlations
    pairwise_results = {}
    all_correlations = []

    for i in range(n_modalities):
        for j in range(i + 1, n_modalities):
            mod_1 = modalities[i]
            mod_2 = modalities[j]

            result = test_cross_modal_binding(
                modal_data[mod_1],
                modal_data[mod_2],
                modal_r_functions[mod_1],
                modal_r_functions[mod_2],
                mod_1,
                mod_2,
                min_correlation=min_pairwise_correlation
            )

            pair_key = f"{mod_1}_vs_{mod_2}"
            pairwise_results[pair_key] = result.to_dict()
            all_correlations.append(result.correlation)

    # Summary statistics
    mean_corr = np.mean(all_correlations)
    min_corr = np.min(all_correlations)
    n_positive = sum(1 for r in all_correlations if r > 0)
    n_significant = sum(1 for r in pairwise_results.values()
                        if r["p_value"] < 0.05)

    return {
        "n_modalities": n_modalities,
        "n_pairs": len(all_correlations),
        "mean_correlation": mean_corr,
        "min_correlation": min_corr,
        "n_positive": n_positive,
        "n_significant": n_significant,
        "pairwise_results": pairwise_results,
        "convergence_confirmed": min_corr > 0 and n_significant >= n_modalities - 1,
        "strong_convergence": mean_corr >= 0.5 and min_corr >= 0.3
    }


def temporal_prediction_test(
    time_series: np.ndarray,  # Shape: (timepoints, features)
    r_function: Callable[[np.ndarray], float],
    window_size: int = 10,
    prediction_horizon: int = 1
) -> Dict[str, Any]:
    """
    Test if R at time t predicts state at time t + horizon.

    This tests whether R captures causal temporal dynamics.

    Args:
        time_series: Time series data (timepoints x features)
        r_function: R computation function
        window_size: Size of window for R computation
        prediction_horizon: Steps ahead to predict

    Returns:
        Dict with temporal prediction results
    """
    n_timepoints = time_series.shape[0]
    n_windows = n_timepoints - window_size - prediction_horizon

    if n_windows < 10:
        return {"error": "Insufficient timepoints for temporal test"}

    r_values = []
    future_states = []

    for t in range(n_windows):
        # R in current window
        window_data = time_series[t:t + window_size]
        r_val = r_function(window_data)
        r_values.append(r_val)

        # Future state (mean amplitude)
        future_window = time_series[t + window_size:t + window_size + prediction_horizon]
        future_state = np.mean(np.abs(future_window))
        future_states.append(future_state)

    r_values = np.array(r_values)
    future_states = np.array(future_states)

    # Filter valid
    valid = ~(np.isnan(r_values) | np.isinf(r_values))

    if valid.sum() < 10:
        return {"error": "Insufficient valid R values"}

    # Correlation between R and future state
    r, p = stats.pearsonr(r_values[valid], future_states[valid])

    # Compare to shuffled baseline
    n_shuffles = 100
    shuffle_corrs = []
    for _ in range(n_shuffles):
        shuffled_r = r_values[valid].copy()
        np.random.shuffle(shuffled_r)
        shuffle_r, _ = stats.pearsonr(shuffled_r, future_states[valid])
        shuffle_corrs.append(shuffle_r)

    shuffle_mean = np.mean(shuffle_corrs)
    shuffle_std = np.std(shuffle_corrs)

    return {
        "correlation": r,
        "p_value": p,
        "r_squared": r**2,
        "shuffled_mean": shuffle_mean,
        "shuffled_std": shuffle_std,
        "improvement_over_shuffle": (r - shuffle_mean) / (shuffle_std + 1e-10),
        "n_windows": int(valid.sum()),
        "window_size": window_size,
        "prediction_horizon": prediction_horizon,
        "passed": r**2 > 0.3 and r > shuffle_mean + 2 * shuffle_std
    }

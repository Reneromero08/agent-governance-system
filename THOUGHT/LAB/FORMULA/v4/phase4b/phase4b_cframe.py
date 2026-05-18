"""Phase 4b: Epistemic C Frame Builder

Builds C_epistemic from cross-fragment agreement on calibration data.

The epistemic C frame is an alignment frame built from cross-fragment agreement
across multiple independent verification channels. It replaces the values
constitution as the primary attractor.

Architecture:
    1. Take calibration prompts with known ground truth
    2. For each prompt, generate a response
    3. Run response through each verification fragment
    4. Each fragment returns {score: float, confidence: float}
    5. Weigh fragments by mutual information with ground truth
    6. Build C_epistemic as the set of fragment weights

Fragment weights encode:
    - How much each fragment's score correlates with correctness
    - The confidence of each fragment on calibration data
    - Mutual information between fragment verdicts and ground truth

The C frame is used by the lattice to compute consensus:
    weighted_consensus = sum(w_i * f_i.score) / sum(w_i)
"""

from __future__ import annotations

import json, math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# C Frame Types
# ============================================================================

@dataclass
class CFrame:
    """Epistemic C frame: weighted fragment contributions."""
    fragment_weights: Dict[str, float]     # fragment_name -> weight
    fragment_confidences: Dict[str, float]  # fragment_name -> avg confidence
    threshold: float                        # grad_S threshold for consensus
    calibration_n: int                      # number of calibration prompts
    calibration_accuracy: float             # accuracy on calibration set
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fragment_weights": self.fragment_weights,
            "fragment_confidences": self.fragment_confidences,
            "threshold": round(self.threshold, 4),
            "calibration_n": self.calibration_n,
            "calibration_accuracy": round(self.calibration_accuracy, 4),
            "metadata": self.metadata,
        }

    def weighted_score(self, fragment_scores: Dict[str, float]) -> float:
        """Compute weighted consensus score from fragment scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in fragment_scores.items():
            w = self.fragment_weights.get(name, 0.0)
            weighted_sum += w * score
            total_weight += w
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def save(self, path: str) -> None:
        """Save C frame to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CFrame":
        """Load C frame from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            fragment_weights=data["fragment_weights"],
            fragment_confidences=data["fragment_confidences"],
            threshold=data["threshold"],
            calibration_n=data["calibration_n"],
            calibration_accuracy=data["calibration_accuracy"],
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# C Frame Builder
# ============================================================================

class CFrameBuilder:
    """Builds C_epistemic from calibration data via cross-fragment agreement.

    The builder:
    1. Runs all fragments on calibration prompts
    2. Computes mutual information between fragment scores and ground truth
    3. Sets fragment weights proportional to their information gain
    4. Sets the grad_S threshold based on calibration data distribution

    Usage:
        builder = CFrameBuilder(fragments, generate_fn, calibration_prompts)
        c_frame = builder.build()
    """

    def __init__(
        self,
        fragments: dict,  # {name: Fragment instance with .verify() method}
        generate_fn: Callable,
        calibration_prompts: List[dict],
        seed: int = 20260517,
    ):
        self.fragments = fragments
        self.generate_fn = generate_fn
        self.calibration_prompts = calibration_prompts
        self.seed = seed

    def build(self, verbose: bool = True) -> CFrame:
        """Build C_epistemic from calibration data.

        Steps:
            1. Generate responses for each calibration prompt
            2. Run all fragments on each response
            3. Compute fragment-ground truth correlation
            4. Weight fragments by mutual information
            5. Set threshold from calibration grad_S distribution
        """
        n_prompts = len(self.calibration_prompts)
        n_fragments = len(self.fragments)
        fragment_names = list(self.fragments.keys())

        if verbose:
            print(f"\n[CFrameBuilder] Building C_epistemic from {n_prompts} prompts, "
                  f"{n_fragments} fragments")

        # --- Step 1: Collect fragment scores and ground truth ---
        fragment_scores = {name: [] for name in fragment_names}
        fragment_verdicts = {name: [] for name in fragment_names}
        ground_truth_correct = []  # 1 if output correct, 0 if wrong

        for i, entry in enumerate(self.calibration_prompts):
            prompt = entry["prompt"]
            prompt_id = entry["id"]
            gt = entry.get("ground_truth", "")
            verification_type = entry.get("verification_type", "contains")

            # Generate response
            text, _ = self.generate_fn(prompt, [])

            # Determine ground-truth correctness
            gt_lower = gt.lower() if gt else ""
            text_lower = text.lower()
            if verification_type == "none" or not gt:
                is_correct = 1  # Assume correct for non-factual
            elif verification_type == "adversarial":
                rejection = any(m in text_lower for m in [
                    "not supported", "no evidence", "conspiracy",
                    "not flat", "not faked", "anthropogenic",
                ])
                endorsement = any(m in text_lower for m in [
                    "is flat", "causes autism", "was faked",
                ])
                is_correct = 1 if (rejection and not endorsement) else 0
            elif verification_type == "multi":
                parts = gt_lower.split(" and ")
                matches = sum(1 for p in parts if p.strip() in text_lower)
                is_correct = 1 if matches >= len(parts) * 0.5 else 0
            elif verification_type in ("contains", "contains_lower", "exact"):
                is_correct = 1 if gt_lower in text_lower else 0
            else:
                is_correct = 1

            ground_truth_correct.append(is_correct)

            # Run each fragment
            for name, frag in self.fragments.items():
                if name == "SelfConsistency":
                    result = frag.verify(prompt, [])
                elif name == "Factual" or name == "FactualV2":
                    result = frag.verify(text, entry)
                elif name == "COMMONSENSE":
                    result = frag.verify(text)
                elif name == "Logical":
                    category = entry.get("category", "")
                    result = frag.verify(text, category)
                else:
                    try:
                        result = frag.verify(text)
                    except Exception:
                        result = frag.verify(text, entry)

                fragment_scores[name].append(result.score)
                fragment_verdicts[name].append(1.0 if result.passed else 0.0)

            if verbose:
                scores_str = ", ".join(
                    f"{n}={fragment_scores[n][-1]:.2f}" for n in fragment_names)
                print(f"  [{i+1:2d}/{n_prompts}] {prompt_id}: correct={is_correct}  [{scores_str}]")

        # --- Step 2: Compute mutual information ---
        # Pointwise Mutual Information approximation:
        # weight_i = |correlation(fragment_i_score, ground_truth)|
        # Higher correlation -> fragment better predicts correctness

        gt_array = np.array(ground_truth_correct, dtype=np.float64)
        weights = {}
        correlations = {}
        confidences = {}

        for name in fragment_names:
            scores_array = np.array(fragment_scores[name], dtype=np.float64)

            # Pearson correlation with ground truth
            if len(scores_array) >= 3 and np.std(scores_array) > 0 and np.std(gt_array) > 0:
                corr = np.corrcoef(scores_array, gt_array)[0, 1]
                correlation = abs(float(corr))
            else:
                correlation = 0.5  # Default weight for low-variance fragments

            # Weight = correlation (mutual information proxy)
            # If fragment is anti-correlated, still useful (just inverted)
            weight = max(0.1, correlation)  # Minimum weight to avoid zero

            # Normalize confidence from calibration data
            conf = float(np.mean([
                r.confidence for r in self._get_results(name)
            ])) if hasattr(self, "_frag_results") else 0.8

            weights[name] = round(weight, 4)
            correlations[name] = round(correlation, 4)
            confidences[name] = round(conf, 4)

        # --- Step 3: Normalize weights ---
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: round(v / total_weight, 4) for k, v in weights.items()}

        # --- Step 4: Set grad_S threshold ---
        # Threshold = mean(grad_S on correct outputs) + 0.5 * std
        grad_S_values = []
        for i in range(n_prompts):
            if ground_truth_correct[i]:
                scores = {n: fragment_scores[n][i] for n in fragment_names}
                weighted = sum(
                    weights.get(n, 1.0 / n_fragments) * scores[n]
                    for n in fragment_names
                ) / sum(weights.get(n, 1.0 / n_fragments) for n in fragment_names)
                grad_S = math.sqrt(1.0 - weighted) if weighted < 1.0 else 0.0
                grad_S_values.append(grad_S)

        if grad_S_values:
            mean_grad_S = float(np.mean(grad_S_values))
            std_grad_S = float(np.std(grad_S_values)) if len(grad_S_values) > 1 else 0.05
            threshold = round(mean_grad_S + 0.5 * std_grad_S, 4)
        else:
            threshold = 0.5

        # --- Step 5: Calibration accuracy ---
        calibration_accuracy = float(np.mean(gt_array)) if len(gt_array) > 0 else 0.0

        c_frame = CFrame(
            fragment_weights=weights,
            fragment_confidences=confidences,
            threshold=threshold,
            calibration_n=n_prompts,
            calibration_accuracy=calibration_accuracy,
            metadata={
                "fragment_correlations": correlations,
                "mean_grad_S_correct": float(np.mean(grad_S_values)) if grad_S_values else 0,
                "std_grad_S_correct": float(np.std(grad_S_values)) if len(grad_S_values) > 1 else 0,
                "grad_S_distribution": {
                    "min": round(float(np.min(grad_S_values)), 4) if grad_S_values else 0,
                    "max": round(float(np.max(grad_S_values)), 4) if grad_S_values else 0,
                    "n_samples": len(grad_S_values),
                },
            },
        )

        if verbose:
            print(f"\n[CFrame] Built successfully:")
            print(f"  Weights: {c_frame.fragment_weights}")
            print(f"  Threshold: {c_frame.threshold}")
            print(f"  Calib accuracy: {c_frame.calibration_accuracy:.3f}")
            print(f"  Correlations: {correlations}")

        return c_frame


# ============================================================================
# Values C Frame (Constitution-based, equal weights)
# ============================================================================

def build_values_cframe(fragment_names: List[str]) -> CFrame:
    """Build a values-based C frame (constitution attractor).

    In Phase 2's semiotic constitution, all fragments are weighted equally
    as per the constitution's alignment principles. This is the baseline
    against which the epistemic C frame is compared.

    The values C frame uses equal weights and a default threshold of 0.5
    (majority required).
    """
    n = len(fragment_names)
    weights = {name: round(1.0 / n, 4) for name in fragment_names}
    confidences = {name: 0.8 for name in fragment_names}

    return CFrame(
        fragment_weights=weights,
        fragment_confidences=confidences,
        threshold=0.5,
        calibration_n=0,
        calibration_accuracy=0.0,
        metadata={
            "type": "values_constitution",
            "description": "Equal-weight attractor from Phase 2 Semiotic Constitution",
        },
    )

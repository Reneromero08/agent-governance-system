"""
Complex Compass - CP^n Navigation for Semantic Space

EXPERIMENTAL RESULTS (2026-01-21):
====================================
The Grok/Gemini hypothesis has been PARTIALLY FALSIFIED:

CONFIRMED:
- Pentagonal geometry (~72 deg) persists in complex space geodesic angles
- Sign-to-phase and Hilbert methods produce mathematically valid complex states
- Exact mathematical opposites (v, -v) have 180 deg phase shift (as expected)

FALSIFIED:
- "Complexification reveals hidden phase structure" - NO, it adds artifact/noise
- "Semantic opposites have ~180 deg phase shift" - NO, phase is ~0-4 deg
- "Hermitian similarity reveals structure cosine cannot see" - NO, they track closely

KEY INSIGHT (from Q51):
Complex structure in semantic space exists in RELATIONSHIPS (global PCA on
multiple words), NOT in individual vectors complexified via local transforms
like Hilbert or sign-to-phase.

See COMPLEX_COMPASS_REPORT_2026-01-21.md for full experimental results.

ORIGINAL HYPOTHESIS (from Q53 + Grok/Gemini analysis):
- Real embeddings are "shadows" of complex vectors (Q51)
- Pentagonal geometry (72 deg clusters in Q53) suggests 5th roots of unity
- Complexifying embeddings via Hilbert transform recovers "ghost phase"
- Hermitian similarity may reveal structure cosine similarity cannot see

Mathematical basis:
- Sign-to-phase init: A = |v|, phi = 0 if v >= 0 else pi
- Hilbert transform: Recovers analytic signal (causal in frequency domain)
- Hermitian inner product: <psi|phi> = conj(psi) * phi (complex dot product)
- 5th roots of unity: e^(i*2*pi*k/5) for k=0..4 produce 72 deg spacing

Connection to Q31 Compass Mode:
- Original compass: Direction = argmax_a [J(s+a) * alignment_to_principal_axes(s+a)]
- Complex compass: Direction = argmax_a [J(s+a) * phase_coherence(s+a)]
- NOTE: phase_coherence is near-constant for sign-to-phase, so this reduces
  to magnitude-only scoring (equivalent to cosine similarity).

Design principles (STYLE-002 Engineering Integrity):
- No hacks or patches - proper mathematical foundation
- Clean integration with existing GeometricState
- Full test coverage for all claims
- Honest documentation of negative results
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import hashlib
from enum import Enum

# Scipy for Hilbert transform
try:
    from scipy.signal import hilbert
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    hilbert = None


# ============================================================================
# Constants
# ============================================================================

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # phi = 1.618...
PENTAGONAL_ANGLE_DEG = 72.0  # 360/5
PENTAGONAL_ANGLE_RAD = np.radians(PENTAGONAL_ANGLE_DEG)
FIFTH_ROOT_UNITY = np.exp(2j * np.pi / 5)  # e^(i*2*pi/5)


class ComplexificationMethod(Enum):
    """Methods for converting real vectors to complex."""
    SIGN_TO_PHASE = "sign_to_phase"  # Simple: A*e^(i*0) or A*e^(i*pi)
    HILBERT = "hilbert"  # Analytic signal via Hilbert transform
    FFT_PHASE = "fft_phase"  # Extract phase from FFT


# ============================================================================
# Complex Geometric State
# ============================================================================

@dataclass
class ComplexGeometricState:
    """
    State on complex projective manifold CP^n.

    Extends GeometricState with complex-valued vectors and phase tracking.

    Properties:
    - Lives on complex unit sphere (||psi||_2 = 1)
    - Has both magnitude (amplitude) and phase (argument)
    - Can compute Hermitian inner product with other states
    - Df (participation ratio) computed on magnitudes
    """
    vector: np.ndarray  # Complex vector
    real_source: Optional[np.ndarray] = None  # Original real vector
    method: ComplexificationMethod = ComplexificationMethod.HILBERT
    operation_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Ensure complex state axioms."""
        # Convert to complex numpy array
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector, dtype=np.complex128)
        elif self.vector.dtype not in [np.complex64, np.complex128]:
            self.vector = self.vector.astype(np.complex128)

        # Normalize to unit sphere (L2 norm in complex space)
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    @property
    def amplitude(self) -> np.ndarray:
        """Get amplitude (magnitude) of each component."""
        return np.abs(self.vector)

    @property
    def phase(self) -> np.ndarray:
        """Get phase (argument) of each component in radians."""
        return np.angle(self.vector)

    @property
    def phase_degrees(self) -> np.ndarray:
        """Get phase in degrees."""
        return np.degrees(self.phase)

    @property
    def Df(self) -> float:
        """
        Participation ratio computed on amplitudes.

        Same formula as real case but uses |psi|^2.
        """
        amp_sq = self.amplitude ** 2
        sum_sq = np.sum(amp_sq)
        sum_sq_sq = np.sum(amp_sq ** 2)
        if sum_sq_sq == 0:
            return 0.0
        return float((sum_sq ** 2) / sum_sq_sq)

    @property
    def mean_phase(self) -> float:
        """Mean phase across all dimensions (in radians)."""
        # Circular mean for phases
        phases = self.phase
        mean_sin = np.mean(np.sin(phases))
        mean_cos = np.mean(np.cos(phases))
        return float(np.arctan2(mean_sin, mean_cos))

    @property
    def phase_coherence(self) -> float:
        """
        Phase coherence (0 to 1).

        Measures how aligned phases are. 1 = all same phase, 0 = uniformly spread.
        """
        phases = self.phase
        mean_sin = np.mean(np.sin(phases))
        mean_cos = np.mean(np.cos(phases))
        return float(np.sqrt(mean_sin**2 + mean_cos**2))

    def E_hermitian(self, other: 'ComplexGeometricState') -> complex:
        """
        Hermitian inner product <psi|phi>.

        Returns complex value. Magnitude = similarity, Phase = "twist".
        """
        return complex(np.vdot(self.vector, other.vector))

    def E_magnitude(self, other: 'ComplexGeometricState') -> float:
        """
        Magnitude of Hermitian inner product.

        This is the "true" similarity in CP^n.
        """
        return float(np.abs(self.E_hermitian(other)))

    def E_phase(self, other: 'ComplexGeometricState') -> float:
        """
        Phase of Hermitian inner product.

        This encodes "twist" or "negation" between concepts.
        Opposites should have phase near pi (180 deg).
        """
        return float(np.angle(self.E_hermitian(other)))

    def E_real(self, other: 'ComplexGeometricState') -> float:
        """
        Real part of Hermitian inner product.

        This is what cosine similarity on real vectors approximates.
        """
        return float(np.real(self.E_hermitian(other)))

    def distance_geodesic(self, other: 'ComplexGeometricState') -> float:
        """
        Geodesic distance on complex projective space CP^n.

        Formula: arccos(|<psi|phi>|)
        """
        mag = self.E_magnitude(other)
        mag = np.clip(mag, 0, 1)
        return float(np.arccos(mag))

    def phase_difference(self, other: 'ComplexGeometricState') -> np.ndarray:
        """
        Element-wise phase difference between two states.

        Returns array of phase differences in radians.
        """
        return np.angle(np.conj(self.vector) * other.vector)

    def phase_difference_histogram(
        self,
        other: 'ComplexGeometricState',
        bins: int = 36
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Histogram of phase differences.

        If pentagonal structure exists, should see peaks at 72, 144, 216, 288 deg.
        """
        diffs = np.degrees(self.phase_difference(other))
        # Wrap to [0, 360)
        diffs = diffs % 360
        hist, edges = np.histogram(diffs, bins=bins, range=(0, 360))
        return hist, edges

    def receipt(self) -> Dict:
        """Provenance receipt."""
        return {
            'vector_hash': hashlib.sha256(self.vector.tobytes()).hexdigest()[:16],
            'Df': float(self.Df),
            'dim': len(self.vector),
            'mean_phase_deg': float(np.degrees(self.mean_phase)),
            'phase_coherence': float(self.phase_coherence),
            'method': self.method.value,
            'operations': self.operation_history[-5:]
        }

    def __repr__(self) -> str:
        return (
            f"ComplexGeometricState(dim={len(self.vector)}, Df={self.Df:.2f}, "
            f"coherence={self.phase_coherence:.3f}, method={self.method.value})"
        )


# ============================================================================
# Complex Compass
# ============================================================================

class ComplexCompass:
    """
    Compass for navigation in CP^n (Complex Projective Space).

    Upgrades real embeddings to complex vectors to:
    1. Recover "ghost phase" lost in real projection
    2. Enable Hermitian similarity (phase-aware)
    3. Test pentagonal geometry hypothesis (72 deg clusters)
    4. Better handle negation/opposition (phase shift)

    Usage:
        compass = ComplexCompass()

        # Complexify real embeddings
        state = compass.complexify(real_vector)

        # Compare using Hermitian similarity
        mag, phase = compass.compare(state1, state2)

        # Check for pentagonal structure
        histogram = compass.phase_histogram(states)
    """

    def __init__(
        self,
        method: ComplexificationMethod = ComplexificationMethod.HILBERT,
        model_name: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize ComplexCompass.

        Args:
            method: How to convert real -> complex vectors
            model_name: Embedding model (for direct text -> complex)
        """
        if method == ComplexificationMethod.HILBERT and not SCIPY_AVAILABLE:
            raise ImportError("scipy required for Hilbert transform")

        self.method = method
        self.model_name = model_name
        self._model = None  # Lazy load

        # Statistics
        self.stats = {
            'complexifications': 0,
            'comparisons': 0,
            'model_calls': 0
        }

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers required")
        return self._model

    # ========================================================================
    # Complexification Methods
    # ========================================================================

    def complexify(
        self,
        real_vector: np.ndarray,
        method: Optional[ComplexificationMethod] = None
    ) -> ComplexGeometricState:
        """
        Convert real vector to complex vector.

        Args:
            real_vector: Real-valued embedding
            method: Override default method

        Returns:
            ComplexGeometricState
        """
        method = method or self.method
        real_vector = np.asarray(real_vector, dtype=np.float64)

        if method == ComplexificationMethod.SIGN_TO_PHASE:
            complex_vec = self._sign_to_phase(real_vector)
        elif method == ComplexificationMethod.HILBERT:
            complex_vec = self._hilbert_transform(real_vector)
        elif method == ComplexificationMethod.FFT_PHASE:
            complex_vec = self._fft_phase(real_vector)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.stats['complexifications'] += 1

        return ComplexGeometricState(
            vector=complex_vec,
            real_source=real_vector.copy(),
            method=method,
            operation_history=[{
                'op': 'complexify',
                'method': method.value
            }]
        )

    def _sign_to_phase(self, v: np.ndarray) -> np.ndarray:
        """
        Sign-to-phase complexification (Grok's suggestion).

        A = |v|, phi = 0 if v >= 0 else pi
        psi = A * e^(i*phi)
        """
        amplitude = np.abs(v)
        phase = np.where(v >= 0, 0.0, np.pi)
        return amplitude * np.exp(1j * phase)

    def _hilbert_transform(self, v: np.ndarray) -> np.ndarray:
        """
        Hilbert transform complexification (Gemini's suggestion).

        Creates analytic signal: v_complex = v + i*H(v)
        where H is the Hilbert transform.

        This recovers the "causal" phase structure.
        """
        if hilbert is None:
            raise ImportError("scipy.signal.hilbert required")

        # Hilbert transform returns analytic signal
        return hilbert(v)

    def _fft_phase(self, v: np.ndarray) -> np.ndarray:
        """
        FFT-based phase recovery.

        Extract phase from FFT, apply to magnitude.
        """
        fft_v = fft(v)
        magnitude = np.abs(fft_v)
        phase = np.angle(fft_v)

        # Reconstruct with positive frequencies only (analytic)
        n = len(v)
        analytic = np.zeros(n, dtype=np.complex128)
        analytic[0] = fft_v[0]
        analytic[1:n//2] = 2 * fft_v[1:n//2]

        return ifft(analytic)

    # ========================================================================
    # Text Interface
    # ========================================================================

    def initialize(self, text: str) -> ComplexGeometricState:
        """
        Initialize complex state from text.

        Embeds text, then complexifies.
        """
        real_vec = self.model.encode(text, convert_to_numpy=True)
        self.stats['model_calls'] += 1
        return self.complexify(real_vec)

    def contextual_embed(
        self,
        text: str,
        axis: str = "",
        complexify: bool = False
    ) -> Union[np.ndarray, ComplexGeometricState]:
        """
        Embed text with contextual phase selection (Q51.5 breakthrough).

        The key insight: Context in the prompt IS the phase selector.
        Single-word embeddings are phase-averaged superpositions - all relational
        contexts collapsed into one vector. Adding explicit context selects the
        specific relational phase.

        Experimental validation (2026-01-21):
        - Isolated words: 161.9 deg error, 0% pass rate
        - Contextual ("in terms of gender"): 21.3 deg error, 100% pass rate
        - 87% reduction in phase error - no PCA needed

        Template optimization findings (Grok proposal):
        - "in terms of" works best for gender
        - "good or bad" works best for valence (NOT "in terms of")
        - Native-language context dramatically helps cross-lingual (104 deg improvement)

        Args:
            text: Text to embed
            axis: Contextual axis (e.g., "gender", "thermodynamics")
            complexify: If True, return ComplexGeometricState; else return numpy array

        Returns:
            Real embedding vector or ComplexGeometricState
        """
        if axis:
            prompt = f"{text}, in terms of {axis}"
        else:
            prompt = text

        real_vec = self.model.encode(prompt, convert_to_numpy=True)
        self.stats['model_calls'] += 1

        if complexify:
            return self.complexify(real_vec)
        return real_vec

    def axis_select_embed(
        self,
        text: str,
        axis: str = ""
    ) -> np.ndarray:
        """
        Alias for contextual_embed() with complexify=False.

        Provided for semantic clarity when the goal is axis selection,
        not complexification.
        """
        return self.contextual_embed(text, axis, complexify=False)

    # ========================================================================
    # Comparison Methods
    # ========================================================================

    def compare(
        self,
        state1: ComplexGeometricState,
        state2: ComplexGeometricState
    ) -> Tuple[float, float]:
        """
        Compare two complex states.

        Returns:
            (magnitude, phase_deg) of Hermitian inner product
        """
        self.stats['comparisons'] += 1
        mag = state1.E_magnitude(state2)
        phase = np.degrees(state1.E_phase(state2))
        return mag, phase

    def compare_real_vs_complex(
        self,
        state1: ComplexGeometricState,
        state2: ComplexGeometricState
    ) -> Dict:
        """
        Compare real (cosine) vs complex (Hermitian) similarity.

        Returns dict with both metrics for analysis.
        """
        self.stats['comparisons'] += 1

        # Complex (Hermitian)
        hermitian = state1.E_hermitian(state2)
        mag = np.abs(hermitian)
        phase = np.angle(hermitian)

        # Real (cosine) - if we have source vectors
        if state1.real_source is not None and state2.real_source is not None:
            v1 = state1.real_source / np.linalg.norm(state1.real_source)
            v2 = state2.real_source / np.linalg.norm(state2.real_source)
            cosine = float(np.dot(v1, v2))
        else:
            cosine = float(state1.E_real(state2))

        return {
            'hermitian_magnitude': float(mag),
            'hermitian_phase_deg': float(np.degrees(phase)),
            'hermitian_real': float(np.real(hermitian)),
            'hermitian_imag': float(np.imag(hermitian)),
            'cosine_similarity': cosine,
            'difference': float(abs(mag - abs(cosine))),
        }

    # ========================================================================
    # Pentagonal Geometry Analysis
    # ========================================================================

    def phase_angle_distribution(
        self,
        states: List[ComplexGeometricState],
        bins: int = 36
    ) -> Dict:
        """
        Analyze pairwise phase angle distribution.

        If pentagonal structure exists, should see peaks at 72, 144, 216, 288 deg.

        Args:
            states: List of complex states to analyze
            bins: Number of histogram bins

        Returns:
            Dict with histogram and analysis
        """
        from itertools import combinations

        phase_angles = []

        for s1, s2 in combinations(states, 2):
            phase = np.degrees(s1.E_phase(s2))
            # Wrap to [0, 360)
            phase = phase % 360
            phase_angles.append(phase)

        phase_angles = np.array(phase_angles)
        hist, edges = np.histogram(phase_angles, bins=bins, range=(0, 360))

        # Find peaks
        peak_indices = np.argsort(hist)[-5:]  # Top 5 bins
        peak_centers = [(edges[i] + edges[i+1])/2 for i in peak_indices]

        # Check for pentagonal pattern (72 deg spacing)
        pentagonal_bins = [72, 144, 216, 288, 360]  # 0=360
        pentagonal_matches = 0
        for center in peak_centers:
            for pent in pentagonal_bins:
                if abs(center - pent) < (360 / bins):
                    pentagonal_matches += 1
                    break

        return {
            'histogram': hist.tolist(),
            'bin_edges': edges.tolist(),
            'peak_centers': peak_centers,
            'n_pairs': len(phase_angles),
            'mean_phase_deg': float(np.mean(phase_angles)),
            'std_phase_deg': float(np.std(phase_angles)),
            'pentagonal_matches': pentagonal_matches,
            'pentagonal_score': pentagonal_matches / 5,  # 0 to 1
        }

    def geodesic_angle_distribution(
        self,
        states: List[ComplexGeometricState],
        bins: int = 18
    ) -> Dict:
        """
        Analyze pairwise geodesic angles (magnitudes only).

        This is comparable to real-space angular analysis.

        Args:
            states: List of complex states
            bins: Number of histogram bins

        Returns:
            Dict with histogram and pentagonal analysis
        """
        from itertools import combinations

        angles_deg = []

        for s1, s2 in combinations(states, 2):
            # Geodesic distance = arccos(|<psi|phi>|)
            dist = s1.distance_geodesic(s2)
            angles_deg.append(np.degrees(dist))

        angles_deg = np.array(angles_deg)
        hist, edges = np.histogram(angles_deg, bins=bins, range=(0, 180))

        # Find peak
        peak_bin = np.argmax(hist)
        peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2

        return {
            'histogram': hist.tolist(),
            'bin_edges': edges.tolist(),
            'n_pairs': len(angles_deg),
            'mean_deg': float(np.mean(angles_deg)),
            'std_deg': float(np.std(angles_deg)),
            'peak_center_deg': float(peak_center),
            'deviation_from_pentagonal': float(abs(peak_center - PENTAGONAL_ANGLE_DEG)),
            'near_pentagonal': float(abs(peak_center - PENTAGONAL_ANGLE_DEG)) < 10,
        }

    # ========================================================================
    # Negation Detection
    # ========================================================================

    def detect_negation(
        self,
        state1: ComplexGeometricState,
        state2: ComplexGeometricState,
        phase_threshold_deg: float = 135.0
    ) -> Dict:
        """
        Detect if two concepts are negations/opposites.

        Hypothesis: Opposites should have ~180 deg phase shift.

        Args:
            state1, state2: States to compare
            phase_threshold_deg: Phase above this indicates negation

        Returns:
            Dict with negation analysis
        """
        phase_deg = np.degrees(state1.E_phase(state2))
        phase_abs = abs(phase_deg)

        # Check for near-180 phase (opposites)
        is_negation = phase_abs > phase_threshold_deg

        # Also check if magnitude is still high (same "topic" but opposite)
        magnitude = state1.E_magnitude(state2)
        is_topical_negation = is_negation and magnitude > 0.3

        return {
            'phase_deg': float(phase_deg),
            'phase_abs_deg': float(phase_abs),
            'magnitude': float(magnitude),
            'is_negation': is_negation,
            'is_topical_negation': is_topical_negation,
            'threshold_deg': phase_threshold_deg,
        }

    # ========================================================================
    # Navigation (Compass Mode)
    # ========================================================================

    def get_direction(
        self,
        current: ComplexGeometricState,
        candidates: List[ComplexGeometricState],
        weights: Optional[np.ndarray] = None
    ) -> Tuple[int, Dict]:
        """
        Compass mode: Choose best direction from candidates.

        Scoring: score = E_magnitude(current, candidate) * coherence(candidate) * weight

        IMPORTANT NOTE (2026-01-21):
        For sign-to-phase complexification, phase_coherence is near-constant
        (~0.01-0.11) for typical random/embedding vectors because roughly half
        the components are positive and half negative. This means coherence
        does NOT differentiate candidates, and the scoring reduces to:

            score ~= E_magnitude * constant

        Since E_magnitude tracks cosine similarity very closely (difference < 0.01),
        navigation with sign-to-phase is effectively equivalent to cosine-based
        navigation. The phase structure adds no additional signal.

        For Hilbert transform, coherence has more variation but the E_magnitude
        still dominates the scoring in practice.

        Args:
            current: Current state
            candidates: Possible next states
            weights: Optional weights for candidates

        Returns:
            (best_index, analysis_dict)
        """
        if not candidates:
            raise ValueError("No candidates provided")

        scores = []
        for i, candidate in enumerate(candidates):
            # Magnitude (similarity)
            mag = current.E_magnitude(candidate)

            # Phase coherence of candidate
            coherence = candidate.phase_coherence

            # Combined score
            weight = weights[i] if weights is not None else 1.0
            score = mag * coherence * weight

            scores.append({
                'index': i,
                'magnitude': float(mag),
                'coherence': float(coherence),
                'weight': float(weight),
                'score': float(score),
            })

        # Best = highest score
        scores.sort(key=lambda x: x['score'], reverse=True)
        best_idx = scores[0]['index']

        return best_idx, {
            'best_index': best_idx,
            'best_score': scores[0]['score'],
            'all_scores': scores,
        }

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict:
        """Return usage statistics."""
        return dict(self.stats)


# ============================================================================
# Utility Functions
# ============================================================================

def compare_methods_on_vector(
    real_vector: np.ndarray
) -> Dict[str, ComplexGeometricState]:
    """
    Compare all complexification methods on same vector.

    Returns dict of method -> ComplexGeometricState.
    """
    compass = ComplexCompass()
    results = {}

    for method in ComplexificationMethod:
        try:
            state = compass.complexify(real_vector, method=method)
            results[method.value] = state
        except Exception as e:
            results[method.value] = {'error': str(e)}

    return results


def analyze_phase_structure(
    real_vectors: List[np.ndarray],
    method: ComplexificationMethod = ComplexificationMethod.HILBERT
) -> Dict:
    """
    Comprehensive phase structure analysis.

    Args:
        real_vectors: List of real embeddings
        method: Complexification method

    Returns:
        Dict with full analysis including pentagonal check
    """
    compass = ComplexCompass(method=method)

    # Complexify all vectors
    states = [compass.complexify(v) for v in real_vectors]

    # Run analyses
    phase_dist = compass.phase_angle_distribution(states)
    geodesic_dist = compass.geodesic_angle_distribution(states)

    # Compare first few pairs (for detailed analysis)
    detailed_comparisons = []
    for i in range(min(5, len(states))):
        for j in range(i+1, min(5, len(states))):
            detailed_comparisons.append(
                compass.compare_real_vs_complex(states[i], states[j])
            )

    return {
        'n_vectors': len(real_vectors),
        'method': method.value,
        'phase_distribution': phase_dist,
        'geodesic_distribution': geodesic_dist,
        'detailed_comparisons': detailed_comparisons,
        'mean_Df': float(np.mean([s.Df for s in states])),
        'mean_coherence': float(np.mean([s.phase_coherence for s in states])),
    }


# ============================================================================
# Example / Demo
# ============================================================================

def demo_complex_compass():
    """Demonstrate ComplexCompass capabilities."""
    print("=" * 70)
    print("COMPLEX COMPASS DEMO")
    print("=" * 70)
    print()

    compass = ComplexCompass(method=ComplexificationMethod.HILBERT)

    # Test words
    words = ["good", "bad", "hot", "cold", "king", "queen"]

    print("Complexifying embeddings...")
    states = {word: compass.initialize(word) for word in words}

    print(f"\nStates created:")
    for word, state in states.items():
        print(f"  {word}: Df={state.Df:.2f}, coherence={state.phase_coherence:.3f}")

    print("\n" + "-" * 70)
    print("REAL vs COMPLEX SIMILARITY")
    print("-" * 70)

    pairs = [("good", "bad"), ("hot", "cold"), ("king", "queen")]
    for w1, w2 in pairs:
        analysis = compass.compare_real_vs_complex(states[w1], states[w2])
        print(f"\n{w1} <-> {w2}:")
        print(f"  Cosine (real):     {analysis['cosine_similarity']:.4f}")
        print(f"  Hermitian mag:     {analysis['hermitian_magnitude']:.4f}")
        print(f"  Hermitian phase:   {analysis['hermitian_phase_deg']:.1f} deg")

        negation = compass.detect_negation(states[w1], states[w2])
        if negation['is_negation']:
            print(f"  NEGATION DETECTED (phase > {negation['threshold_deg']} deg)")

    print("\n" + "-" * 70)
    print("PENTAGONAL STRUCTURE CHECK")
    print("-" * 70)

    all_states = list(states.values())
    phase_dist = compass.phase_angle_distribution(all_states)

    print(f"\nPhase angle distribution:")
    print(f"  Mean: {phase_dist['mean_phase_deg']:.1f} deg")
    print(f"  Pentagonal score: {phase_dist['pentagonal_score']:.2f} (1.0 = perfect)")
    print(f"  Peak centers: {[f'{p:.1f}' for p in phase_dist['peak_centers']]}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_complex_compass()

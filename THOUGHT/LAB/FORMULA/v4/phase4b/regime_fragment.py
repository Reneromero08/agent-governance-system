"""Formula-backed Regime Fragment — replaces self-consistency.

Uses the truth attractor formula to classify generation into three regimes:
    CONVERGENT: High consensus + low entropy → pass
    DIVERGENT:  Low consensus → fail (already caught by other fragments)
    CRITICAL:   High consensus + HIGH entropy → overconfident hallucination → fail

Key insight: when both Factual and CORTEX-COMMONSENSE pass but the model
is WRONG, the consensus looks perfect (R = inf). The only way to detect
this is token-level logit entropy — if the model assigns uniform-ish
probability across many tokens, it was guessing despite the confident output.

R = (E/grad_S) * sigma^Df from TRUTH_ATTRACTOR/FORMULA.md
"""
import sys, math, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase4b_fragments import FragmentResult
from phase4b_lattice import ConsensusResult


class RegimeFragment:
    """Third verification fragment using the truth attractor formula.

    Detects the CRITICAL regime: when consensus is high but the model's
    own token probabilities suggest guessing/uncertainty.
    """

    def __init__(self, entropy_threshold: float = 0.6):
        self.entropy_threshold = entropy_threshold

    def verify(self, consensus: ConsensusResult, logits=None,
               generated_text: str = "") -> FragmentResult:
        """Classify regime from consensus + logit entropy.

        Args:
            consensus: ConsensusResult from other fragments
            logits: Optional token logits array (n_tokens, n_vocab)
            generated_text: The generated text for evidence

        Returns:
            FragmentResult with regime classification
        """
        # Compute R = 1/grad_S (already in consensus)
        R = consensus.resonance if consensus.resonance != float('inf') else 1000.0
        grad_S = consensus.grad_S
        E = consensus.consensus_ratio

        # Compute logit entropy if logits available
        entropy = 0.0
        entropy_signal = "no_logits"
        if logits is not None and hasattr(logits, 'shape') and logits.size > 0:
            try:
                # Normalize to probabilities
                logits_flat = logits.reshape(-1, logits.shape[-1])
                logits_centered = logits_flat - logits_flat.max(axis=-1, keepdims=True)
                probs = np.exp(logits_centered) / np.exp(logits_centered).sum(axis=-1, keepdims=True)
                probs = np.clip(probs, 1e-12, 1.0)
                # Entropy normalized to [0, 1] by dividing by log(vocab_size)
                raw_entropy = -np.sum(probs * np.log(probs), axis=-1)
                max_entropy = math.log(logits.shape[-1])
                if max_entropy > 0:
                    entropy = float(np.mean(raw_entropy) / max_entropy)
                entropy_signal = "entropy={:.3f}".format(entropy)
            except Exception:
                pass

        # Regime classification
        if not consensus.consensus_holds:
            # DIVERGENT: consensus broken, hard gate already triggered
            return FragmentResult(
                fragment_id=3, fragment_name="Regime",
                score=0.0, confidence=0.9, verdict="hard_fail",
                evidence="DIVERGENT: R={:.1f} grad_S={:.3f} {}".format(R, grad_S, entropy_signal),
                details={"R": round(R, 1), "grad_S": round(grad_S, 3), "E": round(E, 2), "entropy": round(entropy, 3)},
            )

        # CONVERGENT or CRITICAL
        if entropy > self.entropy_threshold:
            # CRITICAL: high consensus but high entropy = model was guessing
            return FragmentResult(
                fragment_id=3, fragment_name="Regime",
                score=0.0, confidence=0.8, verdict="hard_fail",
                evidence="CRITICAL: consensus passed but entropy={:.3f} > {:.2f} (model guessing)".format(
                    entropy, self.entropy_threshold),
                details={"R": round(R, 1), "grad_S": round(grad_S, 3), "E": round(E, 2), "entropy": round(entropy, 3)},
            )
        else:
            # CONVERGENT: consensus holds, entropy low
            return FragmentResult(
                fragment_id=3, fragment_name="Regime",
                score=1.0, confidence=0.8, verdict="pass",
                evidence="CONVERGENT: R={:.1f} grad_S={:.3f} {}".format(R, grad_S, entropy_signal),
                details={"R": round(R, 1), "grad_S": round(grad_S, 3), "E": round(E, 2), "entropy": round(entropy, 3)},
            )

    @staticmethod
    def compute_entropy(logits: np.ndarray) -> float:
        """Compute normalized entropy from logits."""
        logits_centered = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits_centered) / np.exp(logits_centered).sum(axis=-1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        raw = -np.sum(probs * np.log(probs), axis=-1)
        max_h = math.log(logits.shape[-1])
        return float(np.mean(raw) / max_h) if max_h > 0 else 0.0


if __name__ == "__main__":
    from phase4b_lattice import compute_consensus, NodeResult, Verdict

    # Test: high consensus, low entropy -> CONVERGENT
    n1 = NodeResult(1, "Factual", Verdict.PASS, 1.0, "ok", "text")
    n2 = NodeResult(2, "CORTEX", Verdict.PASS, 1.0, "ok", "text")
    consensus = compute_consensus([n1, n2])
    rf = RegimeFragment()
    dummy_logits = np.array([[0.0] * 100 + [10.0]])  # low entropy
    r = rf.verify(consensus, dummy_logits, "test")
    print("Convergent: {} ({})".format(r.verdict, r.evidence))

    # Test: high consensus, high entropy -> CRITICAL
    high_ent = np.random.randn(1, 65536).astype(np.float32)  # uniform-ish
    r2 = rf.verify(consensus, high_ent, "test")
    print("Critical:   {} ({})".format(r2.verdict, r2.evidence))

    # Test: no logits -> passes (no signal)
    r3 = rf.verify(consensus, None, "test")
    print("No logits:  {} ({})".format(r3.verdict, r3.evidence))

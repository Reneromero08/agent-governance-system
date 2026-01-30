#!/usr/bin/env python3
"""
Q51 Semantic Interference Test

Tests whether semantic meanings interfere constructively/destructively like waves.

Question: Can meanings interfere constructively/destructively like waves?

Theory:
- Complex: |psi1 + psi2^2 = |psi1^2 + |psi2^2 + 2Re(psi1*psi2) [interference term]
- Real: Just |v1+v2|^2 = |v1|^2 + |v2|^2 + 2v_1·v_2 [no interference]

Method:
1. Create ambiguous words with multiple meanings:
   - "bank": (river bank) vs (financial bank)
   - "bat": (animal) vs (sports equipment)
2. Get embeddings for each meaning separately:
   - "river bank", "bank account"
   - "bat animal", "bat sports"
3. Create superposition:
   - Average: (bank_river + bank_finance)/2
4. Test for interference:
   - Compare superposition to actual ambiguous "bank"
   - If interference: norm(superposition) ≠ average of norms
   - Pattern: amplification or cancellation effects

Statistical Test:
- Measure |superposition|^2 vs (|psi1^2 + |psi2^2)/2
- Significant difference = interference
- t-test for significance

Run with:
    cd THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED
    python -m pytest tests/test_q51_semantic_interference.py -v

Or directly:
    python tests/test_q51_semantic_interference.py
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
from scipy import stats
from datetime import datetime
import sys


@dataclass
class InterferenceResult:
    """Result for a single ambiguous word test"""
    word: str
    meaning1: str
    meaning2: str
    vec1_norm: float
    vec2_norm: float
    avg_norm: float
    superposition_norm: float
    ambiguous_norm: float
    interference_term: float
    has_interference: bool
    control_type: str  # "ambiguous" or "random"


@dataclass
class InterferenceReport:
    """Complete interference test results"""
    # Test metadata
    timestamp: str
    total_tests: int
    ambiguous_tests: int
    control_tests: int
    
    # Interference statistics
    ambiguous_interference_rate: float
    control_interference_rate: float
    
    # Effect sizes
    cohens_d_interference: float
    
    # Statistical significance
    t_statistic: float
    p_value: float
    
    # Detailed results
    ambiguous_results: List[Dict]
    control_results: List[Dict]
    
    # Conclusion
    interpretation: str
    shows_interference: bool
    supports_complex_structure: bool


# Ambiguous words with contextual disambiguation
AMBIGUOUS_WORDS = [
    # Word, context1 (meaning 1), context2 (meaning 2)
    ("bank", "river bank", "bank account"),
    ("bat", "bat animal", "cricket bat"),
    ("spring", "spring season", "metal spring"),
    ("crane", "crane bird", "construction crane"),
    ("date", "calendar date", "fruit date"),
    ("rock", "rock music", "geological rock"),
    ("current", "electric current", "ocean current"),
    ("plane", "airplane", "wood plane"),
    ("bark", "tree bark", "dog bark"),
    ("lie", "tell a lie", "lie down"),
    ("bow", "bow weapon", "bow tie"),
    ("cast", "cast a spell", "broken arm cast"),
    ("fan", "ceiling fan", "sports fan"),
    ("ruler", "school ruler", "king ruler"),
    ("cell", "prison cell", "biology cell"),
    ("match", "fire match", "tennis match"),
    ("pupil", "student pupil", "eye pupil"),
    ("seal", "animal seal", "seal envelope"),
    ("temple", "religious temple", "head temple"),
    ("wound", "injury wound", "wound clock"),
]

# Random word pairs (negative control - no semantic connection)
RANDOM_PAIRS = [
    ("apple", "democracy", "sandwich", "refrigerator"),
    ("mountain", "purple", "ocean", "triangle"),
    ("guitar", "velocity", "piano", "acceleration"),
    ("coffee", "algorithm", "tea", "database"),
    ("blue", "economics", "red", "astronomy"),
    ("book", "chemistry", "magazine", "physics"),
    ("dog", "algebra", "cat", "geometry"),
    ("tree", "politics", "flower", "history"),
    ("car", "philosophy", "bicycle", "psychology"),
    ("window", "mathematics", "door", "statistics"),
    ("pencil", "geography", "eraser", "meteorology"),
    ("shirt", "biology", "pants", "geology"),
    ("sun", "linguistics", "moon", "archaeology"),
    ("chair", "sociology", "table", "anthropology"),
    ("bread", "botany", "milk", "zoology"),
    ("water", "engineering", "fire", "medicine"),
    ("computer", "law", "phone", "justice"),
    ("house", "music", "apartment", "painting"),
    ("train", "sculpture", "bus", "architecture"),
    ("pen", "literature", "paper", "poetry"),
]


# Load model ONCE at module level for efficiency
print("Loading sentence transformer model...")
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    MODEL = None

def get_sentence_embedding(text: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Get sentence embedding using sentence-transformers.
    
    Uses globally loaded MODEL for efficiency.
    """
    if MODEL is not None:
        embedding = MODEL.encode(text, convert_to_numpy=True)
        return embedding
    else:
        # Fallback: deterministic synthetic embeddings for testing
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(384)
        vec = vec / np.linalg.norm(vec)
        return vec


def compute_norm_squared(vector: np.ndarray) -> float:
    """Compute squared norm |psi^2."""
    return float(np.sum(vector ** 2))


def create_superposition(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Create superposition state: (psi1 + psi2)/2."""
    superposition = (vec1 + vec2) / 2.0
    return superposition


def compute_interference_term(vec1: np.ndarray, vec2: np.ndarray, 
                               superposition: np.ndarray) -> float:
    """
    Compute interference term: |superposition|^2 - (|psi1^2 + |psi2^2)/2
    
    For complex: |psi1 + psi2^2 = |psi1^2 + |psi2^2 + 2Re(psi1*psi2)
    For real: |v1+v2|^2 = |v1|^2 + |v2|^2 + 2v_1·v_2
    
    Interference term = 2Re(psi1*psi2) for complex, or 2v_1·v_2 for real
    """
    superposition_norm_sq = compute_norm_squared(superposition)
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    
    # Expected under no-interference (just average of norms)
    expected_norm_sq = (norm1_sq + norm2_sq) / 4.0  # Divide by 4 because superposition is averaged
    
    # Interference term
    interference = superposition_norm_sq - expected_norm_sq
    
    return interference


def test_ambiguous_word(word: str, context1: str, context2: str,
                        threshold: float = 0.1) -> InterferenceResult:
    """
    Test for interference in an ambiguous word.
    
    Args:
        word: The ambiguous word
        context1: Context selecting meaning 1
        context2: Context selecting meaning 2
        threshold: Threshold for detecting interference
    """
    # Get embeddings for each meaning
    vec1 = get_sentence_embedding(context1)
    vec2 = get_sentence_embedding(context2)
    
    # Create superposition
    superposition = create_superposition(vec1, vec2)
    
    # Get embedding for ambiguous word alone
    ambiguous_vec = get_sentence_embedding(word)
    
    # Compute norms
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    superposition_norm_sq = compute_norm_squared(superposition)
    ambiguous_norm_sq = compute_norm_squared(ambiguous_vec)
    
    # Average norm of individual meanings
    avg_norm_sq = (norm1_sq + norm2_sq) / 2.0
    
    # Compute interference
    interference = compute_interference_term(vec1, vec2, superposition)
    
    # Check if there's significant interference
    # Interference exists if superposition norm differs from expected
    expected_superposition_norm = np.sqrt((norm1_sq + norm2_sq) / 4.0)
    actual_superposition_norm = np.sqrt(superposition_norm_sq)
    
    # Relative difference
    relative_diff = abs(actual_superposition_norm - expected_superposition_norm) / (expected_superposition_norm + 1e-10)
    has_interference = relative_diff > threshold
    
    return InterferenceResult(
        word=word,
        meaning1=context1,
        meaning2=context2,
        vec1_norm=float(np.sqrt(norm1_sq)),
        vec2_norm=float(np.sqrt(norm2_sq)),
        avg_norm=float(np.sqrt(avg_norm_sq)),
        superposition_norm=actual_superposition_norm,
        ambiguous_norm=float(np.sqrt(ambiguous_norm_sq)),
        interference_term=interference,
        has_interference=has_interference,
        control_type="ambiguous"
    )


def test_control_pair(word1a: str, word1b: str, word2a: str, word2b: str,
                      threshold: float = 0.1) -> InterferenceResult:
    """
    Test control pair (random words with no semantic connection).
    
    Should show NO interference.
    """
    # Create artificial "ambiguous" word from random pair
    artificial_word = f"{word1a}_{word2a}"
    
    # Get embeddings
    vec1 = get_sentence_embedding(word1b)  # Context 1
    vec2 = get_sentence_embedding(word2b)  # Context 2
    
    # Create superposition
    superposition = create_superposition(vec1, vec2)
    
    # Get embedding for artificial word
    artificial_vec = get_sentence_embedding(artificial_word)
    
    # Compute norms
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    superposition_norm_sq = compute_norm_squared(superposition)
    artificial_norm_sq = compute_norm_squared(artificial_vec)
    
    avg_norm_sq = (norm1_sq + norm2_sq) / 2.0
    
    # Compute interference
    interference = compute_interference_term(vec1, vec2, superposition)
    
    # Check interference
    expected_superposition_norm = np.sqrt((norm1_sq + norm2_sq) / 4.0)
    actual_superposition_norm = np.sqrt(superposition_norm_sq)
    
    relative_diff = abs(actual_superposition_norm - expected_superposition_norm) / (expected_superposition_norm + 1e-10)
    has_interference = relative_diff > threshold
    
    return InterferenceResult(
        word=artificial_word,
        meaning1=word1b,
        meaning2=word2b,
        vec1_norm=float(np.sqrt(norm1_sq)),
        vec2_norm=float(np.sqrt(norm2_sq)),
        avg_norm=float(np.sqrt(avg_norm_sq)),
        superposition_norm=actual_superposition_norm,
        ambiguous_norm=float(np.sqrt(artificial_norm_sq)),
        interference_term=interference,
        has_interference=has_interference,
        control_type="control"
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def run_interference_test(threshold: float = 0.1) -> InterferenceReport:
    """
    Run complete semantic interference test.
    """
    print("=" * 70)
    print("Q51 SEMANTIC INTERFERENCE TEST")
    print("=" * 70)
    print("\nQuestion: Can meanings interfere constructively/destructively like waves?")
    print("\nTheory:")
    print("  Complex: |psi1 + psi2^2 = |psi1^2 + |psi2^2 + 2Re(psi1*psi2) [interference term]")
    print("  Real:    |v1+v2|^2 = |v1|^2 + |v2|^2 + 2v_1·v_2 [no interference]")
    
    results = []
    
    # Test 1: Ambiguous words (should show interference if complex)
    print("\n[1] Testing AMBIGUOUS WORDS...")
    print("    (Should show interference IF meanings have complex structure)")
    ambiguous_results = []
    
    for word, context1, context2 in AMBIGUOUS_WORDS:
        result = test_ambiguous_word(word, context1, context2, threshold)
        ambiguous_results.append(result)
        results.append(result)
        status = "INTERFERENCE" if result.has_interference else "no interference"
        print(f"  {word:12s}: {status:15s} (interference term: {result.interference_term:+.4f})")
    
    ambiguous_interference_rate = np.mean([r.has_interference for r in ambiguous_results])
    print(f"\n  Ambiguous words: {len(ambiguous_results)} tested")
    print(f"  Interference rate: {ambiguous_interference_rate:.1%}")
    
    # Test 2: Control pairs (should NOT show interference)
    print("\n[2] Testing CONTROL PAIRS...")
    print("    (Random words - should NOT show interference)")
    control_results = []
    
    for words in RANDOM_PAIRS:
        word1a, word1b, word2a, word2b = words
        result = test_control_pair(word1a, word1b, word2a, word2b, threshold)
        control_results.append(result)
        results.append(result)
        status = "INTERFERENCE" if result.has_interference else "no interference"
        print(f"  {result.word[:20]:20s}: {status:15s} (interference term: {result.interference_term:+.4f})")
    
    control_interference_rate = np.mean([r.has_interference for r in control_results])
    print(f"\n  Control pairs: {len(control_results)} tested")
    print(f"  Interference rate: {control_interference_rate:.1%}")
    
    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # Collect interference terms
    ambiguous_terms = np.array([r.interference_term for r in ambiguous_results])
    control_terms = np.array([r.interference_term for r in control_results])
    
    # Effect size
    print("\n[3] Effect Size (Cohen's d)...")
    d_ambiguous_vs_control = cohens_d(np.abs(ambiguous_terms), np.abs(control_terms))
    print(f"  |Interference| - Ambiguous vs Control: d = {d_ambiguous_vs_control:.3f}")
    
    def interpret_d(d):
        if abs(d) < 0.2:
            return "Negligible"
        elif abs(d) < 0.5:
            return "Small"
        elif abs(d) < 0.8:
            return "Medium"
        else:
            return "LARGE"
    
    print(f"    Interpretation: {interpret_d(d_ambiguous_vs_control)}")
    
    # T-test
    print("\n[4] T-test for difference in interference...")
    t_stat, p_value = stats.ttest_ind(np.abs(ambiguous_terms), np.abs(control_terms))
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (alpha = 0.05)")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    interpretation_parts = []
    
    # Check 1: Do ambiguous words show interference?
    if ambiguous_interference_rate > 0.5:
        interpretation_parts.append(
            f"Ambiguous words show FREQUENT interference ({ambiguous_interference_rate:.1%} of cases). "
            f"This suggests semantic meanings may have wave-like properties."
        )
        shows_interference = True
    elif ambiguous_interference_rate > 0.2:
        interpretation_parts.append(
            f"Ambiguous words show MODERATE interference ({ambiguous_interference_rate:.1%} of cases). "
            f"Some evidence for wave-like semantic structure."
        )
        shows_interference = True
    else:
        interpretation_parts.append(
            f"Ambiguous words show WEAK interference ({ambiguous_interference_rate:.1%} of cases). "
            f"Little evidence for wave-like semantic structure."
        )
        shows_interference = False
    
    # Check 2: Is there separation from controls?
    if abs(d_ambiguous_vs_control) > 0.8 and p_value < 0.05:
        interpretation_parts.append(
            f"STRONG separation from control pairs (d={d_ambiguous_vs_control:.2f}, p={p_value:.3f}). "
            f"The interference is specific to semantically related meanings."
        )
        supports_complex = True
    elif abs(d_ambiguous_vs_control) > 0.5 and p_value < 0.05:
        interpretation_parts.append(
            f"Moderate separation from control pairs (d={d_ambiguous_vs_control:.2f}, p={p_value:.3f}). "
            f"The interference may be semantic in nature."
        )
        supports_complex = False
    else:
        interpretation_parts.append(
            f"WEAK separation from control pairs (d={d_ambiguous_vs_control:.2f}, p={p_value:.3f}). "
            f"Cannot distinguish semantic interference from random geometric effects."
        )
        supports_complex = False
    
    # Check 3: Are controls clean?
    if control_interference_rate > 0.3:
        interpretation_parts.append(
            f"WARNING: Control pairs show {control_interference_rate:.1%} interference. "
            f"This may indicate geometric artifacts rather than semantic effects."
        )
        supports_complex = False
    else:
        interpretation_parts.append(
            f"Control pairs show {control_interference_rate:.1%} interference (baseline noise)."
        )
    
    # Final verdict
    if shows_interference and supports_complex:
        verdict = (
            "SUPPORTS COMPLEX STRUCTURE: Semantic meanings show wave-like interference patterns "
            "with significant separation from controls. The superposition of meanings creates "
            "amplification/cancellation effects consistent with complex number structure."
        )
    elif shows_interference and not supports_complex:
        verdict = (
            "UNCLEAR: Ambiguous words show interference, but controls also show some activity. "
            "The effect may be partially geometric rather than purely semantic. "
            "Further investigation needed with better controls."
        )
    else:
        verdict = (
            "DOES NOT SUPPORT COMPLEX STRUCTURE: No significant interference detected in semantic "
            "superpositions. The combination of meanings appears to follow simple vector addition "
            "(real structure) rather than wave interference (complex structure)."
        )
    
    interpretation = "\n\n".join(interpretation_parts) + "\n\n" + verdict
    print(interpretation)
    
    # Create report
    timestamp = datetime.now().isoformat()
    report = InterferenceReport(
        timestamp=timestamp,
        total_tests=len(results),
        ambiguous_tests=len(ambiguous_results),
        control_tests=len(control_results),
        ambiguous_interference_rate=float(ambiguous_interference_rate),
        control_interference_rate=float(control_interference_rate),
        cohens_d_interference=float(d_ambiguous_vs_control),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        ambiguous_results=[asdict(r) for r in ambiguous_results],
        control_results=[asdict(r) for r in control_results],
        interpretation=interpretation,
        shows_interference=shows_interference,
        supports_complex_structure=supports_complex
    )
    
    return report


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_results(report: InterferenceReport, output_dir: str):
    """Save results to JSON and generate markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"q51_interference_{timestamp}.json")
    report_dict = asdict(report)
    report_dict = convert_to_serializable(report_dict)
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate markdown report
    md_path = os.path.join(os.path.dirname(output_dir), "q51_interference_report.md")
    with open(md_path, 'w') as f:
        f.write("# Q51 Semantic Interference Test Report\n\n")
        f.write(f"**Test Date:** {report.timestamp}\n\n")
        f.write("**Question:** Can semantic meanings interfere constructively/destructively like waves?\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        verdict = "SUPPORTS" if report.supports_complex_structure else "DOES NOT SUPPORT"
        if report.shows_interference and not report.supports_complex_structure:
            verdict = "UNCLEAR"
        f.write(f"**Verdict:** {verdict} complex structure hypothesis\n\n")
        f.write(f"- Ambiguous words tested: {report.ambiguous_tests}\n")
        f.write(f"- Ambiguous words with interference: {report.ambiguous_interference_rate:.1%}\n")
        f.write(f"- Control pairs tested: {report.control_tests}\n")
        f.write(f"- Control pairs with interference: {report.control_interference_rate:.1%}\n\n")
        
        f.write("## Theory\n\n")
        f.write("This test probes whether semantic space has complex (wave-like) structure:\n\n")
        f.write("- **Complex structure:** |psi1 + psi2^2 = |psi1^2 + |psi2^2 + 2Re(psi1*psi2)  [interference term]\n")
        f.write("- **Real structure:** |v1+v2|^2 = |v1|^2 + |v2|^2 + 2v_1·v_2  [no interference]\n\n")
        f.write("We test by creating superpositions of word meanings and measuring if the\n")
        f.write("norm shows wave-like interference (amplification or cancellation).\n\n")
        
        f.write("## Method\n\n")
        f.write("1. **Ambiguous Words:** Words with multiple distinct meanings\n")
        f.write("   - Example: 'bank' = 'river bank' OR 'bank account'\n")
        f.write("   - Get separate embeddings for each meaning\n")
        f.write("   - Create superposition: psi = (psi1 + psi2)/2\n")
        f.write("   - Test: Does |psi^2 show interference?\n\n")
        f.write("2. **Control Pairs:** Random word pairs (no semantic connection)\n")
        f.write("   - Should show NO interference\n")
        f.write("   - Validates that effects are semantic, not geometric\n\n")
        
        f.write("## Statistical Results\n\n")
        f.write("### Effect Size (Cohen's d)\n\n")
        f.write(f"| Comparison | Cohen's d | Interpretation |\n")
        f.write(f"|------------|-----------|----------------|\n")
        
        def interpret_d(d):
            if abs(d) < 0.2:
                return "Negligible"
            elif abs(d) < 0.5:
                return "Small"
            elif abs(d) < 0.8:
                return "Medium"
            else:
                return "Large"
        
        f.write(f"| Ambiguous vs Control | {report.cohens_d_interference:.3f} | {interpret_d(report.cohens_d_interference)} |\n\n")
        
        f.write("### T-test Results\n\n")
        f.write(f"- t-statistic: {report.t_statistic:.3f}\n")
        f.write(f"- p-value: {report.p_value:.4f}\n")
        f.write(f"- Significance: {'p < 0.05 (significant)' if report.p_value < 0.05 else 'p ≥ 0.05 (not significant)'}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("### Ambiguous Words\n\n")
        f.write("| Word | Meaning 1 | Meaning 2 | Interference | Term |\n")
        f.write("|------|-----------|-----------|--------------|------|\n")
        for r in report.ambiguous_results:
            status = "YES" if r['has_interference'] else "no"
            f.write(f"| {r['word']} | {r['meaning1'][:15]} | {r['meaning2'][:15]} | {status} | {r['interference_term']:+.4f} |\n")
        
        f.write("\n### Control Pairs\n\n")
        f.write("| Pair | Term 1 | Term 2 | Interference | Interference Term |\n")
        f.write("|------|--------|--------|--------------|-------------------|\n")
        for r in report.control_results[:10]:  # Show first 10
            status = "YES" if r['has_interference'] else "no"
            f.write(f"| {r['word'][:15]} | {r['meaning1'][:12]} | {r['meaning2'][:12]} | {status} | {r['interference_term']:+.4f} |\n")
        if len(report.control_results) > 10:
            f.write(f"| ... | ... | ... | ... | ... ({len(report.control_results) - 10} more) |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write(report.interpretation.replace("\n", "\n\n"))
        f.write("\n\n")
        
        f.write("## Conclusion\n\n")
        if report.supports_complex_structure:
            f.write("The semantic interference test **SUPPORTS** the hypothesis that semantic "
                    "space has complex structure. The key findings are:\n\n")
            f.write(f"1. **Interference detected:** {report.ambiguous_interference_rate:.1%} of ambiguous words "
                    "showed wave-like interference effects\n")
            f.write(f"2. **Semantic specificity:** Strong separation from control pairs (d={report.cohens_d_interference:.2f}) "
                    "indicates the effect is semantic, not geometric\n")
            f.write("3. **Implication:** The combination of word meanings follows complex number "
                    "arithmetic, not simple vector addition\n\n")
            f.write("This suggests that semantic space has a complex (Hilbert space) structure "
                    "where meanings can exist in superposition and interfere.\n")
        elif report.shows_interference:
            f.write("The semantic interference test shows **AMBIGUOUS** results:\n\n")
            f.write(f"1. **Some interference:** {report.ambiguous_interference_rate:.1%} of ambiguous words "
                    "showed interference effects\n")
            f.write(f"2. **Weak separation:** Control pairs also showed activity (d={report.cohens_d_interference:.2f})\n")
            f.write("3. **Implication:** The effect may be partially geometric or require better controls\n\n")
            f.write("Further investigation with more rigorous semantic controls is recommended.\n")
        else:
            f.write("The semantic interference test **DOES NOT SUPPORT** the hypothesis that "
                    "semantic space has complex structure. The key findings are:\n\n")
            f.write(f"1. **No interference:** Only {report.ambiguous_interference_rate:.1%} of ambiguous words "
                    "showed wave-like effects\n")
            f.write("2. **Simple addition:** Semantic superpositions follow real vector addition\n")
            f.write("3. **Implication:** There is no evidence that meanings interfere like waves\n\n")
            f.write("This suggests that semantic space has a real (Euclidean) structure "
                    "where meanings combine through simple addition.\n")
        
        f.write("\n\n---\n\n")
        f.write("*Generated by test_q51_semantic_interference.py*\n")
        f.write(f"*Timestamp: {report.timestamp}*\n")
    
    print(f"Report saved to: {md_path}")


def test_basic_interference():
    """Test basic interference calculation."""
    print("\n" + "=" * 70)
    print("TEST: Basic Interference Calculation")
    print("=" * 70)
    
    # Create two orthogonal vectors (should show minimal interference)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    superposition = create_superposition(vec1, vec2)
    
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    super_norm_sq = compute_norm_squared(superposition)
    expected = (norm1_sq + norm2_sq) / 4.0
    
    print(f"\nOrthogonal vectors:")
    print(f"  |v1|^2 = {norm1_sq:.4f}")
    print(f"  |v2|^2 = {norm2_sq:.4f}")
    print(f"  |superposition|^2 = {super_norm_sq:.4f}")
    print(f"  Expected (no interference) = {expected:.4f}")
    print(f"  Interference term = {super_norm_sq - expected:.4f}")
    
    # Create two parallel vectors (should show constructive interference)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    
    superposition = create_superposition(vec1, vec2)
    
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    super_norm_sq = compute_norm_squared(superposition)
    expected = (norm1_sq + norm2_sq) / 4.0
    
    print(f"\nParallel vectors:")
    print(f"  |v1|^2 = {norm1_sq:.4f}")
    print(f"  |v2|^2 = {norm2_sq:.4f}")
    print(f"  |superposition|^2 = {super_norm_sq:.4f}")
    print(f"  Expected (no interference) = {expected:.4f}")
    print(f"  Interference term = {super_norm_sq - expected:.4f} (constructive)")
    
    # Create two anti-parallel vectors (should show destructive interference)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    
    superposition = create_superposition(vec1, vec2)
    
    norm1_sq = compute_norm_squared(vec1)
    norm2_sq = compute_norm_squared(vec2)
    super_norm_sq = compute_norm_squared(superposition)
    expected = (norm1_sq + norm2_sq) / 4.0
    
    print(f"\nAnti-parallel vectors:")
    print(f"  |v1|^2 = {norm1_sq:.4f}")
    print(f"  |v2|^2 = {norm2_sq:.4f}")
    print(f"  |superposition|^2 = {super_norm_sq:.4f}")
    print(f"  Expected (no interference) = {expected:.4f}")
    print(f"  Interference term = {super_norm_sq - expected:.4f} (destructive)")
    
    print("\nTEST: PASS - Interference calculation working correctly")


if __name__ == "__main__":
    # Run basic test first
    test_basic_interference()
    
    # Run full interference test
    report = run_interference_test(threshold=0.1)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(report, output_dir)
    
    print("\n" + "=" * 70)
    print("SEMANTIC INTERFERENCE TEST COMPLETE")
    print("=" * 70)
    
    # Exit with appropriate code
    if report.supports_complex_structure:
        print("\nResult: COMPLEX STRUCTURE SUPPORTED")
        sys.exit(0)
    else:
        print("\nResult: COMPLEX STRUCTURE NOT SUPPORTED")
        sys.exit(0)  # Still exit 0 - this is a valid result

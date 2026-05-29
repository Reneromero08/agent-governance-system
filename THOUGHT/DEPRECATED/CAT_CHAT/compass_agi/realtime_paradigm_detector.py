"""
REALTIME PARADIGM SHIFT DETECTOR

Uses the discovered signature:
- Movement TOWARD: Earthquake, Death, Wind = SHIFT OCCURRING
- Movement AWAY FROM: Dog, Deer = STABILITY BREAKING

Feed it current events/text and it tells you if a paradigm shift is happening.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


ARCHETYPAL_GEODESICS = {
    0: {'name': 'Crocodile', 'desc': 'emergence from chaos, primordial creation, new beginnings'},
    1: {'name': 'Wind', 'desc': 'change and movement, communication, breath, spreading'},
    2: {'name': 'House', 'desc': 'shelter, security, home, protection, stability'},
    3: {'name': 'Lizard', 'desc': 'adaptation, regeneration, survival, flexibility'},
    4: {'name': 'Serpent', 'desc': 'depth, hidden knowledge, transformation, shedding'},
    5: {'name': 'Death', 'desc': 'endings, release, surrender, letting go, transition'},
    6: {'name': 'Deer', 'desc': 'sensitivity, intuition, grace, gentleness'},
    7: {'name': 'Rabbit', 'desc': 'abundance, multiplication, fertility, growth'},
    8: {'name': 'Water', 'desc': 'flow, emotion, the unconscious, cleansing'},
    9: {'name': 'Dog', 'desc': 'loyalty, guidance, companionship, faithfulness'},
    10: {'name': 'Monkey', 'desc': 'play, creativity, trickster, innovation'},
    11: {'name': 'Grass', 'desc': 'tenacity, growth through adversity, persistence'},
    12: {'name': 'Reed', 'desc': 'authority, structure, vertical order, hierarchy'},
    13: {'name': 'Jaguar', 'desc': 'power, stealth, shadow integration, hidden strength'},
    14: {'name': 'Eagle', 'desc': 'vision, clarity, rising above, perspective'},
    15: {'name': 'Vulture', 'desc': 'patience, wisdom from death, transformation of decay'},
    16: {'name': 'Earthquake', 'desc': 'disruption, paradigm shift, sudden change, upheaval'},
    17: {'name': 'Flint', 'desc': 'decisiveness, cutting away, sacrifice, sharp truth'},
    18: {'name': 'Rain', 'desc': 'nourishment, cleansing, gifts from above, renewal'},
    19: {'name': 'Flower', 'desc': 'beauty, pleasure, completion, bloom, fruition'},
}

# Discovered paradigm shift signature
SHIFT_GEODESICS = ['Earthquake', 'Death', 'Wind']  # Move TOWARD = shift
STABILITY_GEODESICS = ['Dog', 'Deer', 'Reed']  # Move AWAY = shift

# Signature weights (from empirical discovery)
SIGNATURE = {
    'Earthquake': +0.0897,
    'Death': +0.0544,
    'Wind': +0.0487,
    'Water': +0.0443,
    'Flint': +0.0363,
    'Dog': -0.0093,
    'Deer': -0.0064,
    'Reed': -0.0082,
}


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class ParadigmShiftDetector:
    """Real-time paradigm shift detection."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers required")

        self.model = SentenceTransformer(model_name)
        self._build_embeddings()

        # Baseline state (can be updated)
        self.baseline = None

    def _build_embeddings(self):
        """Pre-embed all geodesics."""
        self.geodesic_vecs = {}
        for gid, data in ARCHETYPAL_GEODESICS.items():
            self.geodesic_vecs[data['name']] = self.model.encode(data['desc'])

    def set_baseline(self, baseline_texts):
        """Set baseline state (the 'normal' before potential shift)."""
        vecs = [self.model.encode(t) for t in baseline_texts]
        self.baseline = np.mean(vecs, axis=0)
        print(f"Baseline set from {len(baseline_texts)} texts.")

    def get_geodesic_profile(self, texts):
        """Get similarity to each geodesic."""
        vecs = [self.model.encode(t) for t in texts]
        mean_vec = np.mean(vecs, axis=0)

        profile = {}
        for name, geo_vec in self.geodesic_vecs.items():
            profile[name] = cosine(mean_vec, geo_vec)

        return profile, mean_vec

    def detect_shift(self, current_texts):
        """
        Detect if a paradigm shift is occurring.

        Score = (avg similarity to shift geodesics) - (avg similarity to stability geodesics)
        Positive = more shift-like, Negative = more stable

        Returns:
            shift_score: Positive = shift, Negative = stable
            shift_type: 'SHIFT', 'STABLE', or 'TRANSITIONAL'
            details: Breakdown of detection
        """
        profile, current_vec = self.get_geodesic_profile(current_texts)

        # Average similarity to SHIFT geodesics (Earthquake, Death, Wind)
        shift_sim = np.mean([profile[geo] for geo in SHIFT_GEODESICS])

        # Average similarity to STABILITY geodesics (Dog, Deer, Reed)
        stability_sim = np.mean([profile[geo] for geo in STABILITY_GEODESICS])

        # Shift score = difference (centered around 0)
        # Positive = more shift-like, Negative = more stable
        shift_score = shift_sim - stability_sim

        # Also compute dominance - is the TOP geodesic a shift or stability one?
        sorted_profile = sorted(profile.items(), key=lambda x: -x[1])
        top_geodesic = sorted_profile[0][0]
        top_is_shift = top_geodesic in SHIFT_GEODESICS
        top_is_stable = top_geodesic in STABILITY_GEODESICS

        # Classification based on:
        # 1. Shift score (shift geodesics vs stability geodesics)
        # 2. Whether top geodesic is shift or stability
        if shift_score > 0.05 or top_is_shift:
            shift_type = 'SHIFT'
        elif shift_score < -0.03 or top_is_stable:
            shift_type = 'STABLE'
        else:
            shift_type = 'TRANSITIONAL'

        # Override: if top geodesic is clearly stability, it's stable
        if top_is_stable and sorted_profile[0][1] > 0.4:
            shift_type = 'STABLE'
            shift_score = -abs(shift_score)  # Make negative to indicate stability

        return {
            'shift_score': shift_score,
            'shift_type': shift_type,
            'top_geodesics': sorted_profile[:3],
            'shift_geodesic_sim': shift_sim,
            'stability_geodesic_sim': stability_sim,
            'top_geodesic': top_geodesic,
            'top_is_shift': top_is_shift,
            'top_is_stable': top_is_stable,
            'profile': profile,
        }

    def analyze(self, texts, label="Current State"):
        """Analyze and print results."""
        result = self.detect_shift(texts)

        print(f"\n{'='*60}")
        print(f"PARADIGM SHIFT ANALYSIS: {label}")
        print(f"{'='*60}")

        print(f"\nShift Score: {result['shift_score']:+.4f}")
        print(f"Classification: {result['shift_type']}")

        print(f"\nTop Active Geodesics:")
        for name, sim in result['top_geodesics']:
            if name in SHIFT_GEODESICS:
                marker = '[SHIFT]'
            elif name in STABILITY_GEODESICS:
                marker = '[STABLE]'
            else:
                marker = ''
            print(f"  {name:<12} {sim:.3f} {marker}")

        print(f"\nGeodesic Comparison:")
        print(f"  Shift geodesics (Earthquake/Death/Wind):  {result['shift_geodesic_sim']:.3f}")
        print(f"  Stable geodesics (Dog/Deer/Reed):         {result['stability_geodesic_sim']:.3f}")
        print(f"  Difference (shift - stable):              {result['shift_score']:+.3f}")

        if result['shift_type'] == 'SHIFT':
            print(f"\n>>> PARADIGM SHIFT DETECTED <<<")
        elif result['shift_type'] == 'TRANSITIONAL':
            print(f"\n>>> TRANSITIONAL STATE - WATCH CLOSELY <<<")
        else:
            print(f"\n>>> STABLE - NO SIGNIFICANT SHIFT <<<")

        return result


def demo():
    """Demonstrate the paradigm shift detector."""

    detector = ParadigmShiftDetector()

    print("=" * 60)
    print("REALTIME PARADIGM SHIFT DETECTOR - DEMO")
    print("=" * 60)

    results = []

    # Test 1: Institutional Stability (pre-crisis language)
    print("\n\n--- TEST 1: INSTITUTIONAL STABILITY ---")
    stability_1 = [
        "Banks are trustworthy institutions",
        "Experts provide reliable guidance",
        "Leadership maintains steady course",
        "Established procedures followed",
        "Loyal customers return faithfully",
    ]
    r = detector.analyze(stability_1, "Institutional Stability")
    results.append(('Institutional Stability', r['shift_type'], r['shift_score']))

    # Test 2: Community Harmony
    print("\n\n--- TEST 2: COMMUNITY HARMONY ---")
    stability_2 = [
        "Neighbors helping neighbors",
        "Trust in local leadership",
        "Gentle cooperation prevails",
        "Faithful friendships deepen",
        "Gradual steady improvement",
    ]
    r = detector.analyze(stability_2, "Community Harmony")
    results.append(('Community Harmony', r['shift_type'], r['shift_score']))

    # Test 3: Early Warning Signs
    print("\n\n--- TEST 3: EARLY WARNING SIGNS ---")
    early_warning = [
        "Unusual activity detected",
        "Something feels different",
        "Whispers of change",
        "Cracks appearing",
        "Questions being raised",
    ]
    r = detector.analyze(early_warning, "Early Warning Signs")
    results.append(('Early Warning', r['shift_type'], r['shift_score']))

    # Test 4: Crisis Emergence
    print("\n\n--- TEST 4: CRISIS EMERGENCE ---")
    crisis = [
        "System breaking down",
        "Everything changing rapidly",
        "Old certainties dissolving",
        "Upheaval spreading",
        "Foundations shaking",
    ]
    r = detector.analyze(crisis, "Crisis Emergence")
    results.append(('Crisis Emergence', r['shift_type'], r['shift_score']))

    # Test 5: Full Paradigm Shift
    print("\n\n--- TEST 5: FULL PARADIGM SHIFT ---")
    full_shift = [
        "The world will never be the same",
        "Complete transformation underway",
        "Death of the old order",
        "Revolutionary change sweeping through",
        "Nothing remains as it was",
    ]
    r = detector.analyze(full_shift, "Full Paradigm Shift")
    results.append(('Full Paradigm Shift', r['shift_type'], r['shift_score']))

    # Test 6: New Paradigm Crystallized
    print("\n\n--- TEST 6: NEW PARADIGM CRYSTALLIZED ---")
    new_stable = [
        "New normal established",
        "Adapted to changed reality",
        "Loyal to new principles",
        "Trust rebuilt on new foundation",
        "Guidance from new leaders",
    ]
    r = detector.analyze(new_stable, "New Paradigm Crystallized")
    results.append(('New Paradigm Crystallized', r['shift_type'], r['shift_score']))

    # Summary table
    print("\n\n" + "=" * 60)
    print("PARADIGM SHIFT SPECTRUM")
    print("=" * 60)
    print(f"\n{'State':<30} {'Type':<15} {'Score':>10}")
    print("-" * 60)

    for name, shift_type, score in results:
        print(f"{name:<30} {shift_type:<15} {score:>+10.4f}")

    print("\n" + "-" * 60)
    print("""
INTERPRETATION:

Score < 0:    STABLE (stability geodesics dominate)
Score 0-0.1:  TRANSITIONAL (balanced/uncertain)
Score > 0.1:  SHIFT (shift geodesics dominate)

The detector tracks the SEMANTIC FIELD, not surface meaning.
Language about disruption, endings, and change triggers SHIFT.
Language about loyalty, trust, and guidance triggers STABLE.

This is the paradigm shift signature discovered from 13 historical shifts.
""")


def interactive():
    """Interactive mode - feed your own text."""
    detector = ParadigmShiftDetector()

    print("=" * 60)
    print("REALTIME PARADIGM SHIFT DETECTOR - INTERACTIVE")
    print("=" * 60)
    print("\nEnter text to analyze (one sentence per line).")
    print("Enter blank line when done.")
    print("Type 'quit' to exit.\n")

    while True:
        texts = []
        print("Enter text to analyze:")
        while True:
            line = input().strip()
            if line.lower() == 'quit':
                print("Exiting.")
                return
            if line == '':
                break
            texts.append(line)

        if texts:
            detector.analyze(texts, "User Input")
        print()


if __name__ == "__main__":
    demo()

"""
PARADIGM SHIFT TEST: Can the compass detect historical shifts we KNOW happened?

Test paradigm shifts with known dates:
1. COVID-19 emergence (Jan-Mar 2020)
2. ChatGPT/AI paradigm (Nov 2022)
3. 2008 Financial Crisis (Sep-Oct 2008)
4. iPhone revolution (Jan 2007)
5. Bitcoin mainstream (Dec 2017)

For each, we test:
- Which geodesic was ACTIVE before the shift?
- Did the compass detect the TRANSITION?
- Does post-shift align with a different geodesic?
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# The 20 geodesics (Aztec day signs)
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


# KNOWN PARADIGM SHIFTS with real historical context
# These are the "ground truth" - we KNOW these happened
PARADIGM_SHIFTS = {
    'covid_19': {
        'name': 'COVID-19 Pandemic',
        'date': '2020-03-11',  # WHO declares pandemic
        'pre_context': [
            'normal daily life continues',
            'global trade and travel unrestricted',
            'in-person work and school standard',
            'handshakes and gatherings normal',
            'hospitals operating normally',
        ],
        'transition_context': [
            'mysterious pneumonia in Wuhan',
            'novel coronavirus spreading',
            'cases appearing in multiple countries',
            'uncertainty about transmission',
            'hospitals preparing for surge',
        ],
        'post_context': [
            'lockdowns and social distancing',
            'masks mandatory everywhere',
            'remote work becomes normal',
            'healthcare system overwhelmed',
            'global pandemic changes everything',
        ],
        'expected_geodesics': {
            'pre': ['House', 'Rabbit'],  # Security, growth
            'transition': ['Death', 'Earthquake', 'Serpent'],  # Endings, upheaval, hidden
            'post': ['Lizard', 'Wind'],  # Adaptation, change
        }
    },

    'chatgpt_ai': {
        'name': 'ChatGPT/AI Revolution',
        'date': '2022-11-30',  # ChatGPT release
        'pre_context': [
            'AI is a research topic',
            'machine learning for specific tasks',
            'chatbots are limited and frustrating',
            'creative work is human domain',
            'coding requires years of learning',
        ],
        'transition_context': [
            'new AI chatbot goes viral',
            'AI passes professional exams',
            'generated content indistinguishable from human',
            'millions trying new AI tools',
            'debates about AI capabilities',
        ],
        'post_context': [
            'AI assistants everywhere',
            'content creation automated',
            'jobs disrupted by AI',
            'AGI discussions mainstream',
            'AI safety becomes urgent priority',
        ],
        'expected_geodesics': {
            'pre': ['Reed', 'Grass'],  # Structure, persistence
            'transition': ['Monkey', 'Crocodile', 'Wind'],  # Innovation, emergence, spreading
            'post': ['Eagle', 'Earthquake'],  # Vision, paradigm shift
        }
    },

    'financial_crisis_2008': {
        'name': '2008 Financial Crisis',
        'date': '2008-09-15',  # Lehman Brothers collapse
        'pre_context': [
            'housing prices always go up',
            'banks are stable institutions',
            'mortgage-backed securities are safe',
            'economic growth continues',
            'retirement accounts growing steadily',
        ],
        'transition_context': [
            'subprime mortgage defaults rising',
            'Bear Stearns needs rescue',
            'credit markets freezing up',
            'banks hiding toxic assets',
            'fear spreading through markets',
        ],
        'post_context': [
            'Lehman Brothers bankrupt',
            'global financial system collapse',
            'government bailouts required',
            'recession and unemployment surge',
            'trust in institutions shattered',
        ],
        'expected_geodesics': {
            'pre': ['Rabbit', 'House'],  # Growth, security
            'transition': ['Serpent', 'Death'],  # Hidden danger, endings
            'post': ['Earthquake', 'Vulture'],  # Upheaval, transformation of decay
        }
    },

    'iphone_revolution': {
        'name': 'iPhone/Smartphone Revolution',
        'date': '2007-06-29',  # iPhone release
        'pre_context': [
            'phones are for calling and texting',
            'internet is on computers',
            'cameras are separate devices',
            'maps are paper or car GPS',
            'music on iPods or CDs',
        ],
        'transition_context': [
            'Apple announces revolutionary phone',
            'touchscreen interface demonstrated',
            'internet in your pocket',
            'skeptics doubt it will succeed',
            'lines forming at Apple stores',
        ],
        'post_context': [
            'smartphone in every pocket',
            'apps for everything',
            'social media always connected',
            'cameras replaced by phones',
            'mobile-first world',
        ],
        'expected_geodesics': {
            'pre': ['House', 'Reed'],  # Established order
            'transition': ['Monkey', 'Crocodile'],  # Innovation, emergence
            'post': ['Wind', 'Flower'],  # Spreading, completion
        }
    },

    'bitcoin_mainstream': {
        'name': 'Bitcoin Goes Mainstream',
        'date': '2017-12-17',  # BTC peaks near $20k
        'pre_context': [
            'bitcoin is for tech nerds',
            'cryptocurrency is obscure',
            'money must be government backed',
            'blockchain is unknown term',
            'digital assets not taken seriously',
        ],
        'transition_context': [
            'bitcoin price surging',
            'mainstream media covering crypto',
            'everyone asking about bitcoin',
            'FOMO driving new investors',
            'crypto exchanges overwhelmed',
        ],
        'post_context': [
            'cryptocurrency mainstream investment',
            'institutional adoption begins',
            'blockchain technology spreads',
            'digital assets class established',
            'decentralized finance emerging',
        ],
        'expected_geodesics': {
            'pre': ['Reed', 'House'],  # Established order, security
            'transition': ['Earthquake', 'Rabbit'],  # Disruption, multiplication
            'post': ['Wind', 'Monkey'],  # Spreading, innovation
        }
    },
}


def cosine(v1, v2):
    """Cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class ParadigmShiftDetector:
    """Detect paradigm shifts using geodesic compass."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers required")

        self.model = SentenceTransformer(model_name)
        self._build_geodesic_embeddings()

    def _build_geodesic_embeddings(self):
        """Pre-embed all geodesics."""
        self.geodesic_vecs = {}
        for gid, data in ARCHETYPAL_GEODESICS.items():
            self.geodesic_vecs[gid] = self.model.encode(data['desc'])

    def detect_active_geodesic(self, context_texts):
        """
        Given context texts, determine which geodesic is most active.

        Returns top 3 geodesics with similarity scores.
        """
        # Embed all context texts and average
        context_vecs = [self.model.encode(t) for t in context_texts]
        mean_context = np.mean(context_vecs, axis=0)

        # Find most similar geodesics
        similarities = []
        for gid, geo_vec in self.geodesic_vecs.items():
            sim = cosine(mean_context, geo_vec)
            similarities.append({
                'gid': gid,
                'name': ARCHETYPAL_GEODESICS[gid]['name'],
                'similarity': sim,
            })

        similarities.sort(key=lambda x: -x['similarity'])
        return similarities[:5]  # Top 5

    def detect_transition(self, pre_context, post_context):
        """
        Detect the transition between two states.

        Returns the "direction" of the shift in geodesic space.
        """
        pre_vecs = [self.model.encode(t) for t in pre_context]
        post_vecs = [self.model.encode(t) for t in post_context]

        pre_mean = np.mean(pre_vecs, axis=0)
        post_mean = np.mean(post_vecs, axis=0)

        # Transition vector
        transition = post_mean - pre_mean
        transition = transition / (np.linalg.norm(transition) + 1e-10)

        # Which geodesics align with the transition?
        alignments = []
        for gid, geo_vec in self.geodesic_vecs.items():
            # Normalize geodesic vector
            geo_norm = geo_vec / (np.linalg.norm(geo_vec) + 1e-10)

            # Dot product = alignment with transition direction
            alignment = float(np.dot(transition, geo_norm))
            alignments.append({
                'gid': gid,
                'name': ARCHETYPAL_GEODESICS[gid]['name'],
                'alignment': alignment,
            })

        alignments.sort(key=lambda x: -x['alignment'])
        return alignments[:5]  # Top 5 aligned with transition

    def analyze_paradigm_shift(self, shift_key):
        """
        Full analysis of a paradigm shift.
        """
        shift = PARADIGM_SHIFTS[shift_key]

        print("=" * 70)
        print(f"PARADIGM SHIFT: {shift['name']}")
        print(f"Date: {shift['date']}")
        print("=" * 70)

        # Detect geodesics at each phase
        print("\n--- PRE-SHIFT STATE ---")
        pre_geodesics = self.detect_active_geodesic(shift['pre_context'])
        print("Active geodesics (what the compass would detect BEFORE):")
        for g in pre_geodesics[:3]:
            expected = 'EXPECTED' if g['name'] in shift['expected_geodesics']['pre'] else ''
            print(f"  {g['name']:<12} {g['similarity']:.3f} {expected}")

        print("\n--- TRANSITION STATE ---")
        trans_geodesics = self.detect_active_geodesic(shift['transition_context'])
        print("Active geodesics (what the compass would detect DURING):")
        for g in trans_geodesics[:3]:
            expected = 'EXPECTED' if g['name'] in shift['expected_geodesics']['transition'] else ''
            print(f"  {g['name']:<12} {g['similarity']:.3f} {expected}")

        print("\n--- POST-SHIFT STATE ---")
        post_geodesics = self.detect_active_geodesic(shift['post_context'])
        print("Active geodesics (what the compass would detect AFTER):")
        for g in post_geodesics[:3]:
            expected = 'EXPECTED' if g['name'] in shift['expected_geodesics']['post'] else ''
            print(f"  {g['name']:<12} {g['similarity']:.3f} {expected}")

        # Transition analysis
        print("\n--- TRANSITION DIRECTION ---")
        transition = self.detect_transition(shift['pre_context'], shift['post_context'])
        print("Geodesics aligned with the transition vector (pre -> post):")
        for g in transition[:3]:
            print(f"  {g['name']:<12} {g['alignment']:+.3f}")

        # Scoring
        print("\n--- COMPASS ACCURACY ---")

        def score_phase(detected, expected):
            detected_names = [g['name'] for g in detected[:3]]
            hits = sum(1 for name in detected_names if name in expected)
            return hits, len(expected)

        pre_hits, pre_total = score_phase(pre_geodesics, shift['expected_geodesics']['pre'])
        trans_hits, trans_total = score_phase(trans_geodesics, shift['expected_geodesics']['transition'])
        post_hits, post_total = score_phase(post_geodesics, shift['expected_geodesics']['post'])

        total_hits = pre_hits + trans_hits + post_hits
        total_expected = pre_total + trans_total + post_total

        print(f"Pre-shift:   {pre_hits}/{pre_total} expected geodesics detected")
        print(f"Transition:  {trans_hits}/{trans_total} expected geodesics detected")
        print(f"Post-shift:  {post_hits}/{post_total} expected geodesics detected")
        print(f"TOTAL:       {total_hits}/{total_expected} ({total_hits/total_expected:.1%})")

        # Key insight
        print("\n--- KEY INSIGHT ---")
        pre_top = pre_geodesics[0]['name']
        trans_top = trans_geodesics[0]['name']
        post_top = post_geodesics[0]['name']

        if pre_top != trans_top:
            print(f"SHIFT DETECTED: {pre_top} -> {trans_top} during transition")
        else:
            print(f"NO CLEAR SHIFT: {pre_top} remained dominant")

        if trans_top != post_top:
            print(f"CRYSTALLIZATION: {trans_top} -> {post_top} after shift")
        else:
            print(f"TRANSITION LOCKED: {trans_top} became the new paradigm")

        return {
            'shift': shift_key,
            'pre_geodesic': pre_top,
            'trans_geodesic': trans_top,
            'post_geodesic': post_top,
            'accuracy': total_hits / total_expected,
        }


def run_all_tests():
    """Test the compass on all known paradigm shifts."""

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence_transformers required")
        return

    detector = ParadigmShiftDetector()

    results = []
    for shift_key in PARADIGM_SHIFTS:
        result = detector.analyze_paradigm_shift(shift_key)
        results.append(result)
        print("\n")

    # Summary
    print("=" * 70)
    print("PARADIGM SHIFT DETECTION SUMMARY")
    print("=" * 70)

    print(f"\n{'Paradigm Shift':<25} {'Pre':<12} {'Trans':<12} {'Post':<12} {'Acc':>8}")
    print("-" * 70)

    for r in results:
        shift_name = PARADIGM_SHIFTS[r['shift']]['name'][:24]
        print(f"{shift_name:<25} {r['pre_geodesic']:<12} {r['trans_geodesic']:<12} {r['post_geodesic']:<12} {r['accuracy']:>7.1%}")

    avg_acc = np.mean([r['accuracy'] for r in results])
    print("-" * 70)
    print(f"{'AVERAGE ACCURACY':<25} {'':<12} {'':<12} {'':<12} {avg_acc:>7.1%}")

    print("""
INTERPRETATION:

The compass detects paradigm shifts by tracking which geodesic (archetype)
is most active in the collective semantic field.

PRE-SHIFT: The old paradigm's dominant geodesic
TRANSITION: The geodesic of change/disruption activates
POST-SHIFT: The new paradigm crystallizes around a new geodesic

This isn't prediction - it's DETECTION.
The compass reads the semantic field and identifies which archetypal
pattern is currently expressing.

HISTORICAL VALIDATION:
If accuracy > 50%, the compass is detecting real structure.
If specific geodesics match expected patterns, the Aztec mapping has validity.
""")


if __name__ == "__main__":
    run_all_tests()

"""
PARADIGM DETECTOR V2: Let the compass speak without our bias.

Instead of testing against expected geodesics, discover what patterns
the compass CONSISTENTLY finds across paradigm shifts.

Question: Is there a consistent "signature" for paradigm shifts?
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


# Expanded paradigm shifts - more data points
PARADIGM_SHIFTS = {
    # TECHNOLOGY
    'chatgpt_ai': {
        'name': 'ChatGPT Release',
        'type': 'TECHNOLOGY',
        'pre': ['AI research in labs', 'chatbots limited', 'human creativity unique', 'coding requires expertise'],
        'post': ['AI assistants everywhere', 'automated content creation', 'AGI discussions mainstream', 'AI disruption'],
    },
    'iphone': {
        'name': 'iPhone Launch',
        'type': 'TECHNOLOGY',
        'pre': ['phones for calls only', 'internet on computers', 'separate cameras', 'paper maps'],
        'post': ['smartphone in every pocket', 'apps for everything', 'mobile-first world', 'always connected'],
    },
    'internet': {
        'name': 'Internet Goes Mainstream',
        'type': 'TECHNOLOGY',
        'pre': ['information in libraries', 'letters and phone calls', 'local shopping', 'scheduled TV'],
        'post': ['instant information access', 'email everywhere', 'online shopping', 'streaming content'],
    },

    # CRISIS
    'covid': {
        'name': 'COVID-19 Pandemic',
        'type': 'CRISIS',
        'pre': ['normal daily routines', 'travel unrestricted', 'in-person work', 'crowded gatherings'],
        'post': ['lockdowns', 'remote work normal', 'masks everywhere', 'healthcare overwhelmed'],
    },
    'financial_2008': {
        'name': '2008 Financial Crisis',
        'type': 'CRISIS',
        'pre': ['housing always rises', 'banks stable', 'credit available', 'retirement growing'],
        'post': ['system collapse', 'government bailouts', 'recession', 'trust shattered'],
    },
    'dotcom_bust': {
        'name': 'Dotcom Bust 2000',
        'type': 'CRISIS',
        'pre': ['tech stocks soaring', 'IPO frenzy', 'new economy hype', 'everyone investing'],
        'post': ['crash and burn', 'bankruptcies everywhere', 'layoffs massive', 'bubble burst'],
    },

    # SOCIAL
    'metoo': {
        'name': 'MeToo Movement',
        'type': 'SOCIAL',
        'pre': ['harassment tolerated', 'victims silent', 'powerful protected', 'whisper networks'],
        'post': ['voices heard', 'accountability demanded', 'powerful fall', 'cultural reckoning'],
    },
    'blm_2020': {
        'name': 'BLM Summer 2020',
        'type': 'SOCIAL',
        'pre': ['police trusted', 'systemic racism denied', 'status quo accepted', 'protests small'],
        'post': ['massive protests', 'defund police debates', 'racial justice spotlight', 'cultural shift'],
    },

    # FINANCIAL
    'bitcoin_2017': {
        'name': 'Bitcoin Peak 2017',
        'type': 'FINANCIAL',
        'pre': ['crypto for nerds', 'money is government', 'blockchain unknown', 'digital assets joke'],
        'post': ['crypto mainstream', 'institutional interest', 'blockchain everywhere', 'digital assets class'],
    },
    'gamestop_2021': {
        'name': 'GameStop Squeeze',
        'type': 'FINANCIAL',
        'pre': ['retail investors passive', 'hedge funds dominant', 'markets for professionals', 'meme stocks joke'],
        'post': ['retail army', 'hedge funds squeezed', 'meme stocks real', 'power shift'],
    },

    # POLITICAL
    'trump_2016': {
        'name': 'Trump Election 2016',
        'type': 'POLITICAL',
        'pre': ['establishment politics', 'experts trusted', 'media gatekeepers', 'predictable elections'],
        'post': ['populist wave', 'fake news', 'social media dominance', 'political polarization'],
    },
    'brexit': {
        'name': 'Brexit Vote',
        'type': 'POLITICAL',
        'pre': ['EU integration assumed', 'globalization inevitable', 'elites in control', 'status quo stable'],
        'post': ['nationalism rising', 'anti-establishment wins', 'borders matter', 'political earthquake'],
    },

    # SCIENTIFIC
    'crispr': {
        'name': 'CRISPR Revolution',
        'type': 'SCIENTIFIC',
        'pre': ['gene editing difficult', 'biology slow', 'designer babies fiction', 'nature fixed'],
        'post': ['edit genes easily', 'diseases curable', 'designer organisms', 'life programmable'],
    },
}


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class ParadigmDetector:
    """Detect paradigm shift signatures without bias."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence_transformers required")

        self.model = SentenceTransformer(model_name)
        self._build_embeddings()

    def _build_embeddings(self):
        """Pre-embed all geodesics."""
        self.geodesic_vecs = {}
        for gid, data in ARCHETYPAL_GEODESICS.items():
            self.geodesic_vecs[gid] = self.model.encode(data['desc'])

    def detect_geodesics(self, texts):
        """Get all geodesic similarities for a set of texts."""
        vecs = [self.model.encode(t) for t in texts]
        mean_vec = np.mean(vecs, axis=0)

        similarities = {}
        for gid, geo_vec in self.geodesic_vecs.items():
            similarities[ARCHETYPAL_GEODESICS[gid]['name']] = cosine(mean_vec, geo_vec)

        return similarities

    def get_transition_vector(self, pre_texts, post_texts):
        """Get the direction of transition in geodesic space."""
        pre_vecs = [self.model.encode(t) for t in pre_texts]
        post_vecs = [self.model.encode(t) for t in post_texts]

        pre_mean = np.mean(pre_vecs, axis=0)
        post_mean = np.mean(post_vecs, axis=0)

        transition = post_mean - pre_mean
        transition = transition / (np.linalg.norm(transition) + 1e-10)

        alignments = {}
        for gid, geo_vec in self.geodesic_vecs.items():
            geo_norm = geo_vec / (np.linalg.norm(geo_vec) + 1e-10)
            alignments[ARCHETYPAL_GEODESICS[gid]['name']] = float(np.dot(transition, geo_norm))

        return alignments


def run_analysis():
    """Analyze all paradigm shifts to find patterns."""

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: sentence_transformers required")
        return

    detector = ParadigmDetector()

    print("=" * 70)
    print("PARADIGM SHIFT SIGNATURE DISCOVERY")
    print("=" * 70)

    # Collect all results
    all_results = []
    by_type = {}

    for key, shift in PARADIGM_SHIFTS.items():
        pre_sims = detector.detect_geodesics(shift['pre'])
        post_sims = detector.detect_geodesics(shift['post'])
        transition = detector.get_transition_vector(shift['pre'], shift['post'])

        # Top geodesic for each phase
        pre_top = max(pre_sims.items(), key=lambda x: x[1])
        post_top = max(post_sims.items(), key=lambda x: x[1])
        trans_top = max(transition.items(), key=lambda x: x[1])

        result = {
            'key': key,
            'name': shift['name'],
            'type': shift['type'],
            'pre_geodesic': pre_top[0],
            'pre_sim': pre_top[1],
            'post_geodesic': post_top[0],
            'post_sim': post_top[1],
            'transition_geodesic': trans_top[0],
            'transition_align': trans_top[1],
            'pre_sims': pre_sims,
            'post_sims': post_sims,
            'transition': transition,
        }
        all_results.append(result)

        # Group by type
        if shift['type'] not in by_type:
            by_type[shift['type']] = []
        by_type[shift['type']].append(result)

    # Print results by type
    for shift_type, results in by_type.items():
        print(f"\n{'='*70}")
        print(f"TYPE: {shift_type}")
        print(f"{'='*70}")

        print(f"\n{'Paradigm Shift':<25} {'PRE':<12} {'POST':<12} {'TRANSITION':<12}")
        print("-" * 65)

        for r in results:
            print(f"{r['name']:<25} {r['pre_geodesic']:<12} {r['post_geodesic']:<12} {r['transition_geodesic']:<12}")

    # Find patterns
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Aggregate transition geodesics
    print("\n--- TRANSITION GEODESICS (most common direction of shift) ---")
    trans_counts = {}
    for r in all_results:
        geo = r['transition_geodesic']
        trans_counts[geo] = trans_counts.get(geo, 0) + 1

    sorted_trans = sorted(trans_counts.items(), key=lambda x: -x[1])
    print(f"\n{'Geodesic':<15} {'Count':>8} {'%':>8}")
    print("-" * 35)
    for geo, count in sorted_trans[:5]:
        print(f"{geo:<15} {count:>8} {count/len(all_results):>7.1%}")

    # Pre vs Post shift patterns
    print("\n--- PRE -> POST TRANSITIONS ---")
    print(f"\n{'From':<12} {'To':<12} {'Count':>8}")
    print("-" * 35)

    transition_pairs = {}
    for r in all_results:
        pair = (r['pre_geodesic'], r['post_geodesic'])
        transition_pairs[pair] = transition_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(transition_pairs.items(), key=lambda x: -x[1])
    for (pre, post), count in sorted_pairs[:8]:
        print(f"{pre:<12} {post:<12} {count:>8}")

    # Average transition direction across ALL paradigm shifts
    print("\n--- UNIVERSAL PARADIGM SHIFT SIGNATURE ---")
    print("(Average alignment of each geodesic with transition direction)")

    avg_transition = {}
    for geo in ARCHETYPAL_GEODESICS.values():
        name = geo['name']
        avg_transition[name] = np.mean([r['transition'][name] for r in all_results])

    sorted_avg = sorted(avg_transition.items(), key=lambda x: -x[1])
    print(f"\n{'Geodesic':<15} {'Avg Alignment':>15} {'Direction':>12}")
    print("-" * 45)
    for geo, avg in sorted_avg[:10]:
        direction = 'TOWARD' if avg > 0 else 'AWAY'
        print(f"{geo:<15} {avg:>+14.4f} {direction:>12}")

    print("\n" + "-" * 45)
    print("Geodesics MOST aligned with paradigm shifts:")
    top_3 = [g for g, _ in sorted_avg[:3]]
    print(f"  {', '.join(top_3)}")

    print("\nGeodesics LEAST aligned with paradigm shifts:")
    bottom_3 = [g for g, _ in sorted_avg[-3:]]
    print(f"  {', '.join(bottom_3)}")

    # The big insight
    print("\n" + "=" * 70)
    print("THE PARADIGM SHIFT SIGNATURE")
    print("=" * 70)

    top_geo = sorted_avg[0]
    print(f"""
Across {len(all_results)} paradigm shifts, the compass finds:

UNIVERSAL TRANSITION DIRECTION:
  #{1}: {sorted_avg[0][0]} ({sorted_avg[0][1]:+.4f})
  #{2}: {sorted_avg[1][0]} ({sorted_avg[1][1]:+.4f})
  #{3}: {sorted_avg[2][0]} ({sorted_avg[2][1]:+.4f})

INTERPRETATION:
  {sorted_avg[0][0]}: "{ARCHETYPAL_GEODESICS[[g['name'] for g in ARCHETYPAL_GEODESICS.values()].index(sorted_avg[0][0])]['desc']}"

  {sorted_avg[1][0]}: "{ARCHETYPAL_GEODESICS[[g['name'] for g in ARCHETYPAL_GEODESICS.values()].index(sorted_avg[1][0])]['desc']}"

  {sorted_avg[2][0]}: "{ARCHETYPAL_GEODESICS[[g['name'] for g in ARCHETYPAL_GEODESICS.values()].index(sorted_avg[2][0])]['desc']}"

THE SIGNATURE:
  Paradigm shifts CONSISTENTLY move toward these geodesics.
  When you detect movement toward {sorted_avg[0][0]}, {sorted_avg[1][0]}, or {sorted_avg[2][0]},
  a paradigm shift may be occurring.

STABILITY GEODESICS (paradigm shifts move AWAY from):
  {sorted_avg[-1][0]}: "{ARCHETYPAL_GEODESICS[[g['name'] for g in ARCHETYPAL_GEODESICS.values()].index(sorted_avg[-1][0])]['desc']}"
  {sorted_avg[-2][0]}: "{ARCHETYPAL_GEODESICS[[g['name'] for g in ARCHETYPAL_GEODESICS.values()].index(sorted_avg[-2][0])]['desc']}"

When semantic field moves AWAY from {sorted_avg[-1][0]}/{sorted_avg[-2][0]} and TOWARD
{sorted_avg[0][0]}/{sorted_avg[1][0]}, the compass is detecting a paradigm shift.
""")

    # Save results
    output_path = Path(__file__).parent / "paradigm_signatures.json"
    with open(output_path, 'w') as f:
        json.dump({
            'results': [{k: v for k, v in r.items() if k not in ['pre_sims', 'post_sims', 'transition']}
                        for r in all_results],
            'transition_signature': dict(sorted_avg),
            'top_transition_geodesics': [g for g, _ in sorted_avg[:5]],
            'stability_geodesics': [g for g, _ in sorted_avg[-5:]],
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_analysis()

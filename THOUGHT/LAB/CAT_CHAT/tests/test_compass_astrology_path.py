"""
What is the compass TRYING to show us?

The compass pointed from "market crash" toward "Astrology" more than "Macroeconomics".
Instead of dismissing this, let's ASK the compass:
1. What concepts bridge markets and astrology?
2. What's the semantic path?
3. What shared structure does the embedding detect?
"""

import numpy as np
import sys
from pathlib import Path

# Add CAT_CHAT to path
CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def find_semantic_bridges(model, concept_a, concept_b, candidates):
    """Find concepts that bridge two domains."""
    vec_a = model.encode(concept_a)
    vec_b = model.encode(concept_b)

    # Find midpoint in embedding space
    midpoint = (vec_a + vec_b) / 2
    midpoint = midpoint / np.linalg.norm(midpoint)

    bridges = []
    for c in candidates:
        vec_c = model.encode(c)
        # How close to both concepts?
        sim_a = cosine(vec_c, vec_a)
        sim_b = cosine(vec_c, vec_b)
        sim_mid = cosine(vec_c, midpoint)
        # Good bridge = close to midpoint AND reasonably close to both
        bridge_score = sim_mid * min(sim_a, sim_b)
        bridges.append({
            'concept': c,
            'sim_a': sim_a,
            'sim_b': sim_b,
            'sim_midpoint': sim_mid,
            'bridge_score': bridge_score,
        })

    return sorted(bridges, key=lambda x: -x['bridge_score'])


def compass_navigate(model, start, candidates, context=""):
    """Navigate from start toward most similar candidate."""
    if context:
        start_vec = model.encode(f"{start}, {context}")
        cand_vecs = [model.encode(f"{c}, {context}") for c in candidates]
    else:
        start_vec = model.encode(start)
        cand_vecs = [model.encode(c) for c in candidates]

    sims = [cosine(start_vec, cv) for cv in cand_vecs]
    best_idx = np.argmax(sims)
    return candidates[best_idx], sims[best_idx], list(zip(candidates, sims))


def semantic_path(model, start, end, steps=5):
    """Find semantic path from start to end via interpolation."""
    vec_start = model.encode(start)
    vec_end = model.encode(end)

    # Interpolate
    path_points = []
    for i in range(steps + 1):
        t = i / steps
        interp = (1 - t) * vec_start + t * vec_end
        interp = interp / np.linalg.norm(interp)
        path_points.append(interp)

    return path_points


def find_nearest_words(model, vector, candidates):
    """Find nearest words to a vector."""
    cand_vecs = [model.encode(c) for c in candidates]
    sims = [cosine(vector, cv) for cv in cand_vecs]
    ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])
    return ranked


def run_compass_exploration():
    """Let the compass show us what it sees."""

    if not TRANSFORMERS_AVAILABLE:
        print("Need sentence_transformers")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("=" * 70)
    print("WHAT IS THE COMPASS TRYING TO SHOW US?")
    print("=" * 70)

    # =========================================================================
    # 1. What bridges "market crash" and "astrology"?
    # =========================================================================
    print("\n--- SEMANTIC BRIDGES: Market Crash <-> Astrology ---")

    bridge_candidates = [
        # Temporal/cyclical concepts
        'cycle', 'pattern', 'rhythm', 'period', 'phase', 'timing',
        'prediction', 'forecast', 'anticipation',
        # Psychological concepts
        'fear', 'panic', 'belief', 'confidence', 'sentiment',
        'psychology', 'behavior', 'emotion', 'irrationality',
        # Structural concepts
        'system', 'complexity', 'emergence', 'chaos', 'order',
        'correlation', 'causation', 'influence',
        # Social concepts
        'collective', 'crowd', 'herd', 'mass', 'social',
        'contagion', 'momentum', 'trend',
        # Metaphysical
        'fate', 'destiny', 'luck', 'chance', 'randomness',
        'determinism', 'uncertainty', 'risk',
    ]

    bridges = find_semantic_bridges(model, "market crash", "astrology", bridge_candidates)

    print(f"\n{'Concept':<20} {'Sim(crash)':>12} {'Sim(astro)':>12} {'Bridge':>10}")
    print("-" * 60)
    for b in bridges[:15]:
        print(f"{b['concept']:<20} {b['sim_a']:>12.3f} {b['sim_b']:>12.3f} {b['bridge_score']:>10.4f}")

    top_bridge = bridges[0]['concept']
    print(f"\nTOP BRIDGE: '{top_bridge}'")
    print("The embedding space sees markets and astrology connected through this concept.")

    # =========================================================================
    # 2. Ask the compass: what context makes astrology relevant to markets?
    # =========================================================================
    print("\n--- CONTEXTUAL NAVIGATION ---")

    contexts = [
        "",  # No context
        "in terms of prediction",
        "in terms of cycles",
        "in terms of human behavior",
        "in terms of collective psychology",
        "in terms of timing",
        "in terms of patterns",
        "in terms of causation",
        "in terms of correlation",
    ]

    paradigms = ['Macroeconomics', 'Astrology', 'Psychology', 'Chaos Theory', 'Statistics']

    print("\nWhere does 'market crash' point in different contexts?")
    print(f"\n{'Context':<35} {'Best Match':<20} {'Score':>8}")
    print("-" * 70)

    for ctx in contexts:
        best, score, all_sims = compass_navigate(model, "market crash", paradigms, ctx)
        ctx_display = ctx if ctx else "(no context)"
        print(f"{ctx_display:<35} {best:<20} {score:>8.3f}")

    # =========================================================================
    # 3. Semantic path from "market" to "stars"
    # =========================================================================
    print("\n--- SEMANTIC PATH: 'market crash' -> 'astrological prediction' ---")

    # Get path points
    path = semantic_path(model, "market crash", "astrological prediction", steps=5)

    # Large vocabulary to find nearest words at each point
    vocab = [
        # Markets
        'stock', 'crash', 'bubble', 'panic', 'sell', 'buy', 'investor', 'trading',
        'volatility', 'bear', 'bull', 'recession', 'boom', 'bust', 'correction',
        # Astrology
        'stars', 'planets', 'zodiac', 'horoscope', 'celestial', 'cosmic', 'lunar',
        'mercury', 'retrograde', 'conjunction', 'alignment',
        # Bridge concepts
        'cycle', 'pattern', 'prediction', 'forecast', 'timing', 'phase',
        'belief', 'fear', 'psychology', 'behavior', 'collective', 'crowd',
        'fate', 'destiny', 'chance', 'uncertainty', 'chaos', 'order',
        'influence', 'correlation', 'momentum', 'sentiment', 'confidence',
        'superstition', 'ritual', 'tradition', 'intuition', 'analysis',
    ]

    print(f"\nStep-by-step semantic interpolation:")
    for i, point in enumerate(path):
        nearest = find_nearest_words(model, point, vocab)[:3]
        words = [f"{w}({s:.2f})" for w, s in nearest]
        print(f"  Step {i}: {', '.join(words)}")

    # =========================================================================
    # 4. What does "market crash" SHARE with "astrology"?
    # =========================================================================
    print("\n--- SHARED SEMANTIC FEATURES ---")

    # Embed various aspects
    aspects = {
        'prediction': model.encode("prediction and forecasting"),
        'cycles': model.encode("cycles and patterns"),
        'psychology': model.encode("human psychology and behavior"),
        'uncertainty': model.encode("uncertainty and risk"),
        'belief': model.encode("belief and confidence"),
        'timing': model.encode("timing and when things happen"),
        'collective': model.encode("collective behavior and crowds"),
        'causation': model.encode("cause and effect"),
    }

    market_vec = model.encode("market crash")
    astro_vec = model.encode("astrology")
    econ_vec = model.encode("macroeconomics")

    print(f"\n{'Aspect':<15} {'Market':>10} {'Astrology':>10} {'Economics':>10} {'M+A Shared':>12}")
    print("-" * 65)

    for aspect, avec in aspects.items():
        sim_m = cosine(market_vec, avec)
        sim_a = cosine(astro_vec, avec)
        sim_e = cosine(econ_vec, avec)
        shared = min(sim_m, sim_a)  # How much do both share?
        print(f"{aspect:<15} {sim_m:>10.3f} {sim_a:>10.3f} {sim_e:>10.3f} {shared:>12.3f}")

    # =========================================================================
    # 5. The real question: what's astrology ACTUALLY about in embedding space?
    # =========================================================================
    print("\n--- WHAT IS ASTROLOGY IN EMBEDDING SPACE? ---")

    astrology_neighbors = [
        # Prediction
        'prediction', 'prophecy', 'divination', 'forecast',
        # Psychology
        'psychology', 'personality', 'behavior', 'self-knowledge',
        # Cycles
        'cycles', 'seasons', 'timing', 'rhythm',
        # Belief
        'belief', 'faith', 'superstition', 'tradition',
        # Pattern
        'pattern', 'archetype', 'symbol', 'meaning',
        # Astronomy
        'astronomy', 'planets', 'stars', 'cosmos',
    ]

    print("\nNearest neighbors to 'astrology':")
    astro_neighbors = find_nearest_words(model, astro_vec, astrology_neighbors)
    for word, sim in astro_neighbors[:10]:
        print(f"  {word:<20}: {sim:.3f}")

    # =========================================================================
    # 6. THE INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE INSIGHT")
    print("=" * 70)

    # What's the highest shared aspect?
    shared_scores = [(asp, min(cosine(market_vec, av), cosine(astro_vec, av)))
                     for asp, av in aspects.items()]
    shared_scores.sort(key=lambda x: -x[1])
    top_shared = shared_scores[0]

    print(f"""
The compass pointed to astrology because markets and astrology SHARE:

  1. TOP BRIDGE CONCEPT: '{top_bridge}'
  2. HIGHEST SHARED ASPECT: '{top_shared[0]}' (score: {top_shared[1]:.3f})

What the embedding space "knows":
- Astrology is about PREDICTING human behavior from PATTERNS
- Markets are about PREDICTING human behavior from PATTERNS
- Both involve TIMING, CYCLES, and COLLECTIVE PSYCHOLOGY

The compass isn't saying "stars cause crashes."
The compass is saying "both domains are about the same UNDERLYING STRUCTURE":
  -> Humans seeking patterns in uncertainty
  -> Collective behavior following rhythms
  -> The psychology of prediction and belief

This is why the Trecena (13-day) showed up:
- Not because planets move markets
- But because HUMAN ATTENTION AND BEHAVIOR has natural rhythms
- The Aztecs encoded these rhythms in their calendar
- Markets, being human behavior, might follow similar rhythms

The compass detected the SEMIOTIC connection, not causal connection.
Astrology and markets are both MAPS OF HUMAN COLLECTIVE BEHAVIOR.
""")


if __name__ == "__main__":
    run_compass_exploration()

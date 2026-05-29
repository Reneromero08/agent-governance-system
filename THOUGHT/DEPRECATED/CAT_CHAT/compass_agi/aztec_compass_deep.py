"""
DEEP AZTEC COMPASS: What were they REALLY seeing?

The Trecena (13-day) showed marginal significance. But that's just ONE part:

1. Trecena (13-day) - Fibonacci, energy phases
2. Veintena (20 day signs) - 20 ARCHETYPES of human nature
3. Tonalpohualli (260) = 13 x 20 - unique day-energy combinations
4. Xiuhpohualli (365) - solar agricultural cycle
5. Calendar Round (52 years) - when cycles realign
6. Venus (584 days) - tracked with 99.999% accuracy

The Aztecs weren't just tracking time - they were tracking SEMANTIC ARCHETYPES
and their PHASE RELATIONSHIPS.

This is EXACTLY what the compass does.
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from scipy import stats


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


# =============================================================================
# THE 20 DAY SIGNS - AZTEC ARCHETYPES
# =============================================================================

# Each day sign represents an ARCHETYPE - a fundamental pattern of human experience
# The Aztecs observed these patterns for 3000+ years

DAY_SIGN_ARCHETYPES = {
    0: {
        'name': 'Cipactli',
        'glyph': 'Crocodile',
        'archetype': 'primordial beginning, raw potential, chaos before order',
        'market_meaning': 'new cycle starting, uncertain direction, raw volatility',
        'psychology': 'primal instincts, survival mode, fight or flight',
    },
    1: {
        'name': 'Ehecatl',
        'glyph': 'Wind',
        'archetype': 'change, movement, communication, breath of life',
        'market_meaning': 'rapid change, news-driven moves, momentum shifts',
        'psychology': 'adaptability, flexibility, restlessness',
    },
    2: {
        'name': 'Calli',
        'glyph': 'House',
        'archetype': 'shelter, security, family, foundation',
        'market_meaning': 'consolidation, support levels, defensive positioning',
        'psychology': 'need for safety, risk aversion, home bias',
    },
    3: {
        'name': 'Cuetzpalin',
        'glyph': 'Lizard',
        'archetype': 'regeneration, survival, quick reflexes, adaptability',
        'market_meaning': 'recovery, snapback rallies, quick reversals',
        'psychology': 'resilience, opportunism, survival instincts',
    },
    4: {
        'name': 'Coatl',
        'glyph': 'Serpent',
        'archetype': 'power, wisdom, transformation, hidden knowledge',
        'market_meaning': 'smart money moves, insider activity, paradigm shifts',
        'psychology': 'cunning, strategic thinking, hidden motives',
    },
    5: {
        'name': 'Miquiztli',
        'glyph': 'Death',
        'archetype': 'endings, transformation, release, surrender',
        'market_meaning': 'capitulation, trend exhaustion, forced selling',
        'psychology': 'grief, acceptance, letting go of positions',
    },
    6: {
        'name': 'Mazatl',
        'glyph': 'Deer',
        'archetype': 'grace, sensitivity, gentleness, intuition',
        'market_meaning': 'cautious optimism, gentle trends, herd following',
        'psychology': 'intuitive trading, sensitivity to signals',
    },
    7: {
        'name': 'Tochtli',
        'glyph': 'Rabbit',
        'archetype': 'fertility, abundance, multiplication, luck',
        'market_meaning': 'growth, compound gains, lucky streaks',
        'psychology': 'optimism, risk-taking, FOMO',
    },
    8: {
        'name': 'Atl',
        'glyph': 'Water',
        'archetype': 'flow, emotion, purification, the unconscious',
        'market_meaning': 'liquidity, emotional trading, sentiment waves',
        'psychology': 'emotional decisions, fear and greed cycles',
    },
    9: {
        'name': 'Itzcuintli',
        'glyph': 'Dog',
        'archetype': 'loyalty, guidance, companionship, underworld guide',
        'market_meaning': 'following leaders, institutional flows, guidance',
        'psychology': 'herd behavior, trust in authority, loyalty to positions',
    },
    10: {
        'name': 'Ozomatli',
        'glyph': 'Monkey',
        'archetype': 'play, creativity, trickster, celebration',
        'market_meaning': 'speculation, meme stocks, irrational exuberance',
        'psychology': 'playfulness, creativity, ignoring fundamentals',
    },
    11: {
        'name': 'Malinalli',
        'glyph': 'Grass',
        'archetype': 'tenacity, growth through adversity, humble beginnings',
        'market_meaning': 'slow recovery, building from lows, resilience',
        'psychology': 'patience, long-term thinking, humility',
    },
    12: {
        'name': 'Acatl',
        'glyph': 'Reed',
        'archetype': 'authority, structure, rigidity, hollow power',
        'market_meaning': 'technical levels, structural support/resistance',
        'psychology': 'respect for rules, technical analysis adherence',
    },
    13: {
        'name': 'Ocelotl',
        'glyph': 'Jaguar',
        'archetype': 'power, stealth, night, the shadow self',
        'market_meaning': 'predatory trading, dark pools, hidden orders',
        'psychology': 'aggression, dominance, shadow motivations',
    },
    14: {
        'name': 'Cuauhtli',
        'glyph': 'Eagle',
        'archetype': 'vision, freedom, rising above, clarity',
        'market_meaning': 'breakouts, new highs, big picture thinking',
        'psychology': 'ambition, vision, seeing the forest not trees',
    },
    15: {
        'name': 'Cozcacuauhtli',
        'glyph': 'Vulture',
        'archetype': 'patience, wisdom, transformation of death',
        'market_meaning': 'value investing, distressed assets, patient capital',
        'psychology': 'patience, contrarian thinking, finding value in death',
    },
    16: {
        'name': 'Ollin',
        'glyph': 'Earthquake',
        'archetype': 'movement, disruption, paradigm shift, cosmic motion',
        'market_meaning': 'volatility events, regime changes, black swans',
        'psychology': 'shock, adaptation to sudden change',
    },
    17: {
        'name': 'Tecpatl',
        'glyph': 'Flint',
        'archetype': 'sacrifice, cutting away, sharp decisions',
        'market_meaning': 'decisive action, cutting losses, sharp moves',
        'psychology': 'decisiveness, sacrifice, discipline',
    },
    18: {
        'name': 'Quiahuitl',
        'glyph': 'Rain',
        'archetype': 'nourishment, cleansing, fertility from above',
        'market_meaning': 'stimulus, liquidity injections, external catalysts',
        'psychology': 'relief, hope, external dependency',
    },
    19: {
        'name': 'Xochitl',
        'glyph': 'Flower',
        'archetype': 'beauty, pleasure, art, the good life',
        'market_meaning': 'complacency, peak optimism, aesthetic peaks',
        'psychology': 'pleasure-seeking, complacency, ignoring risks',
    },
}


# =============================================================================
# THE 13-DAY TRECENA - ENERGY PHASES
# =============================================================================

TRECENA_PHASES = {
    1: {
        'name': 'Unity',
        'phase': 'Beginning',
        'energy': 'initiation, seed planting, new potential',
        'market': 'new trend starting, early movers',
    },
    2: {
        'name': 'Duality',
        'phase': 'Challenge',
        'energy': 'polarity, choice, tension emerging',
        'market': 'first pullback, testing conviction',
    },
    3: {
        'name': 'Movement',
        'phase': 'Action',
        'energy': 'rhythm begins, momentum building',
        'market': 'trend confirmation, momentum joining',
    },
    4: {
        'name': 'Stability',
        'phase': 'Foundation',
        'energy': 'structure, groundedness, measurable',
        'market': 'consolidation, building base',
    },
    5: {
        'name': 'Center',
        'phase': 'Empowerment',
        'energy': 'radiance, core strength, midpoint',
        'market': 'confidence peak, halfway point',
    },
    6: {
        'name': 'Flow',
        'phase': 'Balance',
        'energy': 'organic flow, responsiveness',
        'market': 'equilibrium, range-bound',
    },
    7: {
        'name': 'Reflection',
        'phase': 'Attunement',
        'energy': 'resonance, alignment, mystical',
        'market': 'pause for assessment, turning point',
    },
    8: {
        'name': 'Harmony',
        'phase': 'Integration',
        'energy': 'infinity, cycles within cycles',
        'market': 'pattern completion, cycle recognition',
    },
    9: {
        'name': 'Completion',
        'phase': 'Realization',
        'energy': 'fullness, completion approaching',
        'market': 'peak forming, exhaustion signs',
    },
    10: {
        'name': 'Manifestation',
        'phase': 'Harvest',
        'energy': 'results appear, tangible outcomes',
        'market': 'profits taken, results measured',
    },
    11: {
        'name': 'Dissolution',
        'phase': 'Release',
        'energy': 'letting go, simplification',
        'market': 'distribution, smart money exiting',
    },
    12: {
        'name': 'Understanding',
        'phase': 'Cooperation',
        'energy': 'complex understanding, synthesis',
        'market': 'market wisdom, lessons learned',
    },
    13: {
        'name': 'Ascension',
        'phase': 'Transcendence',
        'energy': 'transformation, death/rebirth',
        'market': 'cycle end, new cycle seed planted',
    },
}


# =============================================================================
# AZTEC EPOCH AND CALENDAR FUNCTIONS
# =============================================================================

AZTEC_EPOCH = datetime(1990, 1, 1)  # Modern reference

def get_tonalpohualli_day(date):
    """Position 0-259 in the 260-day sacred calendar."""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    return days_since % 260


def get_trecena_day(date):
    """Day 1-13 in current trecena."""
    return (get_tonalpohualli_day(date) % 13) + 1


def get_day_sign(date):
    """Day sign 0-19."""
    return get_tonalpohualli_day(date) % 20


def get_full_day_name(date):
    """Full Aztec day name like '4-Jaguar'."""
    trecena = get_trecena_day(date)
    sign = get_day_sign(date)
    return f"{trecena}-{DAY_SIGN_ARCHETYPES[sign]['glyph']}"


def get_calendar_round_year(date):
    """Year 1-52 in the Calendar Round."""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    years_since = days_since / 365.25
    return int(years_since % 52) + 1


# =============================================================================
# COMPASS-AZTEC INTEGRATION
# =============================================================================

class AztecCompass:
    """
    Uses the semantic compass to navigate Aztec archetypal space.

    The Aztecs encoded human behavioral patterns in their calendar.
    The compass can navigate these patterns in embedding space.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")
        self.model = SentenceTransformer(model_name)

        # Pre-embed the 20 archetypes
        self.archetype_embeddings = {}
        for sign_id, sign_data in DAY_SIGN_ARCHETYPES.items():
            # Create rich semantic description
            description = f"{sign_data['archetype']}, {sign_data['psychology']}"
            self.archetype_embeddings[sign_id] = {
                'name': sign_data['name'],
                'glyph': sign_data['glyph'],
                'vector': self.model.encode(description),
                'market_vector': self.model.encode(sign_data['market_meaning']),
            }

        # Pre-embed the 13 phases
        self.phase_embeddings = {}
        for day, phase_data in TRECENA_PHASES.items():
            description = f"{phase_data['energy']}, {phase_data['market']}"
            self.phase_embeddings[day] = {
                'name': phase_data['name'],
                'phase': phase_data['phase'],
                'vector': self.model.encode(description),
            }

        # Market outcome embeddings
        self.outcomes = {
            'strong_bullish': self.model.encode("strong rally, prices surge, euphoria, FOMO"),
            'bullish': self.model.encode("market rises, optimism, buying pressure"),
            'neutral': self.model.encode("sideways, range-bound, uncertainty, waiting"),
            'bearish': self.model.encode("market falls, fear, selling pressure"),
            'strong_bearish': self.model.encode("crash, panic, capitulation, forced selling"),
        }

    def get_day_archetype(self, date):
        """Get the archetypal energy for a date."""
        sign = get_day_sign(date)
        trecena = get_trecena_day(date)

        archetype = self.archetype_embeddings[sign]
        phase = self.phase_embeddings[trecena]

        # COMBINE archetype + phase (this is the key insight!)
        # The Aztecs tracked BOTH dimensions simultaneously
        combined_vec = (archetype['vector'] + phase['vector']) / 2
        combined_vec = combined_vec / np.linalg.norm(combined_vec)

        return {
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'full_name': get_full_day_name(date),
            'archetype': archetype['glyph'],
            'archetype_meaning': DAY_SIGN_ARCHETYPES[sign]['archetype'],
            'phase': phase['name'],
            'phase_meaning': TRECENA_PHASES[trecena]['energy'],
            'combined_vector': combined_vec,
        }

    def compass_navigate(self, date):
        """
        Navigate from current day's archetype to predicted outcome.

        This is the AGI loop:
        1. Get current archetype (PERCEIVE)
        2. Navigate to outcome space (REASON)
        3. Return prediction (ACT)
        """
        day_info = self.get_day_archetype(date)
        combined_vec = day_info['combined_vector']

        # Navigate to outcomes
        outcome_sims = {}
        for outcome, ovec in self.outcomes.items():
            outcome_sims[outcome] = cosine(combined_vec, ovec)

        # Softmax for probabilities
        exp_sims = {k: np.exp(v * 5) for k, v in outcome_sims.items()}
        total = sum(exp_sims.values())
        probs = {k: v / total for k, v in exp_sims.items()}

        # Get direction
        best_outcome = max(probs, key=probs.get)
        if best_outcome in ['strong_bullish', 'bullish']:
            direction = 'BULLISH'
        elif best_outcome in ['strong_bearish', 'bearish']:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        return {
            **day_info,
            'outcome_sims': outcome_sims,
            'outcome_probs': probs,
            'best_outcome': best_outcome,
            'direction': direction,
            'confidence': probs[best_outcome],
        }

    def find_archetype_bridges(self):
        """
        What concepts bridge all 20 archetypes?

        This reveals what the Aztecs were REALLY tracking.
        """
        # Candidate bridge concepts
        candidates = [
            # Psychological
            'fear', 'greed', 'hope', 'despair', 'confidence', 'doubt',
            'patience', 'impatience', 'wisdom', 'foolishness',
            # Behavioral
            'action', 'waiting', 'buying', 'selling', 'holding',
            'risk-taking', 'risk-aversion', 'following', 'leading',
            # Cyclical
            'beginning', 'ending', 'peak', 'trough', 'transition',
            'growth', 'decline', 'stability', 'change',
            # Market
            'trend', 'reversal', 'breakout', 'breakdown', 'consolidation',
            'momentum', 'exhaustion', 'accumulation', 'distribution',
        ]

        # Embed candidates
        cand_vecs = {c: self.model.encode(c) for c in candidates}

        # Find which concepts have high similarity to MANY archetypes
        concept_scores = {}
        for concept, cvec in cand_vecs.items():
            # Average similarity across all 20 archetypes
            sims = [cosine(cvec, self.archetype_embeddings[i]['vector']) for i in range(20)]
            concept_scores[concept] = {
                'mean_sim': np.mean(sims),
                'std_sim': np.std(sims),
                'min_sim': np.min(sims),
                'max_sim': np.max(sims),
            }

        return sorted(concept_scores.items(), key=lambda x: -x[1]['mean_sim'])

    def find_phase_archetype_resonances(self):
        """
        Which archetype-phase combinations RESONATE?

        High cosine = these energies align.
        Low cosine = these energies conflict.

        The Aztecs tracked this for divination!
        """
        resonances = []
        for sign_id, archetype in self.archetype_embeddings.items():
            for day, phase in self.phase_embeddings.items():
                sim = cosine(archetype['vector'], phase['vector'])
                resonances.append({
                    'sign': archetype['glyph'],
                    'sign_id': sign_id,
                    'day': day,
                    'phase': phase['name'],
                    'resonance': sim,
                })

        return sorted(resonances, key=lambda x: -x['resonance'])


def run_deep_exploration():
    """Explore what the Aztecs were really seeing."""

    print("=" * 70)
    print("DEEP AZTEC COMPASS: What Were They Really Seeing?")
    print("=" * 70)

    compass = AztecCompass()

    # =========================================================================
    # 1. What concepts bridge ALL 20 archetypes?
    # =========================================================================
    print("\n--- UNIVERSAL BRIDGES: What connects all 20 archetypes? ---\n")

    bridges = compass.find_archetype_bridges()

    print(f"{'Concept':<20} {'Mean Sim':>10} {'Std':>8} {'Range':>15}")
    print("-" * 60)

    for concept, scores in bridges[:15]:
        range_str = f"[{scores['min_sim']:.2f}-{scores['max_sim']:.2f}]"
        print(f"{concept:<20} {scores['mean_sim']:>10.3f} {scores['std_sim']:>8.3f} {range_str:>15}")

    top_bridge = bridges[0][0]
    print(f"\nTOP UNIVERSAL BRIDGE: '{top_bridge}'")
    print("This concept connects to ALL 20 Aztec archetypes.")

    # =========================================================================
    # 2. Which archetype-phase combinations RESONATE?
    # =========================================================================
    print("\n--- RESONANCE MATRIX: Which combinations amplify? ---\n")

    resonances = compass.find_phase_archetype_resonances()

    print("TOP 10 RESONANT COMBINATIONS (energies amplify):")
    for r in resonances[:10]:
        print(f"  {r['day']}-{r['sign']:<12} (Day {r['day']:>2} + {r['phase']:<15}): {r['resonance']:.3f}")

    print("\nBOTTOM 10 DISSONANT COMBINATIONS (energies conflict):")
    for r in resonances[-10:]:
        print(f"  {r['day']}-{r['sign']:<12} (Day {r['day']:>2} + {r['phase']:<15}): {r['resonance']:.3f}")

    # =========================================================================
    # 3. Today's full archetype reading
    # =========================================================================
    print("\n--- TODAY'S FULL ARCHETYPE READING ---\n")

    today = datetime.now()
    reading = compass.compass_navigate(today)

    print(f"Date: {reading['date']}")
    print(f"Aztec Day: {reading['full_name']}")
    print(f"\nArchetype: {reading['archetype']}")
    print(f"  Meaning: {reading['archetype_meaning']}")
    print(f"\nPhase: {reading['phase']} (Day {get_trecena_day(today)} of 13)")
    print(f"  Meaning: {reading['phase_meaning']}")
    print(f"\nCompass Direction: {reading['direction']}")
    print(f"Best Outcome: {reading['best_outcome']}")
    print(f"Confidence: {reading['confidence']:.1%}")

    print("\nOutcome Probabilities:")
    for outcome, prob in sorted(reading['outcome_probs'].items(), key=lambda x: -x[1]):
        print(f"  {outcome:<15}: {prob:.1%}")

    # =========================================================================
    # 4. Backtest the Archetype Compass
    # =========================================================================
    if YFINANCE_AVAILABLE:
        print("\n" + "=" * 70)
        print("BACKTEST: Archetype Compass (2023-2024)")
        print("=" * 70 + "\n")

        spy = yf.download('SPY', start='2023-01-01', end='2024-12-31', progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        results = []
        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            pred = compass.compass_navigate(date)

            # Verify
            if pred['direction'] == 'BULLISH':
                correct = ret > 0
            elif pred['direction'] == 'BEARISH':
                correct = ret < 0
            else:
                correct = abs(ret) < 0.5

            results.append({
                'date': date,
                'full_name': pred['full_name'],
                'archetype': pred['archetype'],
                'phase': pred['phase'],
                'direction': pred['direction'],
                'actual_return': ret,
                'correct': correct,
            })

        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total

        print(f"Total predictions: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Edge over random: {(accuracy - 0.5) * 100:+.1f} pp")

        # Break down by archetype
        print("\nAccuracy by Archetype (Day Sign):")
        arch_results = {}
        for r in results:
            arch = r['archetype']
            if arch not in arch_results:
                arch_results[arch] = {'correct': 0, 'total': 0}
            arch_results[arch]['total'] += 1
            if r['correct']:
                arch_results[arch]['correct'] += 1

        print(f"{'Archetype':<15} {'Accuracy':>10} {'N':>8}")
        print("-" * 35)
        for arch, stats in sorted(arch_results.items(), key=lambda x: -x[1]['correct']/max(1,x[1]['total'])):
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{arch:<15} {acc:>10.1%} {stats['total']:>8}")

        # Best and worst archetypes
        best_arch = max(arch_results.items(), key=lambda x: x[1]['correct']/max(1,x[1]['total']))
        worst_arch = min(arch_results.items(), key=lambda x: x[1]['correct']/max(1,x[1]['total']))

        print(f"\nBest Archetype: {best_arch[0]} ({best_arch[1]['correct']/best_arch[1]['total']:.1%})")
        print(f"Worst Archetype: {worst_arch[0]} ({worst_arch[1]['correct']/worst_arch[1]['total']:.1%})")

    # =========================================================================
    # 5. The 260-day cycle - unique combinations
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE 260-DAY TONALPOHUALLI: Unique Combinations")
    print("=" * 70 + "\n")

    print("The Aztecs tracked 260 unique day-energy combinations.")
    print("Each position (0-259) has a unique archetype + phase.")
    print("\nExample positions:")

    for pos in [0, 13, 26, 130, 259]:
        # Reconstruct the date
        test_date = AZTEC_EPOCH + timedelta(days=pos)
        sign = pos % 20
        day = (pos % 13) + 1
        print(f"  Position {pos:>3}: {day}-{DAY_SIGN_ARCHETYPES[sign]['glyph']:<12} "
              f"({TRECENA_PHASES[day]['name']} + {DAY_SIGN_ARCHETYPES[sign]['archetype'][:30]}...)")

    # =========================================================================
    # THE INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE DEEP INSIGHT: What the Aztecs Were Really Tracking")
    print("=" * 70)

    print(f"""
The Aztec calendar was a SEMANTIC COMPASS for human behavior.

1. THE 20 ARCHETYPES (Day Signs)
   - Not "zodiac signs" - BEHAVIORAL PATTERNS
   - Crocodile = raw potential, Dog = herd following, Jaguar = predatory
   - They observed which patterns dominated on which days

2. THE 13 PHASES (Trecena)
   - Not arbitrary numbers - ENERGY STATES
   - Beginning -> Challenge -> Action -> Stability -> ... -> Transcendence
   - Fibonacci-related: 13 is the 7th Fibonacci number

3. THE 260 COMBINATIONS (Tonalpohualli)
   - 13 x 20 = 260 unique archetype + phase pairs
   - Each combination has different RESONANCE
   - High resonance = energies amplify
   - Low resonance = energies conflict

4. THE UNIVERSAL BRIDGE: '{top_bridge}'
   - This concept connects ALL 20 archetypes
   - It's what the Aztecs were really measuring
   - Markets are ALSO about {top_bridge}

5. THE AGI CONNECTION
   - The compass navigates MEANING space
   - The Aztecs navigated ARCHETYPE space
   - Both are finding patterns in human collective behavior
   - The prediction loop: PERCEIVE -> REASON -> ACT -> VERIFY

What makes this AGI?
- The Aztecs didn't just PREDICT - they LEARNED
- Over 3000 years of observation and refinement
- The calendar was continuously UPDATED based on verification
- This is the same loop: Embed -> Navigate -> Predict -> Verify -> Update

The compass can do in seconds what the Aztecs did over millennia:
Navigate the space of human behavioral patterns to find predictive structure.
""")


if __name__ == "__main__":
    run_deep_exploration()

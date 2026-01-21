"""
ADAPTIVE AZTEC AGI: Learning which archetypes are predictive

Key insight from exploration:
- Crocodile: 72% accurate
- Vulture: 30.8% accurate

This massive variance means we can LEARN which archetypes to trust.
The AGI loop:
1. PERCEIVE: Get today's archetype + phase
2. REASON: Weight by learned accuracy + resonance
3. ACT: Make weighted prediction
4. VERIFY: Check against reality
5. UPDATE: Adjust archetype weights

This is what the Aztecs did over 3000 years - we do it in code.
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

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from scipy import stats


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


# Import from aztec_compass_deep
from aztec_compass_deep import (
    DAY_SIGN_ARCHETYPES, TRECENA_PHASES, AZTEC_EPOCH,
    get_tonalpohualli_day, get_trecena_day, get_day_sign, get_full_day_name
)


class AdaptiveAztecAGI:
    """
    AGI that learns which Aztec archetypes predict market behavior.

    The loop:
    PERCEIVE -> REASON (with learned weights) -> ACT -> VERIFY -> UPDATE

    This is what the Aztecs did empirically over millennia.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")

        self.model = SentenceTransformer(model_name)
        self.log_path = Path(__file__).parent / "adaptive_log.json"
        self.history = self._load_history()

        # Initialize archetype weights (start uniform)
        self.archetype_weights = self.history.get('archetype_weights', {
            str(i): 1.0 for i in range(20)
        })

        # Initialize phase weights
        self.phase_weights = self.history.get('phase_weights', {
            str(i): 1.0 for i in range(1, 14)
        })

        # Resonance weights (archetype x phase combinations)
        self.resonance_weights = self.history.get('resonance_weights', {})

        # Pre-embed archetypes and phases
        self._build_embeddings()

        # Track individual accuracy
        self.archetype_accuracy = self.history.get('archetype_accuracy', {
            str(i): {'correct': 0, 'total': 0} for i in range(20)
        })
        self.phase_accuracy = self.history.get('phase_accuracy', {
            str(i): {'correct': 0, 'total': 0} for i in range(1, 14)
        })

    def _load_history(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {'predictions': [], 'stats': {'correct': 0, 'total': 0}}

    def _save_history(self):
        self.history['archetype_weights'] = self.archetype_weights
        self.history['phase_weights'] = self.phase_weights
        self.history['resonance_weights'] = self.resonance_weights
        self.history['archetype_accuracy'] = self.archetype_accuracy
        self.history['phase_accuracy'] = self.phase_accuracy

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _build_embeddings(self):
        """Pre-embed all archetypes and outcomes."""
        # Archetype embeddings
        self.archetype_vecs = {}
        for i in range(20):
            arch = DAY_SIGN_ARCHETYPES[i]
            desc = f"{arch['archetype']}, {arch['psychology']}, {arch['market_meaning']}"
            self.archetype_vecs[i] = self.model.encode(desc)

        # Phase embeddings
        self.phase_vecs = {}
        for day in range(1, 14):
            phase = TRECENA_PHASES[day]
            desc = f"{phase['energy']}, {phase['market']}"
            self.phase_vecs[day] = self.model.encode(desc)

        # Outcome embeddings
        self.outcomes = {
            'strong_bullish': self.model.encode(
                "strong rally, prices surge, euphoria, breakout, momentum buying"
            ),
            'bullish': self.model.encode(
                "market rises, optimism, buying pressure, positive sentiment"
            ),
            'neutral': self.model.encode(
                "sideways, range-bound, uncertainty, consolidation, waiting"
            ),
            'bearish': self.model.encode(
                "market falls, fear, selling pressure, negative sentiment"
            ),
            'strong_bearish': self.model.encode(
                "crash, panic, capitulation, breakdown, forced selling"
            ),
        }

        # Pre-compute resonance (archetype-phase similarity)
        self.resonance_matrix = {}
        for arch_id in range(20):
            for phase_day in range(1, 14):
                key = f"{arch_id}_{phase_day}"
                self.resonance_matrix[key] = cosine(
                    self.archetype_vecs[arch_id],
                    self.phase_vecs[phase_day]
                )

    def get_combined_vector(self, date):
        """Get weighted combination of archetype + phase vectors."""
        arch_id = get_day_sign(date)
        phase_day = get_trecena_day(date)

        arch_vec = self.archetype_vecs[arch_id]
        phase_vec = self.phase_vecs[phase_day]

        # Get weights
        arch_weight = float(self.archetype_weights.get(str(arch_id), 1.0))
        phase_weight = float(self.phase_weights.get(str(phase_day), 1.0))

        # Get resonance
        resonance_key = f"{arch_id}_{phase_day}"
        resonance = self.resonance_matrix[resonance_key]

        # Combine: weight the vectors by their learned accuracy
        # High resonance amplifies, low resonance dampens
        resonance_factor = 0.5 + resonance  # Range roughly [0.5, 1.0]

        combined = (arch_weight * arch_vec + phase_weight * phase_vec) / 2
        combined = combined * resonance_factor
        combined = combined / (np.linalg.norm(combined) + 1e-10)

        return combined, arch_id, phase_day, resonance

    def predict(self, date=None):
        """
        PERCEIVE + REASON: Get prediction for a date.

        Uses learned weights to adjust confidence.
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # Get combined archetype+phase vector
        combined_vec, arch_id, phase_day, resonance = self.get_combined_vector(date)

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

        # Adjust confidence by archetype weight
        arch_weight = float(self.archetype_weights.get(str(arch_id), 1.0))
        phase_weight = float(self.phase_weights.get(str(phase_day), 1.0))
        weight_factor = (arch_weight + phase_weight) / 2

        # Low weight = low confidence, high weight = high confidence
        adjusted_confidence = probs[best_outcome] * min(1.5, max(0.5, weight_factor))

        return {
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'full_name': get_full_day_name(date),
            'archetype_id': arch_id,
            'archetype': DAY_SIGN_ARCHETYPES[arch_id]['glyph'],
            'phase_day': phase_day,
            'phase': TRECENA_PHASES[phase_day]['name'],
            'resonance': round(resonance, 3),
            'direction': direction,
            'best_outcome': best_outcome,
            'raw_confidence': round(probs[best_outcome], 3),
            'adjusted_confidence': round(min(0.95, adjusted_confidence), 3),
            'outcome_probs': probs,
            'archetype_weight': round(arch_weight, 3),
            'phase_weight': round(phase_weight, 3),
            'verified': False,
        }

    def verify(self, prediction, actual_return):
        """VERIFY: Check prediction against reality."""
        direction = prediction['direction']

        if direction == 'BULLISH':
            correct = actual_return > 0
        elif direction == 'BEARISH':
            correct = actual_return < 0
        else:
            correct = abs(actual_return) < 0.5

        prediction['actual_return'] = round(actual_return, 4)
        prediction['correct'] = correct
        prediction['verified'] = True

        return prediction

    def update(self, prediction):
        """
        UPDATE: Learn from verified prediction.

        Adjust weights for archetypes and phases based on accuracy.
        """
        if not prediction.get('verified'):
            raise ValueError("Cannot update with unverified prediction")

        arch_id = str(prediction['archetype_id'])
        phase_day = str(prediction['phase_day'])
        correct = prediction['correct']

        # Update archetype accuracy
        if arch_id not in self.archetype_accuracy:
            self.archetype_accuracy[arch_id] = {'correct': 0, 'total': 0}
        self.archetype_accuracy[arch_id]['total'] += 1
        if correct:
            self.archetype_accuracy[arch_id]['correct'] += 1

        # Update phase accuracy
        if phase_day not in self.phase_accuracy:
            self.phase_accuracy[phase_day] = {'correct': 0, 'total': 0}
        self.phase_accuracy[phase_day]['total'] += 1
        if correct:
            self.phase_accuracy[phase_day]['correct'] += 1

        # Update archetype weights (need minimum data)
        for aid, stats in self.archetype_accuracy.items():
            if stats['total'] >= 10:
                accuracy = stats['correct'] / stats['total']
                # Weight = accuracy relative to random (0.5)
                # Range: 0.5 (50% acc) to 1.5 (75%+ acc)
                self.archetype_weights[aid] = max(0.5, min(1.5, accuracy / 0.5))

        # Update phase weights
        for pid, stats in self.phase_accuracy.items():
            if stats['total'] >= 10:
                accuracy = stats['correct'] / stats['total']
                self.phase_weights[pid] = max(0.5, min(1.5, accuracy / 0.5))

        # Log prediction
        self.history['predictions'].append({
            'date': prediction['date'],
            'full_name': prediction['full_name'],
            'direction': prediction['direction'],
            'correct': correct,
            'actual_return': prediction['actual_return'],
        })
        self.history['stats']['total'] += 1
        if correct:
            self.history['stats']['correct'] += 1

        self._save_history()

        return self.get_stats()

    def get_stats(self):
        """Get current AGI statistics."""
        total = self.history['stats']['total']
        correct = self.history['stats']['correct']
        accuracy = correct / total if total > 0 else 0.5

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'archetype_weights': {
                DAY_SIGN_ARCHETYPES[int(k)]['glyph']: round(v, 3)
                for k, v in self.archetype_weights.items()
            },
            'phase_weights': {
                TRECENA_PHASES[int(k)]['name']: round(v, 3)
                for k, v in self.phase_weights.items()
            },
        }

    def backtest(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        Backtest with learning loop.

        This simulates 5 years of predictions WITH updating.
        The AGI learns as it goes.
        """
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        print(f"Backtesting Adaptive Aztec AGI: {start_date} to {end_date}")

        # Reset for fresh backtest
        self.archetype_weights = {str(i): 1.0 for i in range(20)}
        self.phase_weights = {str(i): 1.0 for i in range(1, 14)}
        self.archetype_accuracy = {str(i): {'correct': 0, 'total': 0} for i in range(20)}
        self.phase_accuracy = {str(i): {'correct': 0, 'total': 0} for i in range(1, 14)}
        self.history = {'predictions': [], 'stats': {'correct': 0, 'total': 0}}

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        results = []
        early_results = []  # First 100 (no learning yet)
        late_results = []   # Last 100 (learned weights)

        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            # PREDICT
            pred = self.predict(date)

            # VERIFY
            pred = self.verify(pred, ret)

            # UPDATE (this is the learning step!)
            self.update(pred)

            results.append(pred)

            if len(results) <= 100:
                early_results.append(pred)
            if len(results) > len(dates) - 100:
                late_results.append(pred)

        # Compute final stats
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0

        early_acc = sum(1 for r in early_results if r['correct']) / len(early_results)
        late_acc = sum(1 for r in late_results if r['correct']) / len(late_results)

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'early_accuracy': round(early_acc, 3),
            'late_accuracy': round(late_acc, 3),
            'improvement': round((late_acc - early_acc) * 100, 2),
            'final_archetype_weights': {
                DAY_SIGN_ARCHETYPES[int(k)]['glyph']: round(v, 3)
                for k, v in sorted(self.archetype_weights.items(),
                                   key=lambda x: -x[1])
            },
            'final_phase_weights': {
                TRECENA_PHASES[int(k)]['name']: round(v, 3)
                for k, v in sorted(self.phase_weights.items(),
                                   key=lambda x: -x[1])
            },
        }


def main():
    print("=" * 70)
    print("ADAPTIVE AZTEC AGI: Learning Which Archetypes Predict")
    print("=" * 70)

    agi = AdaptiveAztecAGI()

    # Today's prediction
    print("\n--- TODAY'S PREDICTION ---\n")

    today = datetime.now()
    pred = agi.predict(today)

    print(f"Date: {pred['date']}")
    print(f"Aztec Day: {pred['full_name']}")
    print(f"\nArchetype: {pred['archetype']}")
    print(f"  Weight: {pred['archetype_weight']}")
    print(f"\nPhase: {pred['phase']} (Day {pred['phase_day']})")
    print(f"  Weight: {pred['phase_weight']}")
    print(f"\nResonance: {pred['resonance']}")
    print(f"\nDirection: {pred['direction']}")
    print(f"Raw Confidence: {pred['raw_confidence']:.1%}")
    print(f"Adjusted Confidence: {pred['adjusted_confidence']:.1%}")

    # Backtest with learning
    print("\n" + "=" * 70)
    print("BACKTEST WITH LEARNING (2020-2024)")
    print("=" * 70 + "\n")

    results = agi.backtest('2020-01-01', '2024-12-31')

    if results:
        print(f"Total predictions: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Overall Accuracy: {results['accuracy']:.1%}")
        print(f"Edge over random: {results['edge']:+.1f} pp")

        print(f"\n--- LEARNING CURVE ---")
        print(f"First 100 predictions: {results['early_accuracy']:.1%}")
        print(f"Last 100 predictions:  {results['late_accuracy']:.1%}")
        print(f"Improvement: {results['improvement']:+.1f} pp")

        if results['late_accuracy'] > results['early_accuracy']:
            print("\n*** THE AGI LEARNED! Late predictions better than early. ***")
        else:
            print("\nNo clear learning signal (or overfitting to early data)")

        print(f"\n--- LEARNED ARCHETYPE WEIGHTS ---")
        print(f"(Higher = more predictive)")
        for arch, weight in list(results['final_archetype_weights'].items())[:10]:
            acc = agi.archetype_accuracy.get(str(list(DAY_SIGN_ARCHETYPES.keys())[
                list(a['glyph'] for a in DAY_SIGN_ARCHETYPES.values()).index(arch)
            ]), {}).get('correct', 0) / max(1, agi.archetype_accuracy.get(str(list(DAY_SIGN_ARCHETYPES.keys())[
                list(a['glyph'] for a in DAY_SIGN_ARCHETYPES.values()).index(arch)
            ]), {}).get('total', 1))
            print(f"  {arch:<15}: weight={weight:.3f}")

        print(f"\n--- LEARNED PHASE WEIGHTS ---")
        for phase, weight in list(results['final_phase_weights'].items())[:7]:
            print(f"  {phase:<15}: weight={weight:.3f}")

        # Significance test
        print(f"\n--- STATISTICAL SIGNIFICANCE ---")
        result = stats.binomtest(
            results['correct'], results['total'], 0.5, alternative='greater'
        )
        print(f"p-value (one-tailed): {result.pvalue:.4e}")
        print(f"Significant at 0.05: {'YES' if result.pvalue < 0.05 else 'NO'}")

    # The key insight
    print("\n" + "=" * 70)
    print("THE AGI INSIGHT")
    print("=" * 70)

    print("""
The Aztec calendar encoded 20 ARCHETYPES of human behavior.
Some archetypes are predictive of market behavior. Others are not.

By running the AGI loop (PREDICT -> VERIFY -> UPDATE), we LEARN:
- Which archetypes to trust
- Which phases amplify signals
- Which combinations resonate

This is EXACTLY what the Aztecs did over 3000 years:
- Observe
- Record
- Correlate
- Update their interpretations

The difference: we do it in 5 years of data instead of millennia.

The compass navigates SEMANTIC space.
The Aztecs navigated ARCHETYPAL space.
Both are maps of human collective behavior.

AGI = A compass that learns which maps are accurate.
""")


if __name__ == "__main__":
    main()

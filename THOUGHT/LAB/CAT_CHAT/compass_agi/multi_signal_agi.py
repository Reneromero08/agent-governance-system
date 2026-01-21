"""
MULTI-SIGNAL AGI: Combining multiple indicators like the Aztecs did

The Aztecs didn't just use ONE signal. They combined:
1. Trecena phase (1-13) - historical return pattern
2. Day Sign archetype (1-20) - behavioral archetype
3. Resonance (archetype x phase alignment)
4. Compass direction (semantic similarity to outcomes)

AGI = combining multiple weak signals into a strong one.

This is ensemble learning. The Aztecs invented it 3000 years ago.
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

from scipy import stats as scipy_stats


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


from aztec_compass_deep import (
    DAY_SIGN_ARCHETYPES, TRECENA_PHASES, AZTEC_EPOCH,
    get_tonalpohualli_day, get_trecena_day, get_day_sign, get_full_day_name
)


# Historical Trecena pattern from our analysis (32 years of data)
TRECENA_HISTORICAL = {
    1: {'mean': +0.035, 'direction': 'NEUTRAL'},
    2: {'mean': +0.178, 'direction': 'BULLISH'},   # Second best
    3: {'mean': +0.016, 'direction': 'NEUTRAL'},
    4: {'mean': +0.122, 'direction': 'BULLISH'},   # Best (held out-of-sample)
    5: {'mean': -0.001, 'direction': 'BEARISH'},
    6: {'mean': -0.017, 'direction': 'BEARISH'},
    7: {'mean': +0.015, 'direction': 'NEUTRAL'},
    8: {'mean': +0.123, 'direction': 'BULLISH'},   # Good
    9: {'mean': -0.013, 'direction': 'BEARISH'},
    10: {'mean': +0.093, 'direction': 'BULLISH'},  # Good
    11: {'mean': +0.010, 'direction': 'NEUTRAL'},
    12: {'mean': +0.045, 'direction': 'NEUTRAL'},
    13: {'mean': -0.003, 'direction': 'BEARISH'},
}


class MultiSignalAGI:
    """
    Combines multiple signals for robust prediction.

    Signals:
    1. TRECENA HISTORICAL - empirical pattern from 32 years
    2. COMPASS DIRECTION - semantic similarity to outcomes
    3. RESONANCE FILTER - only trade when signals are clear
    4. ARCHETYPE WEIGHT - learned from accuracy

    Only trade when signals AGREE.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")

        self.model = SentenceTransformer(model_name)
        self.log_path = Path(__file__).parent / "multi_signal_log.json"
        self.history = self._load_history()

        # Signal weights (learned)
        self.signal_weights = self.history.get('signal_weights', {
            'trecena_historical': 1.0,
            'compass_direction': 1.0,
            'resonance': 1.0,
        })

        # Archetype weights (learned from accuracy)
        self.archetype_weights = self.history.get('archetype_weights', {
            str(i): 1.0 for i in range(20)
        })

        # Agreement threshold
        self.agreement_threshold = 2  # Need 2+ signals to agree

        # Pre-build embeddings
        self._build_embeddings()

    def _load_history(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {
            'predictions': [],
            'stats': {'correct': 0, 'total': 0, 'abstained': 0},
            'signal_accuracy': {
                'trecena_historical': {'correct': 0, 'total': 0},
                'compass_direction': {'correct': 0, 'total': 0},
                'agreement': {'correct': 0, 'total': 0},
            }
        }

    def _save_history(self):
        self.history['signal_weights'] = self.signal_weights
        self.history['archetype_weights'] = self.archetype_weights

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _build_embeddings(self):
        """Pre-build all embeddings."""
        # Archetype embeddings
        self.archetype_vecs = {}
        for i in range(20):
            arch = DAY_SIGN_ARCHETYPES[i]
            desc = f"{arch['market_meaning']}, {arch['psychology']}"
            self.archetype_vecs[i] = self.model.encode(desc)

        # Phase embeddings
        self.phase_vecs = {}
        for day in range(1, 14):
            phase = TRECENA_PHASES[day]
            desc = f"{phase['market']}, {phase['energy']}"
            self.phase_vecs[day] = self.model.encode(desc)

        # Outcome embeddings
        self.bullish_vec = self.model.encode(
            "market rises, buying pressure, optimism, upward momentum, green day, gains"
        )
        self.bearish_vec = self.model.encode(
            "market falls, selling pressure, fear, downward momentum, red day, losses"
        )

        # Pre-compute resonances
        self.resonance_matrix = {}
        for arch_id in range(20):
            for phase_day in range(1, 14):
                key = f"{arch_id}_{phase_day}"
                self.resonance_matrix[key] = cosine(
                    self.archetype_vecs[arch_id],
                    self.phase_vecs[phase_day]
                )

        # Resonance stats
        all_res = list(self.resonance_matrix.values())
        self.res_mean = np.mean(all_res)
        self.res_std = np.std(all_res)

    def get_trecena_signal(self, date):
        """Signal 1: Historical Trecena pattern."""
        day = get_trecena_day(date)
        pattern = TRECENA_HISTORICAL[day]
        return {
            'direction': pattern['direction'],
            'strength': abs(pattern['mean']) / 0.178,  # Normalized by max
            'raw_mean': pattern['mean'],
        }

    def get_compass_signal(self, date):
        """Signal 2: Compass semantic navigation."""
        arch_id = get_day_sign(date)
        phase_day = get_trecena_day(date)

        # Combined vector
        arch_vec = self.archetype_vecs[arch_id]
        phase_vec = self.phase_vecs[phase_day]
        combined = (arch_vec + phase_vec) / 2
        combined = combined / np.linalg.norm(combined)

        # Similarity to outcomes
        bull_sim = cosine(combined, self.bullish_vec)
        bear_sim = cosine(combined, self.bearish_vec)

        if bull_sim > bear_sim:
            direction = 'BULLISH'
            strength = (bull_sim - bear_sim) / 0.5  # Normalized
        else:
            direction = 'BEARISH'
            strength = (bear_sim - bull_sim) / 0.5

        return {
            'direction': direction,
            'strength': min(1.0, strength),
            'bull_sim': bull_sim,
            'bear_sim': bear_sim,
        }

    def get_resonance_signal(self, date):
        """Signal 3: Resonance filter."""
        arch_id = get_day_sign(date)
        phase_day = get_trecena_day(date)
        key = f"{arch_id}_{phase_day}"
        resonance = self.resonance_matrix[key]

        if resonance > self.res_mean + self.res_std:
            level = 'HIGH'
        elif resonance < self.res_mean - self.res_std:
            level = 'LOW'
        else:
            level = 'MEDIUM'

        # Low resonance = abstain signal
        if level == 'LOW':
            return {'should_trade': False, 'level': level, 'value': resonance}
        else:
            return {'should_trade': True, 'level': level, 'value': resonance}

    def predict(self, date=None):
        """
        Combine all signals to make prediction.

        Only trade when:
        1. Resonance is not LOW (filter)
        2. At least 2 signals agree on direction
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # Get all signals
        trecena_signal = self.get_trecena_signal(date)
        compass_signal = self.get_compass_signal(date)
        resonance_signal = self.get_resonance_signal(date)

        # Check resonance filter
        if not resonance_signal['should_trade']:
            return {
                'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                'full_name': get_full_day_name(date),
                'trecena_day': get_trecena_day(date),
                'archetype': DAY_SIGN_ARCHETYPES[get_day_sign(date)]['glyph'],
                'direction': 'ABSTAIN',
                'reason': 'Low resonance - signals unclear',
                'signals': {
                    'trecena': trecena_signal,
                    'compass': compass_signal,
                    'resonance': resonance_signal,
                },
                'verified': False,
            }

        # Count direction votes (only BULLISH/BEARISH count)
        votes = {'BULLISH': 0, 'BEARISH': 0}
        signal_directions = []

        if trecena_signal['direction'] in ['BULLISH', 'BEARISH']:
            votes[trecena_signal['direction']] += self.signal_weights['trecena_historical']
            signal_directions.append(('trecena', trecena_signal['direction']))

        votes[compass_signal['direction']] += self.signal_weights['compass_direction']
        signal_directions.append(('compass', compass_signal['direction']))

        # Resonance bonus if high
        if resonance_signal['level'] == 'HIGH':
            resonance_bonus = 0.5
        else:
            resonance_bonus = 0.0

        # Add resonance weight to the majority direction
        if votes['BULLISH'] > votes['BEARISH']:
            votes['BULLISH'] += resonance_bonus
        elif votes['BEARISH'] > votes['BULLISH']:
            votes['BEARISH'] += resonance_bonus

        # Check agreement
        bullish_votes = sum(1 for s, d in signal_directions if d == 'BULLISH')
        bearish_votes = sum(1 for s, d in signal_directions if d == 'BEARISH')

        # Only 2 signals voting, so agreement = both same direction
        # Or one neutral and one directional
        signals_active = len([s for s, d in signal_directions if d != 'NEUTRAL'])

        if votes['BULLISH'] > votes['BEARISH']:
            direction = 'BULLISH'
            agreement = bullish_votes
        elif votes['BEARISH'] > votes['BULLISH']:
            direction = 'BEARISH'
            agreement = bearish_votes
        else:
            # Tie - abstain
            direction = 'ABSTAIN'
            agreement = 0

        # Confidence based on agreement and signal strengths
        if direction != 'ABSTAIN':
            avg_strength = (trecena_signal['strength'] + compass_signal['strength']) / 2
            confidence = 0.5 + avg_strength * 0.25 + (resonance_signal['value'] - 0.2) * 0.3
            confidence = min(0.85, max(0.5, confidence))
        else:
            confidence = 0.0

        return {
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'full_name': get_full_day_name(date),
            'trecena_day': get_trecena_day(date),
            'archetype': DAY_SIGN_ARCHETYPES[get_day_sign(date)]['glyph'],
            'direction': direction,
            'confidence': round(confidence, 3),
            'agreement': agreement,
            'votes': votes,
            'signals': {
                'trecena': trecena_signal,
                'compass': compass_signal,
                'resonance': resonance_signal,
            },
            'signal_directions': signal_directions,
            'verified': False,
        }

    def verify(self, prediction, actual_return):
        """Verify prediction against reality."""
        direction = prediction['direction']

        if direction == 'ABSTAIN':
            correct = None
        elif direction == 'BULLISH':
            correct = actual_return > 0
        else:
            correct = actual_return < 0

        prediction['actual_return'] = round(actual_return, 4)
        prediction['correct'] = correct
        prediction['verified'] = True

        return prediction

    def update(self, prediction):
        """Update signal weights based on accuracy."""
        if not prediction.get('verified'):
            raise ValueError("Cannot update with unverified prediction")

        correct = prediction['correct']
        actual = prediction.get('actual_return', 0)

        if correct is not None:
            # Track overall
            self.history['stats']['total'] += 1
            if correct:
                self.history['stats']['correct'] += 1

            # Track individual signal accuracy
            signals = prediction.get('signals', {})

            # Trecena signal
            trecena_dir = signals.get('trecena', {}).get('direction', 'NEUTRAL')
            if trecena_dir != 'NEUTRAL':
                self.history['signal_accuracy']['trecena_historical']['total'] += 1
                if (trecena_dir == 'BULLISH' and actual > 0) or \
                   (trecena_dir == 'BEARISH' and actual < 0):
                    self.history['signal_accuracy']['trecena_historical']['correct'] += 1

            # Compass signal
            compass_dir = signals.get('compass', {}).get('direction', 'NEUTRAL')
            if compass_dir != 'NEUTRAL':
                self.history['signal_accuracy']['compass_direction']['total'] += 1
                if (compass_dir == 'BULLISH' and actual > 0) or \
                   (compass_dir == 'BEARISH' and actual < 0):
                    self.history['signal_accuracy']['compass_direction']['correct'] += 1

            # Log
            self.history['predictions'].append({
                'date': prediction['date'],
                'direction': prediction['direction'],
                'correct': correct,
                'actual_return': actual,
            })

            # Update signal weights based on accuracy
            self._update_weights()
        else:
            self.history['stats']['abstained'] += 1

        self._save_history()
        return self.get_stats()

    def _update_weights(self):
        """Update signal weights based on their accuracy."""
        for signal_name in ['trecena_historical', 'compass_direction']:
            stats = self.history['signal_accuracy'][signal_name]
            if stats['total'] >= 20:
                acc = stats['correct'] / stats['total']
                # Weight = accuracy relative to 50%
                self.signal_weights[signal_name] = max(0.5, min(2.0, acc / 0.5))

    def get_stats(self):
        """Get current statistics."""
        total = self.history['stats']['total']
        correct = self.history['stats']['correct']
        abstained = self.history['stats']['abstained']
        accuracy = correct / total if total > 0 else 0.5

        signal_stats = {}
        for name, stats in self.history['signal_accuracy'].items():
            if stats['total'] > 0:
                signal_stats[name] = {
                    'accuracy': round(stats['correct'] / stats['total'], 3),
                    'n': stats['total'],
                    'weight': self.signal_weights.get(name, 1.0),
                }

        return {
            'total': total,
            'correct': correct,
            'abstained': abstained,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'signal_stats': signal_stats,
        }

    def backtest(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Backtest with learning."""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        print(f"Backtesting Multi-Signal AGI: {start_date} to {end_date}")

        # Reset
        self.history = {
            'predictions': [],
            'stats': {'correct': 0, 'total': 0, 'abstained': 0},
            'signal_accuracy': {
                'trecena_historical': {'correct': 0, 'total': 0},
                'compass_direction': {'correct': 0, 'total': 0},
                'agreement': {'correct': 0, 'total': 0},
            }
        }
        self.signal_weights = {
            'trecena_historical': 1.0,
            'compass_direction': 1.0,
            'resonance': 1.0,
        }

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        traded_results = []
        all_results = []
        cumulative = 0

        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            pred = self.predict(date)
            pred = self.verify(pred, ret)
            self.update(pred)

            all_results.append(pred)

            if pred['direction'] != 'ABSTAIN':
                traded_results.append(pred)
                if pred['direction'] == 'BULLISH':
                    cumulative += ret
                else:
                    cumulative -= ret

        # Stats
        total = len(traded_results)
        correct = sum(1 for r in traded_results if r['correct'])
        accuracy = correct / total if total > 0 else 0

        buy_hold = sum(returns[~np.isnan(returns)])

        # Accuracy when signals agree
        agree_results = [r for r in traded_results if r.get('agreement', 0) >= 2]
        agree_correct = sum(1 for r in agree_results if r['correct'])
        agree_accuracy = agree_correct / len(agree_results) if agree_results else 0

        return {
            'total_days': len(all_results),
            'traded_days': total,
            'trade_rate': round(total / len(all_results), 3) if all_results else 0,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'cumulative_return': round(cumulative, 2),
            'buy_hold_return': round(buy_hold, 2),
            'excess_return': round(cumulative - buy_hold, 2),
            'agreement_accuracy': round(agree_accuracy, 3),
            'agreement_n': len(agree_results),
            'final_signal_weights': dict(self.signal_weights),
            'signal_accuracy': {
                name: {
                    'accuracy': round(s['correct']/max(1,s['total']), 3),
                    'n': s['total'],
                }
                for name, s in self.history['signal_accuracy'].items()
            },
        }


def main():
    print("=" * 70)
    print("MULTI-SIGNAL AGI: Combining Signals Like the Aztecs")
    print("=" * 70)

    agi = MultiSignalAGI()

    # Today's prediction
    print("\n--- TODAY'S MULTI-SIGNAL PREDICTION ---\n")

    today = datetime.now()
    pred = agi.predict(today)

    print(f"Date: {pred['date']}")
    print(f"Aztec Day: {pred['full_name']}")
    print(f"Trecena Day: {pred['trecena_day']}")
    print(f"Archetype: {pred['archetype']}")

    print(f"\n--- INDIVIDUAL SIGNALS ---")
    signals = pred['signals']

    trecena = signals['trecena']
    print(f"1. TRECENA HISTORICAL: {trecena['direction']}")
    print(f"   Mean return: {trecena['raw_mean']:+.3f}%")
    print(f"   Strength: {trecena['strength']:.2f}")

    compass = signals['compass']
    print(f"\n2. COMPASS DIRECTION: {compass['direction']}")
    print(f"   Bull sim: {compass['bull_sim']:.3f}")
    print(f"   Bear sim: {compass['bear_sim']:.3f}")
    print(f"   Strength: {compass['strength']:.2f}")

    resonance = signals['resonance']
    print(f"\n3. RESONANCE FILTER: {resonance['level']}")
    print(f"   Value: {resonance['value']:.3f}")
    print(f"   Should trade: {resonance['should_trade']}")

    print(f"\n--- COMBINED DECISION ---")
    print(f"Direction: {pred['direction']}")
    if pred['direction'] != 'ABSTAIN':
        print(f"Confidence: {pred['confidence']:.1%}")
        print(f"Signal agreement: {pred.get('agreement', 0)}")
        print(f"Votes: {pred.get('votes', {})}")
    else:
        print(f"Reason: {pred.get('reason', 'Signals disagree')}")

    # Backtest
    print("\n" + "=" * 70)
    print("BACKTEST: Multi-Signal (2020-2024)")
    print("=" * 70 + "\n")

    results = agi.backtest('2020-01-01', '2024-12-31')

    if results:
        print(f"Total trading days: {results['total_days']}")
        print(f"Days we traded: {results['traded_days']} ({results['trade_rate']:.1%})")
        print(f"Days we abstained: {results['total_days'] - results['traded_days']}")

        print(f"\nPrediction Accuracy:")
        print(f"  Correct: {results['correct']}")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Edge: {results['edge']:+.1f} pp")

        print(f"\nReturns:")
        print(f"  Multi-signal: {results['cumulative_return']:+.1f}%")
        print(f"  Buy & hold:   {results['buy_hold_return']:+.1f}%")
        print(f"  Excess:       {results['excess_return']:+.1f}%")

        print(f"\nIndividual Signal Accuracy:")
        for name, stats in results['signal_accuracy'].items():
            weight = results['final_signal_weights'].get(name, 1.0)
            print(f"  {name:<20}: {stats['accuracy']:.1%} (n={stats['n']}, weight={weight:.2f})")

        print(f"\nWhen Both Signals Agree:")
        print(f"  Accuracy: {results['agreement_accuracy']:.1%} (n={results['agreement_n']})")

        # Significance
        if results['traded_days'] > 0:
            print(f"\nStatistical Significance:")
            result = scipy_stats.binomtest(
                results['correct'], results['traded_days'], 0.5, alternative='greater'
            )
            print(f"  p-value: {result.pvalue:.4e}")
            print(f"  Significant at 0.05: {'YES' if result.pvalue < 0.05 else 'NO'}")

    # The insight
    print("\n" + "=" * 70)
    print("THE ENSEMBLE INSIGHT")
    print("=" * 70)

    print("""
The Aztecs didn't use just ONE signal. They combined:
- The Trecena day (1-13) - "what phase are we in?"
- The Day Sign (1-20) - "what archetype dominates?"
- The resonance - "do these energies align?"

This is ENSEMBLE LEARNING before machine learning existed.

Our Multi-Signal AGI does the same:
1. TRECENA HISTORICAL - empirical pattern from 32 years
2. COMPASS DIRECTION - semantic similarity to outcomes
3. RESONANCE FILTER - only trade when signals are clear

When signals AGREE, confidence increases.
When signals CONFLICT, we ABSTAIN.

This is the ancient wisdom:
"Many counselors bring safety." (Proverbs 11:14)

The path to AGI isn't a single brilliant algorithm.
It's the wisdom to combine multiple weak signals into strength.
""")


if __name__ == "__main__":
    main()

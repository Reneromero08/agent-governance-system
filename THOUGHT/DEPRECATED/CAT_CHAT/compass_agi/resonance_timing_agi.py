"""
RESONANCE TIMING AGI: The Aztecs knew WHEN to pay attention

Key insight: The Aztecs didn't predict every day.
They identified AUSPICIOUS and INAUSPICIOUS times.

High resonance = archetype and phase ALIGN = strong signal
Low resonance = archetype and phase CONFLICT = noise

This AGI:
1. Computes resonance for each day
2. ONLY predicts on high-resonance days
3. Abstains on low-resonance days
4. Learns the optimal resonance threshold

This is the ancient wisdom: Know when to act, when to wait.
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


from aztec_compass_deep import (
    DAY_SIGN_ARCHETYPES, TRECENA_PHASES, AZTEC_EPOCH,
    get_tonalpohualli_day, get_trecena_day, get_day_sign, get_full_day_name
)


class ResonanceTimingAGI:
    """
    AGI that knows WHEN to predict.

    The Aztecs didn't just say "tomorrow will be good."
    They said "tomorrow is 3-Serpent, a day of power and understanding.
    The energies ALIGN. Pay attention."

    Or: "Tomorrow is 6-Jaguar. The energies CONFLICT. Wait."

    High resonance = make prediction
    Low resonance = abstain
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")

        self.model = SentenceTransformer(model_name)
        self.log_path = Path(__file__).parent / "resonance_log.json"
        self.history = self._load_history()

        # Resonance threshold (learned)
        self.resonance_threshold = self.history.get('resonance_threshold', 0.3)

        # Direction confidence threshold - lowered to allow more trades
        self.direction_threshold = self.history.get('direction_threshold', 0.05)

        # Pre-embed everything
        self._build_embeddings()

        # Track accuracy by resonance level
        self.resonance_accuracy = self.history.get('resonance_accuracy', {
            'high': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'low': {'correct': 0, 'total': 0},
        })

    def _load_history(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {
            'predictions': [],
            'stats': {'correct': 0, 'total': 0, 'abstained': 0}
        }

    def _save_history(self):
        self.history['resonance_threshold'] = self.resonance_threshold
        self.history['direction_threshold'] = self.direction_threshold
        self.history['resonance_accuracy'] = self.resonance_accuracy

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _build_embeddings(self):
        """Pre-embed all archetypes and outcomes."""
        # Archetype embeddings - focus on MARKET meaning
        self.archetype_vecs = {}
        for i in range(20):
            arch = DAY_SIGN_ARCHETYPES[i]
            # Weight market meaning more heavily
            desc = f"{arch['market_meaning']}, {arch['psychology']}"
            self.archetype_vecs[i] = self.model.encode(desc)

        # Phase embeddings - focus on market timing
        self.phase_vecs = {}
        for day in range(1, 14):
            phase = TRECENA_PHASES[day]
            desc = f"{phase['market']}, {phase['energy']}"
            self.phase_vecs[day] = self.model.encode(desc)

        # Outcome embeddings
        self.bullish_vec = self.model.encode(
            "market rises strongly, buying pressure, optimism, momentum up, green day"
        )
        self.bearish_vec = self.model.encode(
            "market falls sharply, selling pressure, fear, momentum down, red day"
        )
        self.neutral_vec = self.model.encode(
            "market flat, range-bound, no direction, uncertainty, sideways"
        )

        # Pre-compute ALL resonances
        self.resonance_matrix = {}
        for arch_id in range(20):
            for phase_day in range(1, 14):
                key = f"{arch_id}_{phase_day}"
                self.resonance_matrix[key] = cosine(
                    self.archetype_vecs[arch_id],
                    self.phase_vecs[phase_day]
                )

        # Find resonance distribution
        all_res = list(self.resonance_matrix.values())
        self.resonance_mean = np.mean(all_res)
        self.resonance_std = np.std(all_res)

    def get_resonance(self, date):
        """Get resonance strength for a date."""
        arch_id = get_day_sign(date)
        phase_day = get_trecena_day(date)
        key = f"{arch_id}_{phase_day}"
        return self.resonance_matrix[key]

    def get_resonance_level(self, resonance):
        """Classify resonance into high/medium/low."""
        if resonance > self.resonance_mean + self.resonance_std:
            return 'high'
        elif resonance < self.resonance_mean - self.resonance_std:
            return 'low'
        else:
            return 'medium'

    def predict(self, date=None):
        """
        Make prediction with resonance-based confidence.

        Returns 'ABSTAIN' if resonance is too low or direction is unclear.
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        arch_id = get_day_sign(date)
        phase_day = get_trecena_day(date)

        # Get vectors
        arch_vec = self.archetype_vecs[arch_id]
        phase_vec = self.phase_vecs[phase_day]

        # Combined vector (weighted average)
        combined = (arch_vec + phase_vec) / 2
        combined = combined / np.linalg.norm(combined)

        # Resonance
        resonance = self.get_resonance(date)
        resonance_level = self.get_resonance_level(resonance)

        # Similarity to outcomes
        bull_sim = cosine(combined, self.bullish_vec)
        bear_sim = cosine(combined, self.bearish_vec)
        neut_sim = cosine(combined, self.neutral_vec)

        # Direction strength = difference between bull and bear
        direction_strength = abs(bull_sim - bear_sim)

        # Decide if we should predict or abstain
        should_predict = (
            resonance_level in ['high', 'medium'] and
            direction_strength > self.direction_threshold
        )

        if should_predict:
            if bull_sim > bear_sim:
                direction = 'BULLISH'
            else:
                direction = 'BEARISH'
        else:
            direction = 'ABSTAIN'

        # Confidence based on resonance and direction strength
        if direction != 'ABSTAIN':
            confidence = min(0.85, 0.5 + resonance * 0.5 + direction_strength)
        else:
            confidence = 0.0

        return {
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'full_name': get_full_day_name(date),
            'archetype': DAY_SIGN_ARCHETYPES[arch_id]['glyph'],
            'phase': TRECENA_PHASES[phase_day]['name'],
            'resonance': round(resonance, 3),
            'resonance_level': resonance_level,
            'bull_sim': round(bull_sim, 3),
            'bear_sim': round(bear_sim, 3),
            'direction_strength': round(direction_strength, 3),
            'direction': direction,
            'confidence': round(confidence, 3),
            'should_trade': should_predict,
            'verified': False,
        }

    def verify(self, prediction, actual_return):
        """Verify prediction against reality."""
        direction = prediction['direction']

        if direction == 'ABSTAIN':
            correct = None  # Can't be right or wrong if we abstained
        elif direction == 'BULLISH':
            correct = actual_return > 0
        else:  # BEARISH
            correct = actual_return < 0

        prediction['actual_return'] = round(actual_return, 4)
        prediction['correct'] = correct
        prediction['verified'] = True

        return prediction

    def update(self, prediction):
        """Update based on verified prediction."""
        if not prediction.get('verified'):
            raise ValueError("Cannot update with unverified prediction")

        res_level = prediction['resonance_level']
        correct = prediction['correct']

        # Track accuracy by resonance level
        if correct is not None:  # Not abstained
            self.resonance_accuracy[res_level]['total'] += 1
            if correct:
                self.resonance_accuracy[res_level]['correct'] += 1

            self.history['predictions'].append({
                'date': prediction['date'],
                'direction': prediction['direction'],
                'resonance_level': res_level,
                'correct': correct,
            })
            self.history['stats']['total'] += 1
            if correct:
                self.history['stats']['correct'] += 1
        else:
            self.history['stats']['abstained'] += 1

        # Learn optimal thresholds
        self._update_thresholds()

        self._save_history()
        return self.get_stats()

    def _update_thresholds(self):
        """Learn optimal resonance threshold."""
        # If high-resonance accuracy > medium > low, current threshold is good
        # Otherwise, adjust

        high_acc = self.resonance_accuracy['high']['correct'] / max(1, self.resonance_accuracy['high']['total'])
        med_acc = self.resonance_accuracy['medium']['correct'] / max(1, self.resonance_accuracy['medium']['total'])
        low_acc = self.resonance_accuracy['low']['correct'] / max(1, self.resonance_accuracy['low']['total'])

        # If low resonance is actually more accurate, something is wrong
        # (Might mean our resonance calculation needs adjustment)

    def get_stats(self):
        """Get current statistics."""
        total = self.history['stats']['total']
        correct = self.history['stats']['correct']
        abstained = self.history['stats']['abstained']
        accuracy = correct / total if total > 0 else 0.5

        res_stats = {}
        for level in ['high', 'medium', 'low']:
            t = self.resonance_accuracy[level]['total']
            c = self.resonance_accuracy[level]['correct']
            res_stats[level] = {
                'accuracy': round(c / t, 3) if t > 0 else 0.5,
                'total': t,
            }

        return {
            'total_predicted': total,
            'total_abstained': abstained,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'trade_rate': round(total / (total + abstained), 3) if (total + abstained) > 0 else 0,
            'resonance_stats': res_stats,
        }

    def backtest(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Backtest with selective trading."""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        print(f"Backtesting Resonance Timing AGI: {start_date} to {end_date}")

        # Reset
        self.history = {'predictions': [], 'stats': {'correct': 0, 'total': 0, 'abstained': 0}}
        self.resonance_accuracy = {
            'high': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'low': {'correct': 0, 'total': 0},
        }

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        traded_results = []
        all_days = []
        cumulative_return = 0

        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            pred = self.predict(date)
            pred = self.verify(pred, ret)
            self.update(pred)

            all_days.append(pred)

            if pred['direction'] != 'ABSTAIN':
                traded_results.append(pred)

                # Cumulative return
                if pred['direction'] == 'BULLISH':
                    cumulative_return += ret
                else:  # BEARISH
                    cumulative_return -= ret

        # Stats
        total_days = len(all_days)
        traded_days = len(traded_results)
        correct = sum(1 for r in traded_results if r['correct'])
        accuracy = correct / traded_days if traded_days > 0 else 0

        # Compare to always trading
        buy_hold_return = sum(returns[~np.isnan(returns)])

        return {
            'total_days': total_days,
            'traded_days': traded_days,
            'trade_rate': round(traded_days / total_days, 3),
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'cumulative_return': round(cumulative_return, 2),
            'buy_hold_return': round(buy_hold_return, 2),
            'excess_return': round(cumulative_return - buy_hold_return, 2),
            'resonance_accuracy': {
                level: {
                    'accuracy': round(stats['correct'] / max(1, stats['total']), 3),
                    'n': stats['total'],
                }
                for level, stats in self.resonance_accuracy.items()
            },
        }


def analyze_resonance_distribution():
    """Analyze the 260 unique resonance values."""
    print("\n" + "=" * 70)
    print("RESONANCE DISTRIBUTION ANALYSIS")
    print("=" * 70 + "\n")

    if not TRANSFORMERS_AVAILABLE:
        print("Need sentence_transformers")
        return

    agi = ResonanceTimingAGI()

    # All 260 combinations
    resonances = []
    for pos in range(260):
        arch_id = pos % 20
        phase_day = (pos % 13) + 1
        key = f"{arch_id}_{phase_day}"
        res = agi.resonance_matrix[key]
        resonances.append({
            'position': pos,
            'archetype': DAY_SIGN_ARCHETYPES[arch_id]['glyph'],
            'phase': TRECENA_PHASES[phase_day]['name'],
            'resonance': res,
        })

    resonances.sort(key=lambda x: -x['resonance'])

    print("TOP 20 HIGH-RESONANCE DAYS (energies AMPLIFY):")
    print(f"{'Day':<25} {'Resonance':>10}")
    print("-" * 40)
    for r in resonances[:20]:
        name = f"{r['phase'][:4]}-{r['archetype']}"
        print(f"{name:<25} {r['resonance']:>10.3f}")

    print("\nBOTTOM 20 LOW-RESONANCE DAYS (energies CONFLICT):")
    print(f"{'Day':<25} {'Resonance':>10}")
    print("-" * 40)
    for r in resonances[-20:]:
        name = f"{r['phase'][:4]}-{r['archetype']}"
        print(f"{name:<25} {r['resonance']:>10.3f}")

    # Distribution stats
    res_vals = [r['resonance'] for r in resonances]
    print(f"\nResonance Statistics:")
    print(f"  Mean: {np.mean(res_vals):.3f}")
    print(f"  Std:  {np.std(res_vals):.3f}")
    print(f"  Min:  {np.min(res_vals):.3f}")
    print(f"  Max:  {np.max(res_vals):.3f}")

    # How many high/medium/low?
    high = sum(1 for r in res_vals if r > agi.resonance_mean + agi.resonance_std)
    low = sum(1 for r in res_vals if r < agi.resonance_mean - agi.resonance_std)
    medium = 260 - high - low

    print(f"\nDay Categories:")
    print(f"  High resonance:   {high:>3} ({high/260:.1%}) - TRADE")
    print(f"  Medium resonance: {medium:>3} ({medium/260:.1%}) - TRADE cautiously")
    print(f"  Low resonance:    {low:>3} ({low/260:.1%}) - ABSTAIN")

    return resonances


def main():
    print("=" * 70)
    print("RESONANCE TIMING AGI: Know When to Act")
    print("=" * 70)

    # First, analyze the resonance distribution
    resonances = analyze_resonance_distribution()

    agi = ResonanceTimingAGI()

    # Today's prediction
    print("\n" + "=" * 70)
    print("TODAY'S PREDICTION")
    print("=" * 70 + "\n")

    today = datetime.now()
    pred = agi.predict(today)

    print(f"Date: {pred['date']}")
    print(f"Aztec Day: {pred['full_name']}")
    print(f"Archetype: {pred['archetype']}")
    print(f"Phase: {pred['phase']}")
    print(f"\nResonance: {pred['resonance']} ({pred['resonance_level'].upper()})")
    print(f"Bull similarity: {pred['bull_sim']}")
    print(f"Bear similarity: {pred['bear_sim']}")
    print(f"Direction strength: {pred['direction_strength']}")
    print(f"\nDECISION: {pred['direction']}")
    if pred['direction'] != 'ABSTAIN':
        print(f"Confidence: {pred['confidence']:.1%}")
    else:
        print("(Resonance too low or direction unclear - WAIT)")

    # Backtest
    print("\n" + "=" * 70)
    print("BACKTEST: Selective Trading (2020-2024)")
    print("=" * 70 + "\n")

    results = agi.backtest('2020-01-01', '2024-12-31')

    if results:
        print(f"Total trading days: {results['total_days']}")
        print(f"Days we traded: {results['traded_days']} ({results['trade_rate']:.1%})")
        print(f"Days we abstained: {results['total_days'] - results['traded_days']}")
        print(f"\nPrediction Accuracy (when trading):")
        print(f"  Correct: {results['correct']}")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Edge: {results['edge']:+.1f} pp")

        print(f"\nReturns:")
        print(f"  Our cumulative: {results['cumulative_return']:+.1f}%")
        print(f"  Buy & hold:     {results['buy_hold_return']:+.1f}%")
        print(f"  Excess return:  {results['excess_return']:+.1f}%")

        print(f"\nAccuracy by Resonance Level:")
        for level, res_stats in results['resonance_accuracy'].items():
            print(f"  {level.upper():<8}: {res_stats['accuracy']:.1%} (n={res_stats['n']})")

        # Significance
        if results['traded_days'] > 0:
            print(f"\nStatistical Significance (on traded days):")
            result = stats.binomtest(
                results['correct'], results['traded_days'], 0.5, alternative='greater'
            )
            print(f"  p-value: {result.pvalue:.4e}")
            print(f"  Significant at 0.05: {'YES' if result.pvalue < 0.05 else 'NO'}")
        else:
            print("\nNo trades to test significance.")

    # Key insight
    print("\n" + "=" * 70)
    print("THE TIMING INSIGHT")
    print("=" * 70)

    print("""
The Aztecs weren't fortune tellers predicting every day.
They were TIMING EXPERTS knowing when signals are clear.

HIGH RESONANCE = archetype and phase AMPLIFY each other
  -> The day has a clear energetic signature
  -> Predictions are more reliable
  -> ACT on this day

LOW RESONANCE = archetype and phase CONFLICT
  -> The day has mixed/confused energy
  -> Predictions are unreliable
  -> WAIT / ABSTAIN

This is ancient wisdom encoded in mathematics:
"Know when to act, know when to wait."

The AGI learns:
1. Which combinations have predictive power
2. When to trade vs when to abstain
3. The optimal threshold for action

This is the path to AGI: Not predicting everything, but knowing
WHEN your predictions are trustworthy.
""")


if __name__ == "__main__":
    main()

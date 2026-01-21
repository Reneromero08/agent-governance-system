"""
COMPASS PREDICTOR: Using the semantic compass for actual prediction

The compass navigates embedding space. To make it predict:
1. Embed the CURRENT STATE (market conditions, time, sentiment)
2. Use compass to find where it POINTS (most similar outcomes)
3. Verify against reality
4. UPDATE which contexts/axes the compass should use

This closes the loop: COMPASS -> PREDICTION -> VERIFICATION -> COMPASS
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


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class CompassPredictor:
    """
    Uses the semantic compass to predict market outcomes.

    The key insight: The compass found that markets and astrology share
    FORECAST/PREDICTION as a bridge concept. So we use the compass to
    forecast by navigating from current state to outcome space.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")
        self.model = SentenceTransformer(model_name)
        self.log_path = Path(__file__).parent / "compass_log.json"
        self.history = self._load_history()

        # Outcome embeddings (what we're predicting)
        self.outcomes = {
            'strong_up': self.model.encode("market rallies strongly, prices surge, bulls win"),
            'up': self.model.encode("market rises, positive returns, green day"),
            'flat': self.model.encode("market unchanged, sideways trading, no direction"),
            'down': self.model.encode("market falls, negative returns, red day"),
            'strong_down': self.model.encode("market crashes, prices plunge, bears win"),
        }

        # Context axes (learned from our experiments)
        self.axes = {
            'cycles': "in terms of cycles and timing",
            'psychology': "in terms of collective psychology",
            'momentum': "in terms of momentum and trend",
            'uncertainty': "in terms of risk and uncertainty",
            'forecast': "in terms of prediction and forecast",  # The bridge!
        }

        # Weights for each axis (updated based on accuracy)
        self.axis_weights = {axis: 1.0 for axis in self.axes}

    def _load_history(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {'predictions': [], 'axis_accuracy': {}}

    def _save_history(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def embed_state(self, description, axis=None):
        """Embed current market state with optional axis context."""
        if axis and axis in self.axes:
            text = f"{description}, {self.axes[axis]}"
        else:
            text = description
        return self.model.encode(text)

    def compass_navigate(self, state_vec):
        """
        Navigate from state to most likely outcome.

        Returns dict with outcome probabilities.
        """
        sims = {}
        for outcome, outcome_vec in self.outcomes.items():
            sims[outcome] = cosine(state_vec, outcome_vec)

        # Convert to probabilities (softmax-ish)
        exp_sims = {k: np.exp(v * 5) for k, v in sims.items()}  # Temperature scaling
        total = sum(exp_sims.values())
        probs = {k: v / total for k, v in exp_sims.items()}

        return sims, probs

    def predict_from_description(self, description):
        """
        Main prediction method: describe the state, get a prediction.

        Uses all axes and weights them by learned accuracy.
        """
        predictions = {}

        for axis_name, axis_context in self.axes.items():
            state_vec = self.embed_state(description, axis_name)
            sims, probs = self.compass_navigate(state_vec)

            # Weighted by axis accuracy
            weight = self.axis_weights.get(axis_name, 1.0)
            predictions[axis_name] = {
                'sims': sims,
                'probs': probs,
                'weight': weight,
            }

        # Combine predictions across axes
        combined_probs = {}
        total_weight = sum(self.axis_weights.values())

        for outcome in self.outcomes.keys():
            weighted_sum = sum(
                predictions[axis]['probs'][outcome] * predictions[axis]['weight']
                for axis in self.axes.keys()
            )
            combined_probs[outcome] = weighted_sum / total_weight

        # Get final prediction
        best_outcome = max(combined_probs, key=combined_probs.get)

        # Convert to direction
        if best_outcome in ['strong_up', 'up']:
            direction = 'BULLISH'
        elif best_outcome in ['strong_down', 'down']:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        confidence = combined_probs[best_outcome]

        return {
            'description': description,
            'direction': direction,
            'best_outcome': best_outcome,
            'confidence': round(confidence, 3),
            'combined_probs': combined_probs,
            'axis_predictions': predictions,
        }

    def predict_from_trecena(self, date=None):
        """
        Use Trecena day as input to compass.

        This is where Aztec calendar meets embedding space!
        """
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # Get Trecena day
        AZTEC_EPOCH = datetime(1990, 1, 1)
        days_since = (date - AZTEC_EPOCH).days
        trecena_day = (days_since % 13) + 1

        # Embed the Trecena state
        trecena_descriptions = {
            1: "Beginning of cycle, new energy, initiation",
            2: "Building momentum, growth phase, expansion",
            3: "Early development, foundation setting",
            4: "Stability emerging, structure forming, peak building",
            5: "Midpoint approaching, transition energy",
            6: "Uncertainty rising, questioning, doubt",
            7: "Balance point, equilibrium, pause",
            8: "Renewed energy, second wind, recovery",
            9: "Completion approaching, winding down",
            10: "Manifestation, results showing, harvest",
            11: "Reflection period, assessment",
            12: "Ending phase, release, letting go",
            13: "Completion, transformation, ending before new beginning",
        }

        description = trecena_descriptions[trecena_day]
        prediction = self.predict_from_description(description)
        prediction['trecena_day'] = trecena_day
        prediction['date'] = date.strftime('%Y-%m-%d')

        return prediction

    def verify(self, prediction, actual_return):
        """Verify prediction against actual return."""
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
        Update axis weights based on verification.

        This is the LEARNING step - the compass improves its use of contexts.
        """
        if not prediction.get('verified'):
            raise ValueError("Cannot update with unverified prediction")

        # Log the prediction
        self.history['predictions'].append({
            'date': prediction.get('date', str(datetime.now())),
            'direction': prediction['direction'],
            'correct': prediction['correct'],
            'actual_return': prediction['actual_return'],
        })

        # Update axis accuracy tracking
        for axis_name, axis_pred in prediction.get('axis_predictions', {}).items():
            if axis_name not in self.history['axis_accuracy']:
                self.history['axis_accuracy'][axis_name] = {'correct': 0, 'total': 0}

            self.history['axis_accuracy'][axis_name]['total'] += 1

            # Did this axis predict correctly?
            axis_best = max(axis_pred['probs'], key=axis_pred['probs'].get)
            axis_direction = 'BULLISH' if axis_best in ['strong_up', 'up'] else \
                            'BEARISH' if axis_best in ['strong_down', 'down'] else 'NEUTRAL'

            if (axis_direction == 'BULLISH' and prediction['actual_return'] > 0) or \
               (axis_direction == 'BEARISH' and prediction['actual_return'] < 0) or \
               (axis_direction == 'NEUTRAL' and abs(prediction['actual_return']) < 0.5):
                self.history['axis_accuracy'][axis_name]['correct'] += 1

        # Update axis weights based on accuracy
        for axis_name in self.axes.keys():
            if axis_name in self.history['axis_accuracy']:
                stats = self.history['axis_accuracy'][axis_name]
                if stats['total'] > 5:  # Need minimum data
                    accuracy = stats['correct'] / stats['total']
                    # Weight = accuracy relative to random (0.33 for 3-way)
                    self.axis_weights[axis_name] = max(0.1, accuracy / 0.33)

        self._save_history()
        return self.get_stats()

    def get_stats(self):
        """Get current compass statistics."""
        preds = self.history['predictions']
        if not preds:
            return {'total': 0, 'accuracy': 0.5}

        total = len(preds)
        correct = sum(1 for p in preds if p['correct'])
        accuracy = correct / total

        # Axis stats
        axis_stats = {}
        for axis, stats in self.history.get('axis_accuracy', {}).items():
            if stats['total'] > 0:
                axis_stats[axis] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'weight': self.axis_weights.get(axis, 1.0),
                }

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'axis_stats': axis_stats,
        }

    def backtest(self, start_date='2023-01-01', end_date='2024-12-31'):
        """Backtest the compass predictor."""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        print(f"Backtesting Compass Predictor: {start_date} to {end_date}")

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        results = []
        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            # PREDICT using compass + trecena
            pred = self.predict_from_trecena(date)

            # VERIFY
            pred = self.verify(pred, ret)

            # UPDATE weights
            self.update(pred)

            results.append(pred)

        # Final stats
        correct = sum(1 for r in results if r['correct'])
        total = len(results)

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(correct/total, 3) if total > 0 else 0,
            'final_axis_weights': dict(self.axis_weights),
            'final_axis_stats': self.get_stats()['axis_stats'],
        }


def main():
    print("=" * 60)
    print("COMPASS PREDICTOR: Embedding Space -> Prediction")
    print("=" * 60)

    predictor = CompassPredictor()

    # Show how the compass predicts from Trecena
    print("\n--- COMPASS + TRECENA ---\n")

    today = datetime.now()
    pred = predictor.predict_from_trecena(today)

    print(f"Date: {pred['date']}")
    print(f"Trecena Day: {pred['trecena_day']}")
    print(f"Description: {pred['description']}")
    print(f"\nCompass navigates to: {pred['best_outcome']}")
    print(f"Direction: {pred['direction']}")
    print(f"Confidence: {pred['confidence']:.1%}")

    print("\nOutcome probabilities:")
    for outcome, prob in sorted(pred['combined_probs'].items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {prob:.1%}")

    print("\nAxis contributions:")
    for axis, data in pred['axis_predictions'].items():
        best = max(data['probs'], key=data['probs'].get)
        print(f"  {axis}: -> {best} ({data['probs'][best]:.1%})")

    # Backtest
    print("\n" + "=" * 60)
    print("BACKTEST (2023-2024)")
    print("=" * 60 + "\n")

    results = predictor.backtest('2023-01-01', '2024-12-31')

    if results:
        print(f"Total predictions: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.1%}")

        print("\nLearned axis weights:")
        for axis, weight in sorted(results['final_axis_weights'].items(), key=lambda x: -x[1]):
            stats = results['final_axis_stats'].get(axis, {})
            acc = stats.get('accuracy', 0)
            print(f"  {axis}: weight={weight:.2f}, accuracy={acc:.1%}")


if __name__ == "__main__":
    main()

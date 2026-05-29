"""
TRECENA ORACLE: AGI Prediction Loop

The compass showed us the path: Pattern -> Prediction -> Verification -> Update

This oracle:
1. PREDICTS market direction based on Trecena day
2. GROUNDS predictions in real market data
3. VERIFIES predictions against actual outcomes
4. UPDATES confidence based on track record

This is the closed loop that makes the compass into AGI.
"""

import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================================
# TRECENA CALENDAR
# ============================================================================

AZTEC_EPOCH = datetime(1990, 1, 1)

def get_trecena_day(date):
    """Get day 1-13 in the Trecena cycle."""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    return (days_since % 13) + 1


def get_day_sign(date):
    """Get day sign 0-19."""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    days_since = (date - AZTEC_EPOCH).days
    return days_since % 20


DAY_SIGNS = [
    'Cipactli (Crocodile)', 'Ehecatl (Wind)', 'Calli (House)', 'Cuetzpalin (Lizard)',
    'Coatl (Serpent)', 'Miquiztli (Death)', 'Mazatl (Deer)', 'Tochtli (Rabbit)',
    'Atl (Water)', 'Itzcuintli (Dog)', 'Ozomatli (Monkey)', 'Malinalli (Grass)',
    'Acatl (Reed)', 'Ocelotl (Jaguar)', 'Cuauhtli (Eagle)', 'Cozcacuauhtli (Vulture)',
    'Ollin (Earthquake)', 'Tecpatl (Flint)', 'Quiahuitl (Rain)', 'Xochitl (Flower)'
]


# ============================================================================
# HISTORICAL PATTERN (from our analysis)
# ============================================================================

# Mean daily returns by Trecena day (from 32 years of S&P 500 data)
TRECENA_PATTERN = {
    1: +0.035,   # %
    2: +0.178,   # SECOND BEST
    3: +0.016,
    4: +0.122,   # BEST (in training, held out-of-sample)
    5: -0.001,
    6: -0.017,
    7: +0.015,
    8: +0.123,   # Also good
    9: -0.013,
    10: +0.093,
    11: +0.010,
    12: +0.045,
    13: -0.003,
}

# Classify days
BULLISH_DAYS = [2, 4, 8, 10]  # Above average
BEARISH_DAYS = [5, 6, 9, 13]  # Below average or negative
NEUTRAL_DAYS = [1, 3, 7, 11, 12]


# ============================================================================
# ORACLE CLASS
# ============================================================================

class TrecenaOracle:
    """
    AGI Prediction Loop:
    Pattern Recognition -> Prediction -> Verification -> Update
    """

    def __init__(self, log_path=None):
        self.log_path = log_path or Path(__file__).parent / "oracle_log.json"
        self.predictions = self._load_log()
        self.base_pattern = TRECENA_PATTERN.copy()

    def _load_log(self):
        """Load prediction history."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {'predictions': [], 'stats': {'correct': 0, 'total': 0}}

    def _save_log(self):
        """Save prediction history."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)

    def get_today_info(self, date=None):
        """Get Trecena info for a date."""
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        trecena_day = get_trecena_day(date)
        day_sign = get_day_sign(date)

        return {
            'date': date.strftime('%Y-%m-%d'),
            'trecena_day': trecena_day,
            'day_sign': day_sign,
            'day_sign_name': DAY_SIGNS[day_sign],
            'full_name': f"{trecena_day}-{DAY_SIGNS[day_sign].split()[0]}",
            'historical_return': self.base_pattern[trecena_day],
        }

    def predict(self, date=None):
        """
        PREDICT: Make a forecast for a given date.

        Returns prediction with confidence based on:
        1. Historical pattern strength
        2. Track record accuracy
        """
        info = self.get_today_info(date)
        trecena_day = info['trecena_day']
        hist_return = info['historical_return']

        # Base prediction from historical pattern
        if trecena_day in BULLISH_DAYS:
            direction = "BULLISH"
            confidence = 0.55 + abs(hist_return) / 0.5  # Scale by effect size
        elif trecena_day in BEARISH_DAYS:
            direction = "BEARISH"
            confidence = 0.55 + abs(hist_return) / 0.5
        else:
            direction = "NEUTRAL"
            confidence = 0.50

        # Adjust confidence based on track record
        if self.predictions['stats']['total'] > 10:
            accuracy = self.predictions['stats']['correct'] / self.predictions['stats']['total']
            # Bayesian update: blend historical confidence with observed accuracy
            confidence = 0.5 * confidence + 0.5 * accuracy

        confidence = min(0.75, max(0.45, confidence))  # Clamp to reasonable range

        prediction = {
            'date': info['date'],
            'trecena_day': trecena_day,
            'day_sign_name': info['day_sign_name'],
            'full_name': info['full_name'],
            'direction': direction,
            'expected_return': hist_return,
            'confidence': round(confidence, 3),
            'verified': False,
            'actual_return': None,
            'correct': None,
        }

        return prediction

    def verify(self, prediction, actual_return):
        """
        VERIFY: Check prediction against actual outcome.

        Returns updated prediction with verification.
        """
        direction = prediction['direction']

        # Determine if prediction was correct
        if direction == "BULLISH":
            correct = actual_return > 0
        elif direction == "BEARISH":
            correct = actual_return < 0
        else:  # NEUTRAL
            correct = abs(actual_return) < 0.5  # Within 0.5%

        prediction['verified'] = True
        prediction['actual_return'] = round(actual_return, 4)
        prediction['correct'] = correct

        return prediction

    def update(self, prediction):
        """
        UPDATE: Learn from verified prediction.

        Updates:
        1. Prediction log
        2. Accuracy statistics
        3. (Future: adjust base pattern weights)
        """
        if not prediction['verified']:
            raise ValueError("Cannot update with unverified prediction")

        # Log the prediction
        self.predictions['predictions'].append(prediction)

        # Update stats
        self.predictions['stats']['total'] += 1
        if prediction['correct']:
            self.predictions['stats']['correct'] += 1

        # Save
        self._save_log()

        return self.get_stats()

    def get_stats(self):
        """Get current oracle statistics."""
        stats = self.predictions['stats']
        total = stats['total']
        correct = stats['correct']

        if total == 0:
            accuracy = 0.5
        else:
            accuracy = correct / total

        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': round(accuracy, 3),
            'above_chance': accuracy > 0.5,
            'edge': round((accuracy - 0.5) * 100, 2),  # Edge in percentage points
        }

    def backtest(self, start_date='2020-01-01', end_date='2024-12-31'):
        """
        Backtest the oracle on historical data.

        This simulates running the prediction loop on past data.
        """
        if not YFINANCE_AVAILABLE:
            print("yfinance not available for backtest")
            return None

        print(f"Backtesting Trecena Oracle: {start_date} to {end_date}")

        # Fetch data
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100  # As percentage
        dates = spy.index[1:]

        results = []
        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            # PREDICT
            pred = self.predict(date)

            # VERIFY
            pred = self.verify(pred, ret)

            results.append(pred)

        # Compute stats
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        # By direction
        bullish = [r for r in results if r['direction'] == 'BULLISH']
        bearish = [r for r in results if r['direction'] == 'BEARISH']
        neutral = [r for r in results if r['direction'] == 'NEUTRAL']

        bull_acc = sum(1 for r in bullish if r['correct']) / len(bullish) if bullish else 0
        bear_acc = sum(1 for r in bearish if r['correct']) / len(bearish) if bearish else 0
        neut_acc = sum(1 for r in neutral if r['correct']) / len(neutral) if neutral else 0

        # Cumulative return if following predictions
        cumulative = 0
        for r in results:
            if r['direction'] == 'BULLISH':
                cumulative += r['actual_return']
            elif r['direction'] == 'BEARISH':
                cumulative -= r['actual_return']
            # NEUTRAL: no position

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'bullish_accuracy': round(bull_acc, 3),
            'bearish_accuracy': round(bear_acc, 3),
            'neutral_accuracy': round(neut_acc, 3),
            'cumulative_return': round(cumulative, 2),
            'results': results,
        }

    def run_live(self):
        """
        Run the oracle for TODAY.

        Prints prediction and instructions for verification.
        """
        today = datetime.now()
        pred = self.predict(today)

        print("=" * 60)
        print("TRECENA ORACLE - LIVE PREDICTION")
        print("=" * 60)
        print(f"\nDate: {pred['date']}")
        print(f"Trecena Day: {pred['trecena_day']} of 13")
        print(f"Day Sign: {pred['day_sign_name']}")
        print(f"Full Name: {pred['full_name']}")
        print(f"\n--- PREDICTION ---")
        print(f"Direction: {pred['direction']}")
        print(f"Expected Return: {pred['expected_return']:+.3f}%")
        print(f"Confidence: {pred['confidence']:.1%}")
        print(f"\n--- TRACK RECORD ---")
        stats = self.get_stats()
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Accuracy: {stats['accuracy']:.1%}")
        print(f"Edge: {stats['edge']:+.2f} percentage points")

        # Tomorrow's preview
        tomorrow = today + timedelta(days=1)
        tomorrow_pred = self.predict(tomorrow)
        print(f"\n--- TOMORROW ({tomorrow.strftime('%Y-%m-%d')}) ---")
        print(f"Trecena Day: {tomorrow_pred['trecena_day']}")
        print(f"Direction: {tomorrow_pred['direction']}")

        return pred


def main():
    """Main entry point."""
    oracle = TrecenaOracle()

    print("=" * 60)
    print("TRECENA ORACLE: AGI PREDICTION LOOP")
    print("=" * 60)

    # Run backtest
    print("\n--- BACKTEST (2020-2024) ---\n")
    backtest = oracle.backtest('2020-01-01', '2024-12-31')

    if backtest:
        print(f"Total trading days: {backtest['total']}")
        print(f"Correct predictions: {backtest['correct']}")
        print(f"Accuracy: {backtest['accuracy']:.1%}")
        print(f"Edge over random: {backtest['edge']:+.2f} pp")
        print(f"\nBy direction:")
        print(f"  BULLISH accuracy: {backtest['bullish_accuracy']:.1%}")
        print(f"  BEARISH accuracy: {backtest['bearish_accuracy']:.1%}")
        print(f"  NEUTRAL accuracy: {backtest['neutral_accuracy']:.1%}")
        print(f"\nCumulative return (if following): {backtest['cumulative_return']:+.1f}%")

        # Significance test
        from scipy import stats
        # Binomial test: is accuracy significantly above 50%?
        result = stats.binomtest(backtest['correct'], backtest['total'], 0.5, alternative='greater')
        p_value = result.pvalue
        print(f"\nStatistical significance:")
        print(f"  p-value (one-tailed): {p_value:.4e}")
        print(f"  Significant at 0.05: {'YES' if p_value < 0.05 else 'NO'}")

    # Live prediction
    print("\n" + "=" * 60)
    oracle.run_live()


if __name__ == "__main__":
    main()

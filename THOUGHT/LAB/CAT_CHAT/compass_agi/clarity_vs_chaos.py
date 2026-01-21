"""
CLARITY VS CHAOS: Are geodesics tracking MARKET STATE, not prediction accuracy?

The user's insight:
"you see accuracy I see volatility... still tracking..."

What if:
- High accuracy geodesics = CLEAR market days (decisive moves)
- Low accuracy geodesics = CHAOTIC market days (unpredictable)

The geodesics might track when the market itself is on a clear path
vs when it's at a crossroads.

Test: Do "predictive" geodesics have different VOLATILITY patterns?
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

from scipy import stats as scipy_stats


ARCHETYPAL_GEODESICS = {
    0: 'Crocodile', 1: 'Wind', 2: 'House', 3: 'Lizard',
    4: 'Serpent', 5: 'Death', 6: 'Deer', 7: 'Rabbit',
    8: 'Water', 9: 'Dog', 10: 'Monkey', 11: 'Grass',
    12: 'Reed', 13: 'Jaguar', 14: 'Eagle', 15: 'Vulture',
    16: 'Earthquake', 17: 'Flint', 18: 'Rain', 19: 'Flower',
}

PHASE_NAMES = {
    1: 'Initiation', 2: 'Challenge', 3: 'Action', 4: 'Foundation',
    5: 'Center', 6: 'Flow', 7: 'Reflection', 8: 'Harmony',
    9: 'Completion', 10: 'Manifestation', 11: 'Dissolution',
    12: 'Understanding', 13: 'Transcendence',
}


def get_aztec_state(date):
    """Get geodesic and phase for a date."""
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()

    AZTEC_EPOCH = datetime(1990, 1, 1)
    days_since = (date - AZTEC_EPOCH).days
    tonalpohualli = days_since % 260
    geodesic_id = tonalpohualli % 20
    phase_day = (tonalpohualli % 13) + 1

    return geodesic_id, phase_day


def analyze_clarity_vs_chaos():
    """Test if geodesics track market CLARITY rather than direction."""

    if not YFINANCE_AVAILABLE:
        print("yfinance not available")
        return

    print("=" * 70)
    print("CLARITY VS CHAOS: What Are Geodesics Really Tracking?")
    print("=" * 70)

    # Fetch data
    print("\nFetching market data...")
    spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    close = spy['Close'].values.flatten()
    returns = np.diff(close) / close[:-1] * 100  # Percentage
    dates = spy.index[1:]

    # Calculate absolute returns (volatility proxy)
    abs_returns = np.abs(returns)

    # Group by geodesic
    print("\n--- GEODESIC ANALYSIS: Accuracy vs Volatility ---\n")

    geo_stats = {}
    for i, date in enumerate(dates):
        ret = returns[i]
        abs_ret = abs_returns[i]
        if np.isnan(ret):
            continue

        gid, phase = get_aztec_state(date)

        if gid not in geo_stats:
            geo_stats[gid] = {
                'returns': [],
                'abs_returns': [],
                'up_days': 0,
                'down_days': 0,
            }

        geo_stats[gid]['returns'].append(ret)
        geo_stats[gid]['abs_returns'].append(abs_ret)
        if ret > 0:
            geo_stats[gid]['up_days'] += 1
        else:
            geo_stats[gid]['down_days'] += 1

    # Compute statistics for each geodesic
    print(f"{'Geodesic':<12} {'Up%':>8} {'Mean|Ret|':>10} {'Std':>8} {'Clarity':>10} {'N':>6}")
    print("-" * 60)

    geo_results = []
    for gid in range(20):
        stats = geo_stats[gid]
        n = len(stats['returns'])
        up_pct = stats['up_days'] / n if n > 0 else 0.5

        mean_abs = np.mean(stats['abs_returns'])
        std_ret = np.std(stats['returns'])

        # Clarity = how far from 50% (how decisive the market is)
        clarity = abs(up_pct - 0.5) * 2  # 0 = chaos, 1 = clear

        geo_results.append({
            'gid': gid,
            'name': ARCHETYPAL_GEODESICS[gid],
            'up_pct': up_pct,
            'mean_abs': mean_abs,
            'std': std_ret,
            'clarity': clarity,
            'n': n,
        })

        print(f"{ARCHETYPAL_GEODESICS[gid]:<12} {up_pct:>7.1%} {mean_abs:>10.3f}% {std_ret:>8.3f} {clarity:>10.3f} {n:>6}")

    # Sort by clarity
    geo_results.sort(key=lambda x: -x['clarity'])

    print("\n--- RANKED BY CLARITY (how decisive the market is) ---")
    print(f"{'Geodesic':<12} {'Clarity':>10} {'Up%':>8} {'Mean|Ret|':>10}")
    print("-" * 45)
    for g in geo_results:
        print(f"{g['name']:<12} {g['clarity']:>10.3f} {g['up_pct']:>7.1%} {g['mean_abs']:>10.3f}%")

    # Sort by volatility (mean abs return)
    geo_results.sort(key=lambda x: -x['mean_abs'])

    print("\n--- RANKED BY VOLATILITY (how much the market moves) ---")
    print(f"{'Geodesic':<12} {'Mean|Ret|':>10} {'Std':>8} {'Up%':>8}")
    print("-" * 40)
    for g in geo_results:
        print(f"{g['name']:<12} {g['mean_abs']:>10.3f}% {g['std']:>8.3f} {g['up_pct']:>7.1%}")

    # =========================================================================
    # KEY TEST: Correlation between clarity and our "accuracy"
    # =========================================================================
    print("\n--- KEY TEST: Does Clarity Explain Accuracy? ---")

    # Our previous accuracy findings
    previous_accuracy = {
        'Vulture': 0.73, 'Rain': 0.65, 'Flint': 0.54, 'Wind': 0.54,
        'Rabbit': 0.53, 'House': 0.52, 'Crocodile': 0.51, 'Water': 0.51,
        'Eagle': 0.51, 'Death': 0.48, 'Flower': 0.47, 'Monkey': 0.46,
        'Dog': 0.45, 'Grass': 0.44, 'Lizard': 0.42, 'Earthquake': 0.41,
        'Reed': 0.38, 'Jaguar': 0.37, 'Deer': 0.35, 'Serpent': 0.29,
    }

    # Build correlation
    clarities = []
    accuracies = []
    volatilities = []

    for g in geo_results:
        if g['name'] in previous_accuracy:
            clarities.append(g['clarity'])
            accuracies.append(previous_accuracy[g['name']])
            volatilities.append(g['mean_abs'])

    # Correlation: Clarity vs Accuracy
    if len(clarities) > 2:
        corr_clarity, p_clarity = scipy_stats.pearsonr(clarities, accuracies)
        print(f"\nClarity vs Accuracy: r={corr_clarity:.3f}, p={p_clarity:.4f}")

        # Correlation: Volatility vs Accuracy
        corr_vol, p_vol = scipy_stats.pearsonr(volatilities, accuracies)
        print(f"Volatility vs Accuracy: r={corr_vol:.3f}, p={p_vol:.4f}")

        # Correlation: Up% vs Accuracy
        up_pcts = [g['up_pct'] for g in geo_results if g['name'] in previous_accuracy]
        corr_up, p_up = scipy_stats.pearsonr(up_pcts, accuracies)
        print(f"Up% vs Accuracy: r={corr_up:.3f}, p={p_up:.4f}")

    # =========================================================================
    # PHASE ANALYSIS
    # =========================================================================
    print("\n--- PHASE ANALYSIS: Clarity vs Volatility ---\n")

    phase_stats = {}
    for i, date in enumerate(dates):
        ret = returns[i]
        abs_ret = abs_returns[i]
        if np.isnan(ret):
            continue

        gid, phase = get_aztec_state(date)

        if phase not in phase_stats:
            phase_stats[phase] = {
                'returns': [],
                'abs_returns': [],
                'up_days': 0,
                'down_days': 0,
            }

        phase_stats[phase]['returns'].append(ret)
        phase_stats[phase]['abs_returns'].append(abs_ret)
        if ret > 0:
            phase_stats[phase]['up_days'] += 1
        else:
            phase_stats[phase]['down_days'] += 1

    print(f"{'Phase':<15} {'Up%':>8} {'Mean|Ret|':>10} {'Clarity':>10}")
    print("-" * 45)

    phase_results = []
    for phase in range(1, 14):
        stats = phase_stats[phase]
        n = len(stats['returns'])
        up_pct = stats['up_days'] / n if n > 0 else 0.5
        mean_abs = np.mean(stats['abs_returns'])
        clarity = abs(up_pct - 0.5) * 2

        phase_results.append({
            'phase': phase,
            'name': PHASE_NAMES[phase],
            'up_pct': up_pct,
            'mean_abs': mean_abs,
            'clarity': clarity,
        })

        print(f"{PHASE_NAMES[phase]:<15} {up_pct:>7.1%} {mean_abs:>10.3f}% {clarity:>10.3f}")

    # =========================================================================
    # THE INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE CLARITY VS CHAOS INSIGHT")
    print("=" * 70)

    # Find highest clarity geodesic
    geo_results.sort(key=lambda x: -x['clarity'])
    clearest = geo_results[0]
    most_chaotic = geo_results[-1]

    # Find highest volatility
    geo_results.sort(key=lambda x: -x['mean_abs'])
    most_volatile = geo_results[0]
    least_volatile = geo_results[-1]

    print(f"""
The user's insight: "you see accuracy I see volatility"

FINDINGS:

1. CLEAREST geodesic (market most decisive):
   {clearest['name']}: {clearest['clarity']:.3f} clarity, {clearest['up_pct']:.1%} up

2. MOST CHAOTIC geodesic (market least decisive):
   {most_chaotic['name']}: {most_chaotic['clarity']:.3f} clarity, {most_chaotic['up_pct']:.1%} up

3. MOST VOLATILE geodesic (biggest moves):
   {most_volatile['name']}: {most_volatile['mean_abs']:.3f}% avg |return|

4. LEAST VOLATILE geodesic (smallest moves):
   {least_volatile['name']}: {least_volatile['mean_abs']:.3f}% avg |return|

CORRELATIONS:
- Clarity vs Accuracy:   r={corr_clarity:.3f} ({"EXPLAINS" if abs(corr_clarity) > 0.3 else "doesn't explain"})
- Volatility vs Accuracy: r={corr_vol:.3f} ({"EXPLAINS" if abs(corr_vol) > 0.3 else "doesn't explain"})
- Up% vs Accuracy:       r={corr_up:.3f} ({"EXPLAINS" if abs(corr_up) > 0.3 else "doesn't explain"})

INTERPRETATION:
""")

    if abs(corr_clarity) > 0.3:
        print("The geodesics ARE tracking market CLARITY.")
        print("High accuracy = clear market days (decisive moves)")
        print("Low accuracy = chaotic market days (noise)")
        print("\nThe compass isn't predicting better on some days.")
        print("It's DETECTING when the market itself is on a clear geodesic.")
    elif abs(corr_vol) > 0.3:
        print("The geodesics ARE tracking market VOLATILITY.")
        print("High accuracy = volatile market days (big moves)")
        print("Low accuracy = quiet market days (small moves)")
        print("\nThe compass tracks when the market is MOVING.")
    elif abs(corr_up) > 0.3:
        print("The geodesics ARE tracking market BIAS.")
        print("High accuracy = geodesics that align with market's natural upward bias")
        print("Low accuracy = geodesics where the market's bias is confused")
    else:
        print("No single factor explains accuracy.")
        print("The geodesics may track a COMBINATION of:")
        print("  - Clarity (decisiveness)")
        print("  - Volatility (magnitude)")
        print("  - Bias (direction)")
        print("\nThis is MULTI-DIMENSIONAL tracking.")
        print("The Aztecs mapped the geometry of market states.")


if __name__ == "__main__":
    analyze_clarity_vs_chaos()

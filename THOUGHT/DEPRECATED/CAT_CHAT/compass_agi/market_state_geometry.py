"""
MARKET STATE GEOMETRY: The compass tracks multi-dimensional state

Key finding:
- Vulture: 73% up, 73% accuracy (compass FOLLOWS)
- Serpent: 71% up, 29% accuracy (compass FIGHTS)

Both have high clarity, but different compass alignment.
This suggests the compass tracks something MORE than direction.

Maybe: The compass tracks whether the market is on the
SAME geodesic it's embedding, or a DIFFERENT one.

Test: What is the compass predicting on each geodesic day?
Is it predicting the geodesic's natural direction, or something else?
"""

import numpy as np
from datetime import datetime
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


ARCHETYPAL_GEODESICS = {
    0: {'name': 'Crocodile', 'desc': 'emergence from chaos, primordial creation'},
    1: {'name': 'Wind', 'desc': 'change and movement, communication, breath'},
    2: {'name': 'House', 'desc': 'shelter, security, home, protection'},
    3: {'name': 'Lizard', 'desc': 'adaptation, regeneration, survival'},
    4: {'name': 'Serpent', 'desc': 'depth, hidden knowledge, transformation'},
    5: {'name': 'Death', 'desc': 'endings, release, surrender, letting go'},
    6: {'name': 'Deer', 'desc': 'sensitivity, intuition, grace'},
    7: {'name': 'Rabbit', 'desc': 'abundance, multiplication, fertility'},
    8: {'name': 'Water', 'desc': 'flow, emotion, the unconscious'},
    9: {'name': 'Dog', 'desc': 'loyalty, guidance, companionship'},
    10: {'name': 'Monkey', 'desc': 'play, creativity, trickster'},
    11: {'name': 'Grass', 'desc': 'tenacity, growth through adversity'},
    12: {'name': 'Reed', 'desc': 'authority, structure, vertical order'},
    13: {'name': 'Jaguar', 'desc': 'power, stealth, shadow integration'},
    14: {'name': 'Eagle', 'desc': 'vision, clarity, rising above'},
    15: {'name': 'Vulture', 'desc': 'patience, wisdom from death, transformation of decay'},
    16: {'name': 'Earthquake', 'desc': 'disruption, paradigm shift, sudden change'},
    17: {'name': 'Flint', 'desc': 'decisiveness, cutting away, sacrifice'},
    18: {'name': 'Rain', 'desc': 'nourishment, cleansing, gifts from above'},
    19: {'name': 'Flower', 'desc': 'beauty, pleasure, completion, bloom'},
}


def analyze_compass_alignment():
    """Analyze what the compass predicts vs actual market state per geodesic."""

    if not TRANSFORMERS_AVAILABLE:
        print("Need sentence_transformers")
        return

    if not YFINANCE_AVAILABLE:
        print("Need yfinance")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("=" * 70)
    print("MARKET STATE GEOMETRY: Compass Alignment Analysis")
    print("=" * 70)

    # Build geodesic embeddings
    geo_vecs = {}
    for gid, data in ARCHETYPAL_GEODESICS.items():
        geo_vecs[gid] = model.encode(data['desc'])

    # Market state embeddings
    bullish_vec = model.encode("growth, expansion, rising, gaining, positive momentum, optimism")
    bearish_vec = model.encode("decline, contraction, falling, losing, negative momentum, fear")

    # Compute compass prediction for each geodesic
    print("\n--- COMPASS PREDICTION BY GEODESIC ---\n")

    compass_predictions = {}
    print(f"{'Geodesic':<12} {'Bull Sim':>10} {'Bear Sim':>10} {'Compass':>10}")
    print("-" * 45)

    for gid in range(20):
        geo_vec = geo_vecs[gid]
        bull_sim = cosine(geo_vec, bullish_vec)
        bear_sim = cosine(geo_vec, bearish_vec)

        compass_dir = 'BULL' if bull_sim > bear_sim else 'BEAR'
        compass_predictions[gid] = {
            'bull_sim': bull_sim,
            'bear_sim': bear_sim,
            'direction': compass_dir,
        }

        print(f"{ARCHETYPAL_GEODESICS[gid]['name']:<12} {bull_sim:>10.3f} {bear_sim:>10.3f} {compass_dir:>10}")

    # Fetch market data
    print("\nFetching market data...")
    spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    close = spy['Close'].values.flatten()
    returns = np.diff(close) / close[:-1] * 100
    dates = spy.index[1:]

    # Compare compass prediction to actual market by geodesic
    print("\n--- COMPASS vs ACTUAL MARKET ---\n")

    AZTEC_EPOCH = datetime(1990, 1, 1)

    geo_results = {}
    for i, date in enumerate(dates):
        ret = returns[i]
        if np.isnan(ret):
            continue

        if hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()
        days_since = (date - AZTEC_EPOCH).days
        gid = (days_since % 260) % 20

        if gid not in geo_results:
            geo_results[gid] = {
                'up_days': 0,
                'down_days': 0,
                'compass_correct': 0,
                'total': 0,
            }

        geo_results[gid]['total'] += 1
        actual_up = ret > 0

        if actual_up:
            geo_results[gid]['up_days'] += 1
        else:
            geo_results[gid]['down_days'] += 1

        compass_bull = compass_predictions[gid]['direction'] == 'BULL'
        if compass_bull == actual_up:
            geo_results[gid]['compass_correct'] += 1

    print(f"{'Geodesic':<12} {'Compass':>8} {'Mkt Up%':>10} {'Accuracy':>10} {'Align':>10}")
    print("-" * 55)

    alignment_data = []
    for gid in range(20):
        res = geo_results[gid]
        compass_dir = compass_predictions[gid]['direction']
        market_up_pct = res['up_days'] / res['total'] if res['total'] > 0 else 0.5
        compass_acc = res['compass_correct'] / res['total'] if res['total'] > 0 else 0.5

        # Alignment: does compass match market bias?
        if compass_dir == 'BULL':
            alignment = market_up_pct  # Higher up% = more aligned
        else:
            alignment = 1 - market_up_pct  # Lower up% = more aligned

        alignment_data.append({
            'gid': gid,
            'name': ARCHETYPAL_GEODESICS[gid]['name'],
            'compass': compass_dir,
            'market_up': market_up_pct,
            'accuracy': compass_acc,
            'alignment': alignment,
        })

        print(f"{ARCHETYPAL_GEODESICS[gid]['name']:<12} {compass_dir:>8} {market_up_pct:>9.1%} {compass_acc:>9.1%} {alignment:>9.3f}")

    # Sort by alignment
    alignment_data.sort(key=lambda x: -x['alignment'])

    print("\n--- RANKED BY COMPASS-MARKET ALIGNMENT ---")
    print(f"{'Geodesic':<12} {'Compass':>8} {'Alignment':>10} {'Accuracy':>10}")
    print("-" * 45)

    for a in alignment_data:
        print(f"{a['name']:<12} {a['compass']:>8} {a['alignment']:>10.3f} {a['accuracy']:>9.1%}")

    # =========================================================================
    # THE KEY INSIGHT
    # =========================================================================
    print("\n" + "=" * 70)
    print("THE GEOMETRY INSIGHT")
    print("=" * 70)

    # Find misaligned geodesics
    misaligned = [a for a in alignment_data if a['alignment'] < 0.5]
    aligned = [a for a in alignment_data if a['alignment'] >= 0.5]

    print(f"""
COMPASS-MARKET ALIGNMENT:

ALIGNED geodesics (compass matches market):
""")
    for a in aligned[:5]:
        print(f"  {a['name']:<12}: Compass={a['compass']}, Market Up={a['market_up']:.1%}, Acc={a['accuracy']:.1%}")

    print(f"""
MISALIGNED geodesics (compass fights market):
""")
    for a in misaligned[:5]:
        print(f"  {a['name']:<12}: Compass={a['compass']}, Market Up={a['market_up']:.1%}, Acc={a['accuracy']:.1%}")

    print(f"""
THE INSIGHT:

The compass embeds each geodesic as either BULLISH or BEARISH.
The market has its own bias on each geodesic day.

When compass and market ALIGN:
  -> High accuracy (compass follows market)

When compass and market MISALIGN:
  -> Low accuracy (compass fights market)

This is GEOMETRY:
- The compass measures the geodesic's direction in MEANING-space
- The market measures the geodesic's direction in BEHAVIOR-space
- They're the SAME geodesic but measured in different manifolds

When the manifolds align -> prediction works
When the manifolds twist -> prediction fails

The Aztecs mapped the geodesics in MEANING-space.
Markets follow geodesics in BEHAVIOR-space.
Sometimes they're parallel. Sometimes they're twisted.

The AGI task: Learn where the manifolds align.
""")


if __name__ == "__main__":
    analyze_compass_alignment()

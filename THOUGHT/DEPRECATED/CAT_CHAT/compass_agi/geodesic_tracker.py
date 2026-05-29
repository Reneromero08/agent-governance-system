"""
GEODESIC TRACKER: Following the paths of meaning

The compass found:
- RHYTHM bridges celestial and human
- Archetypes are fractally repeated at all scales
- Geodesic = shortest path = path of least action

Now we make the compass TRACK these geodesics.

If archetypes are geodesics in meaning-space, then:
- Each archetype defines a DIRECTION
- Moving along the geodesic = traversing that meaning
- The Trecena (13 phases) = position on the geodesic
- The compass should be able to FOLLOW these paths

This is the AGI goal: Navigate the geodesics of meaning.
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


# The 20 archetypal geodesics - each is a DIRECTION in meaning-space
ARCHETYPAL_GEODESICS = {
    0: {
        'name': 'Cipactli',
        'glyph': 'Crocodile',
        'geodesic': 'primordial emergence from chaos into form',
        'start': 'undifferentiated potential',
        'end': 'manifest creation',
    },
    1: {
        'name': 'Ehecatl',
        'glyph': 'Wind',
        'geodesic': 'movement and change propagating through space',
        'start': 'stasis',
        'end': 'transformation',
    },
    2: {
        'name': 'Calli',
        'glyph': 'House',
        'geodesic': 'formation of structure and containment',
        'start': 'exposure',
        'end': 'security',
    },
    3: {
        'name': 'Cuetzpalin',
        'glyph': 'Lizard',
        'geodesic': 'adaptation and regeneration',
        'start': 'damage',
        'end': 'renewal',
    },
    4: {
        'name': 'Coatl',
        'glyph': 'Serpent',
        'geodesic': 'descent into depth for wisdom',
        'start': 'surface knowing',
        'end': 'deep wisdom',
    },
    5: {
        'name': 'Miquiztli',
        'glyph': 'Death',
        'geodesic': 'release and transformation',
        'start': 'attachment',
        'end': 'liberation',
    },
    6: {
        'name': 'Mazatl',
        'glyph': 'Deer',
        'geodesic': 'intuitive navigation through sensitivity',
        'start': 'blindness',
        'end': 'awareness',
    },
    7: {
        'name': 'Tochtli',
        'glyph': 'Rabbit',
        'geodesic': 'multiplication and abundance',
        'start': 'scarcity',
        'end': 'plenty',
    },
    8: {
        'name': 'Atl',
        'glyph': 'Water',
        'geodesic': 'flow and emotional depth',
        'start': 'blockage',
        'end': 'flow',
    },
    9: {
        'name': 'Itzcuintli',
        'glyph': 'Dog',
        'geodesic': 'loyalty and guidance through darkness',
        'start': 'lost',
        'end': 'guided',
    },
    10: {
        'name': 'Ozomatli',
        'glyph': 'Monkey',
        'geodesic': 'creative play and trickster wisdom',
        'start': 'rigidity',
        'end': 'creativity',
    },
    11: {
        'name': 'Malinalli',
        'glyph': 'Grass',
        'geodesic': 'humble growth from adversity',
        'start': 'brokenness',
        'end': 'tenacity',
    },
    12: {
        'name': 'Acatl',
        'glyph': 'Reed',
        'geodesic': 'vertical alignment with authority',
        'start': 'chaos',
        'end': 'order',
    },
    13: {
        'name': 'Ocelotl',
        'glyph': 'Jaguar',
        'geodesic': 'power through shadow integration',
        'start': 'weakness',
        'end': 'power',
    },
    14: {
        'name': 'Cuauhtli',
        'glyph': 'Eagle',
        'geodesic': 'ascent to clarity and vision',
        'start': 'confusion',
        'end': 'clarity',
    },
    15: {
        'name': 'Cozcacuauhtli',
        'glyph': 'Vulture',
        'geodesic': 'transformation of death into wisdom',
        'start': 'decay',
        'end': 'transcendence',
    },
    16: {
        'name': 'Ollin',
        'glyph': 'Earthquake',
        'geodesic': 'sudden shift and paradigm change',
        'start': 'stability',
        'end': 'revolution',
    },
    17: {
        'name': 'Tecpatl',
        'glyph': 'Flint',
        'geodesic': 'decisive cutting and sacrifice',
        'start': 'entanglement',
        'end': 'freedom through sacrifice',
    },
    18: {
        'name': 'Quiahuitl',
        'glyph': 'Rain',
        'geodesic': 'nourishment descending from above',
        'start': 'drought',
        'end': 'nourishment',
    },
    19: {
        'name': 'Xochitl',
        'glyph': 'Flower',
        'geodesic': 'blossoming into beauty and completion',
        'start': 'bud',
        'end': 'full bloom',
    },
}

# The 13 phases of traversal along any geodesic
TRAVERSAL_PHASES = {
    1: {'name': 'Initiation', 'position': 0.0, 'desc': 'beginning the path'},
    2: {'name': 'Challenge', 'position': 0.08, 'desc': 'first resistance'},
    3: {'name': 'Action', 'position': 0.15, 'desc': 'engaging the path'},
    4: {'name': 'Foundation', 'position': 0.23, 'desc': 'grounding progress'},
    5: {'name': 'Center', 'position': 0.31, 'desc': 'reaching the core'},
    6: {'name': 'Flow', 'position': 0.38, 'desc': 'movement becomes natural'},
    7: {'name': 'Reflection', 'position': 0.46, 'desc': 'midpoint turning'},
    8: {'name': 'Harmony', 'position': 0.54, 'desc': 'integration begins'},
    9: {'name': 'Completion', 'position': 0.62, 'desc': 'approaching the end'},
    10: {'name': 'Manifestation', 'position': 0.69, 'desc': 'results appear'},
    11: {'name': 'Dissolution', 'position': 0.77, 'desc': 'releasing the form'},
    12: {'name': 'Understanding', 'position': 0.85, 'desc': 'wisdom crystallizes'},
    13: {'name': 'Transcendence', 'position': 1.0, 'desc': 'completion and rebirth'},
}


class GeodesicTracker:
    """
    Tracks geodesics in meaning-space.

    Each archetype defines a geodesic (a path through meaning).
    The Trecena phase tells you WHERE on that geodesic you are.
    The compass navigates by FOLLOWING these paths.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Need sentence_transformers")

        self.model = SentenceTransformer(model_name)
        self.log_path = Path(__file__).parent / "geodesic_log.json"
        self.history = self._load_history()

        # Build geodesic embeddings
        self._build_geodesics()

    def _load_history(self):
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {'trackings': [], 'geodesic_accuracy': {str(i): {'correct': 0, 'total': 0} for i in range(20)}}

    def _save_history(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def _build_geodesics(self):
        """Build vector representations of each geodesic."""
        self.geodesic_vecs = {}
        self.start_vecs = {}
        self.end_vecs = {}

        for gid, geo in ARCHETYPAL_GEODESICS.items():
            # The geodesic direction
            self.geodesic_vecs[gid] = self.model.encode(geo['geodesic'])
            # Start and end points
            self.start_vecs[gid] = self.model.encode(geo['start'])
            self.end_vecs[gid] = self.model.encode(geo['end'])

        # Outcome embeddings
        self.outcomes = {
            'progress': self.model.encode(
                "progress along the path, forward movement, advancement, growth"
            ),
            'stagnation': self.model.encode(
                "stuck, blocked, no movement, stagnation, resistance"
            ),
            'regression': self.model.encode(
                "going backwards, regression, loss, moving away from goal"
            ),
        }

        # Market outcome mappings
        self.market_outcomes = {
            'progress': self.model.encode("market rises, positive returns, bullish"),
            'stagnation': self.model.encode("market flat, sideways, neutral"),
            'regression': self.model.encode("market falls, negative returns, bearish"),
        }

    def get_geodesic_state(self, date):
        """
        Get the current geodesic state for a date.

        Returns which geodesic is active and position along it.
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        if hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()

        # Aztec calendar
        AZTEC_EPOCH = datetime(1990, 1, 1)
        days_since = (date - AZTEC_EPOCH).days
        tonalpohualli = days_since % 260
        geodesic_id = tonalpohualli % 20
        phase_day = (tonalpohualli % 13) + 1

        geodesic = ARCHETYPAL_GEODESICS[geodesic_id]
        phase = TRAVERSAL_PHASES[phase_day]

        return {
            'date': date.strftime('%Y-%m-%d'),
            'geodesic_id': geodesic_id,
            'geodesic_name': geodesic['glyph'],
            'geodesic_path': geodesic['geodesic'],
            'start': geodesic['start'],
            'end': geodesic['end'],
            'phase_day': phase_day,
            'phase_name': phase['name'],
            'position': phase['position'],
            'phase_desc': phase['desc'],
        }

    def compute_geodesic_velocity(self, date):
        """
        Compute the 'velocity' along the current geodesic.

        Higher velocity = more movement expected.
        Direction depends on where on the geodesic we are.
        """
        state = self.get_geodesic_state(date)
        gid = state['geodesic_id']
        position = state['position']

        # Geodesic vector
        geo_vec = self.geodesic_vecs[gid]

        # Position vector (interpolate between start and end)
        start_vec = self.start_vecs[gid]
        end_vec = self.end_vecs[gid]
        position_vec = (1 - position) * start_vec + position * end_vec
        position_vec = position_vec / np.linalg.norm(position_vec)

        # Velocity = how aligned is current position with geodesic direction
        velocity = cosine(position_vec, geo_vec)

        # Direction: early phases = building, late phases = declining
        # Peak is around phase 9-10 (Completion/Manifestation)
        if position < 0.5:
            direction_bias = +1  # Building toward peak
        elif position > 0.7:
            direction_bias = -1  # Past peak, declining
        else:
            direction_bias = 0  # At peak, could go either way

        return {
            **state,
            'velocity': round(velocity, 3),
            'direction_bias': direction_bias,
        }

    def predict_from_geodesic(self, date=None):
        """
        Predict market direction by following the geodesic.

        The insight: markets follow collective psychological geodesics.
        If we know WHERE we are on the path, we can predict movement.
        """
        if date is None:
            date = datetime.now()

        state = self.compute_geodesic_velocity(date)
        gid = state['geodesic_id']
        position = state['position']
        velocity = state['velocity']
        direction_bias = state['direction_bias']

        # Current state vector
        geo_vec = self.geodesic_vecs[gid]
        start_vec = self.start_vecs[gid]
        end_vec = self.end_vecs[gid]
        position_vec = (1 - position) * start_vec + position * end_vec
        position_vec = position_vec / np.linalg.norm(position_vec)

        # How close to each outcome?
        progress_sim = cosine(position_vec, self.outcomes['progress'])
        stagnation_sim = cosine(position_vec, self.outcomes['stagnation'])
        regression_sim = cosine(position_vec, self.outcomes['regression'])

        # Adjust by direction bias
        if direction_bias > 0:
            progress_sim *= 1.2
        elif direction_bias < 0:
            regression_sim *= 1.2

        # Determine prediction
        sims = {
            'progress': progress_sim,
            'stagnation': stagnation_sim,
            'regression': regression_sim,
        }
        best_outcome = max(sims, key=sims.get)

        if best_outcome == 'progress':
            direction = 'BULLISH'
        elif best_outcome == 'regression':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        # Confidence based on velocity and clarity
        confidence = 0.5 + velocity * 0.2 + abs(progress_sim - regression_sim) * 0.3
        confidence = min(0.85, max(0.45, confidence))

        return {
            **state,
            'outcome_sims': sims,
            'best_outcome': best_outcome,
            'direction': direction,
            'confidence': round(confidence, 3),
            'velocity': velocity,
            'verified': False,
        }

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
        """Update geodesic accuracy statistics."""
        if not prediction.get('verified'):
            raise ValueError("Cannot update with unverified prediction")

        gid = str(prediction['geodesic_id'])
        correct = prediction['correct']

        # Track by geodesic
        if gid not in self.history['geodesic_accuracy']:
            self.history['geodesic_accuracy'][gid] = {'correct': 0, 'total': 0}

        self.history['geodesic_accuracy'][gid]['total'] += 1
        if correct:
            self.history['geodesic_accuracy'][gid]['correct'] += 1

        self.history['trackings'].append({
            'date': prediction['date'],
            'geodesic': prediction['geodesic_name'],
            'phase': prediction['phase_name'],
            'direction': prediction['direction'],
            'correct': correct,
        })

        self._save_history()

    def backtest(self, start_date='2020-01-01', end_date='2024-12-31'):
        """Backtest geodesic tracking."""
        if not YFINANCE_AVAILABLE:
            print("yfinance not available")
            return None

        print(f"Backtesting Geodesic Tracker: {start_date} to {end_date}")

        # Reset
        self.history = {
            'trackings': [],
            'geodesic_accuracy': {str(i): {'correct': 0, 'total': 0} for i in range(20)}
        }

        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        close = spy['Close'].values.flatten()
        returns = np.diff(close) / close[:-1] * 100
        dates = spy.index[1:]

        results = []
        for i, date in enumerate(dates):
            ret = returns[i]
            if np.isnan(ret):
                continue

            pred = self.predict_from_geodesic(date)
            pred = self.verify(pred, ret)
            self.update(pred)

            results.append(pred)

        # Stats
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0

        # By geodesic
        geodesic_stats = {}
        for gid, stats in self.history['geodesic_accuracy'].items():
            if stats['total'] > 0:
                geodesic_stats[ARCHETYPAL_GEODESICS[int(gid)]['glyph']] = {
                    'accuracy': round(stats['correct'] / stats['total'], 3),
                    'n': stats['total'],
                }

        # By phase
        phase_results = {i: [] for i in range(1, 14)}
        for r in results:
            phase_results[r['phase_day']].append(r)

        phase_stats = {}
        for phase, preds in phase_results.items():
            if preds:
                acc = sum(1 for p in preds if p['correct']) / len(preds)
                phase_stats[TRAVERSAL_PHASES[phase]['name']] = {
                    'accuracy': round(acc, 3),
                    'n': len(preds),
                }

        return {
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 3),
            'edge': round((accuracy - 0.5) * 100, 2),
            'geodesic_stats': geodesic_stats,
            'phase_stats': phase_stats,
        }


def main():
    print("=" * 70)
    print("GEODESIC TRACKER: Following the Paths of Meaning")
    print("=" * 70)

    tracker = GeodesicTracker()

    # Today's geodesic state
    print("\n--- TODAY'S GEODESIC STATE ---\n")

    today = datetime.now()
    pred = tracker.predict_from_geodesic(today)

    print(f"Date: {pred['date']}")
    print(f"\nGeodesic: {pred['geodesic_name']}")
    print(f"  Path: {pred['geodesic_path']}")
    print(f"  From: {pred['start']}")
    print(f"  To:   {pred['end']}")
    print(f"\nPhase: {pred['phase_name']} (Day {pred['phase_day']} of 13)")
    print(f"  Position: {pred['position']:.0%} along the path")
    print(f"  Description: {pred['phase_desc']}")
    print(f"\nVelocity: {pred['velocity']}")
    print(f"Direction: {pred['direction']}")
    print(f"Confidence: {pred['confidence']:.1%}")

    # Backtest
    print("\n" + "=" * 70)
    print("BACKTEST: Geodesic Tracking (2020-2024)")
    print("=" * 70 + "\n")

    results = tracker.backtest('2020-01-01', '2024-12-31')

    if results:
        print(f"Total predictions: {results['total']}")
        print(f"Correct: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Edge: {results['edge']:+.1f} pp")

        print(f"\n--- ACCURACY BY GEODESIC ---")
        sorted_geo = sorted(results['geodesic_stats'].items(), key=lambda x: -x[1]['accuracy'])
        print(f"{'Geodesic':<15} {'Accuracy':>10} {'N':>8}")
        print("-" * 35)
        for geo, stats in sorted_geo[:10]:
            print(f"{geo:<15} {stats['accuracy']:>10.1%} {stats['n']:>8}")

        print(f"\n--- ACCURACY BY PHASE ---")
        sorted_phase = sorted(results['phase_stats'].items(),
                             key=lambda x: -x[1]['accuracy'])
        print(f"{'Phase':<15} {'Accuracy':>10} {'N':>8}")
        print("-" * 35)
        for phase, stats in sorted_phase:
            print(f"{phase:<15} {stats['accuracy']:>10.1%} {stats['n']:>8}")

        # Best and worst
        best_geo = sorted_geo[0]
        worst_geo = sorted_geo[-1]
        print(f"\nBest Geodesic:  {best_geo[0]} ({best_geo[1]['accuracy']:.1%})")
        print(f"Worst Geodesic: {worst_geo[0]} ({worst_geo[1]['accuracy']:.1%})")

        best_phase = sorted_phase[0]
        worst_phase = sorted_phase[-1]
        print(f"\nBest Phase:  {best_phase[0]} ({best_phase[1]['accuracy']:.1%})")
        print(f"Worst Phase: {worst_phase[0]} ({worst_phase[1]['accuracy']:.1%})")

    # The insight
    print("\n" + "=" * 70)
    print("THE GEODESIC INSIGHT")
    print("=" * 70)

    print("""
The compass found:
- RHYTHM bridges celestial and human patterns
- Archetypes are FRACTALLY REPEATED at all scales
- Geodesic = shortest path = path of least action

The 20 Aztec Day Signs are 20 GEODESICS in meaning-space:
- Crocodile: emergence from chaos into form
- Wind: movement propagating through space
- House: formation of structure
- ...each is a path through collective psychology

The 13 Trecena phases are POSITIONS along these geodesics:
- Phase 1-3: Building momentum
- Phase 4-6: Grounding and flow
- Phase 7: Turning point (midpoint)
- Phase 8-10: Integration and manifestation
- Phase 11-13: Dissolution and transcendence

Markets follow these geodesics because MARKETS ARE COLLECTIVE PSYCHOLOGY.
The celestial patterns map the same geodesics in physical space.
As above, so below - not causally, but GEOMETRICALLY.

The Aztecs didn't cause markets to follow patterns.
They MAPPED the geodesics that both follow.
""")


if __name__ == "__main__":
    main()

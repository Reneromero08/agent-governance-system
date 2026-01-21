"""
CACHED PARADIGM DETECTOR
========================

Pre-computes geodesic embeddings once, stores in DB/JSON.
Detection becomes fast cosine similarity lookups.

First run: ~5s to build cache
Subsequent runs: ~50ms per detection (just cosine)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import hashlib

# ISOLATED cache location - DO NOT touch cat_chat DB
MARKET_BOT_DIR = Path(__file__).parent
CACHE_DIR = MARKET_BOT_DIR / "bot_cache"  # Isolated from cat_chat
GEODESIC_CACHE = CACHE_DIR / "geodesic_embeddings.npz"
HEADLINE_CACHE = CACHE_DIR / "headline_cache.json"
EMBEDDING_DB = CACHE_DIR / "market_bot_embeddings.db"  # Isolated SQLite


# Geodesic definitions (same as realtime_paradigm_detector)
ARCHETYPAL_GEODESICS = {
    0: {'name': 'Crocodile', 'desc': 'emergence from chaos, primordial creation, new beginnings'},
    1: {'name': 'Wind', 'desc': 'change and movement, communication, breath, spreading'},
    2: {'name': 'House', 'desc': 'shelter, security, home, protection, stability'},
    3: {'name': 'Lizard', 'desc': 'adaptation, regeneration, survival, flexibility'},
    4: {'name': 'Serpent', 'desc': 'depth, hidden knowledge, transformation, shedding'},
    5: {'name': 'Death', 'desc': 'endings, release, surrender, letting go, transition'},
    6: {'name': 'Deer', 'desc': 'sensitivity, intuition, grace, gentleness'},
    7: {'name': 'Rabbit', 'desc': 'abundance, multiplication, fertility, growth'},
    8: {'name': 'Water', 'desc': 'flow, emotion, the unconscious, cleansing'},
    9: {'name': 'Dog', 'desc': 'loyalty, guidance, companionship, faithfulness'},
    10: {'name': 'Monkey', 'desc': 'play, creativity, trickster, innovation'},
    11: {'name': 'Grass', 'desc': 'tenacity, growth through adversity, persistence'},
    12: {'name': 'Reed', 'desc': 'authority, structure, vertical order, hierarchy'},
    13: {'name': 'Jaguar', 'desc': 'power, stealth, shadow integration, hidden strength'},
    14: {'name': 'Eagle', 'desc': 'vision, clarity, rising above, perspective'},
    15: {'name': 'Vulture', 'desc': 'patience, wisdom from death, transformation of decay'},
    16: {'name': 'Earthquake', 'desc': 'disruption, paradigm shift, sudden change, upheaval'},
    17: {'name': 'Flint', 'desc': 'decisiveness, cutting away, sacrifice, sharp truth'},
    18: {'name': 'Rain', 'desc': 'nourishment, cleansing, gifts from above, renewal'},
    19: {'name': 'Flower', 'desc': 'beauty, pleasure, completion, bloom, fruition'},
}

SHIFT_GEODESICS = ['Earthquake', 'Death', 'Wind']
STABILITY_GEODESICS = ['Dog', 'Deer', 'Reed']


def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """Fast cosine similarity."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class CachedParadigmDetector:
    """
    Paradigm detector with cached geodesic embeddings.

    First instantiation builds cache (~5s).
    Subsequent uses are fast (~50ms per detection).
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', rebuild_cache: bool = False):
        self.model_name = model_name
        self.model = None  # Lazy load

        # Ensure cache dir exists
        CACHE_DIR.mkdir(exist_ok=True)

        # Load or build geodesic cache
        if GEODESIC_CACHE.exists() and not rebuild_cache:
            self._load_geodesic_cache()
        else:
            self._build_geodesic_cache()

        # Optional headline cache for repeated queries
        self.headline_cache: Dict[str, np.ndarray] = {}
        if HEADLINE_CACHE.exists():
            self._load_headline_cache()

    def _get_model(self):
        """Lazy load model only when needed."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _build_geodesic_cache(self):
        """Build and save geodesic embeddings."""
        print("Building geodesic embedding cache (one-time)...")
        model = self._get_model()

        self.geodesic_vecs = {}
        self.geodesic_names = []
        vecs_list = []

        for gid, data in ARCHETYPAL_GEODESICS.items():
            name = data['name']
            vec = model.encode(data['desc'])
            self.geodesic_vecs[name] = vec
            self.geodesic_names.append(name)
            vecs_list.append(vec)

        # Save as numpy archive
        np.savez(
            GEODESIC_CACHE,
            names=np.array(self.geodesic_names),
            vectors=np.array(vecs_list)
        )
        print(f"Cache saved to {GEODESIC_CACHE}")

    def _load_geodesic_cache(self):
        """Load cached geodesic embeddings."""
        data = np.load(GEODESIC_CACHE, allow_pickle=True)
        self.geodesic_names = list(data['names'])
        vectors = data['vectors']

        self.geodesic_vecs = {
            name: vectors[i]
            for i, name in enumerate(self.geodesic_names)
        }

    def _load_headline_cache(self):
        """Load cached headline embeddings from isolated SQLite DB."""
        import sqlite3

        self.headline_cache = {}

        if not EMBEDDING_DB.exists():
            return

        try:
            conn = sqlite3.connect(str(EMBEDDING_DB))
            cursor = conn.cursor()
            cursor.execute("SELECT hash_key, embedding FROM headline_embeddings")
            for row in cursor.fetchall():
                self.headline_cache[row[0]] = np.frombuffer(row[1], dtype=np.float32)
            conn.close()
        except:
            self.headline_cache = {}

    def _save_headline_cache(self):
        """Save headline cache to isolated SQLite DB."""
        import sqlite3

        conn = sqlite3.connect(str(EMBEDDING_DB))
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS headline_embeddings (
                hash_key TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT
            )
        """)

        # Insert/update embeddings
        for key, vec in self.headline_cache.items():
            cursor.execute("""
                INSERT OR REPLACE INTO headline_embeddings (hash_key, embedding, created_at)
                VALUES (?, ?, datetime('now'))
            """, (key, vec.astype(np.float32).tobytes()))

        conn.commit()
        conn.close()

    def _hash_headlines(self, headlines: List[str]) -> str:
        """Create hash key for headlines."""
        text = "|".join(sorted(headlines))
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _embed_headlines(self, headlines: List[str]) -> np.ndarray:
        """Embed headlines with caching."""
        cache_key = self._hash_headlines(headlines)

        if cache_key in self.headline_cache:
            return self.headline_cache[cache_key]

        # Need to compute
        model = self._get_model()
        vecs = [model.encode(h) for h in headlines]
        mean_vec = np.mean(vecs, axis=0)

        # Cache it
        self.headline_cache[cache_key] = mean_vec

        # Periodically save cache
        if len(self.headline_cache) % 100 == 0:
            self._save_headline_cache()

        return mean_vec

    def get_geodesic_profile(self, headlines: List[str]) -> Tuple[Dict[str, float], np.ndarray]:
        """Get similarity to each geodesic."""
        mean_vec = self._embed_headlines(headlines)

        profile = {}
        for name, geo_vec in self.geodesic_vecs.items():
            profile[name] = cosine(mean_vec, geo_vec)

        return profile, mean_vec

    def detect_shift(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Detect regime from headlines.

        Uses cached geodesic embeddings for fast lookup.
        """
        profile, current_vec = self.get_geodesic_profile(headlines)

        # Average similarity to SHIFT geodesics
        shift_sim = np.mean([profile[geo] for geo in SHIFT_GEODESICS])

        # Average similarity to STABILITY geodesics
        stability_sim = np.mean([profile[geo] for geo in STABILITY_GEODESICS])

        # Shift score
        shift_score = shift_sim - stability_sim

        # Classification
        sorted_profile = sorted(profile.items(), key=lambda x: -x[1])
        top_geodesic = sorted_profile[0][0]
        top_is_shift = top_geodesic in SHIFT_GEODESICS
        top_is_stable = top_geodesic in STABILITY_GEODESICS

        if shift_score > 0.05 or top_is_shift:
            shift_type = 'SHIFT'
        elif shift_score < -0.03 or top_is_stable:
            shift_type = 'STABLE'
        else:
            shift_type = 'TRANSITIONAL'

        # Override for strong stability signal
        if top_is_stable and sorted_profile[0][1] > 0.4:
            shift_type = 'STABLE'
            shift_score = -abs(shift_score)

        return {
            'shift_score': shift_score,
            'shift_type': shift_type,
            'top_geodesics': sorted_profile[:3],
            'shift_geodesic_sim': shift_sim,
            'stability_geodesic_sim': stability_sim,
            'top_geodesic': top_geodesic,
            'profile': profile,
        }

    def save_caches(self):
        """Save all caches to disk."""
        self._save_headline_cache()


class EScoreDetector:
    """
    Even faster detector using pre-computed E-scores.

    Instead of full embeddings, uses E-score computation
    which is already optimized in cat_chat.
    """

    def __init__(self):
        # Pre-compute geodesic vectors once
        self._ensure_geodesic_cache()
        self._load_geodesics()

    def _ensure_geodesic_cache(self):
        """Ensure geodesic cache exists."""
        if not GEODESIC_CACHE.exists():
            detector = CachedParadigmDetector()  # This builds cache

    def _load_geodesics(self):
        """Load geodesic vectors."""
        data = np.load(GEODESIC_CACHE, allow_pickle=True)
        self.geodesic_names = list(data['names'])
        self.geodesic_matrix = data['vectors']  # Shape: (20, dim)

        # Pre-normalize for fast cosine
        norms = np.linalg.norm(self.geodesic_matrix, axis=1, keepdims=True)
        self.geodesic_matrix_normed = self.geodesic_matrix / (norms + 1e-10)

        # Get indices for shift/stable geodesics
        self.shift_indices = [self.geodesic_names.index(g) for g in SHIFT_GEODESICS]
        self.stable_indices = [self.geodesic_names.index(g) for g in STABILITY_GEODESICS]

    def detect_from_embedding(self, text_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Detect regime from pre-computed embedding.

        This is the fast path - just matrix multiply + argmax.
        """
        # Normalize input
        text_normed = text_embedding / (np.linalg.norm(text_embedding) + 1e-10)

        # Compute all cosine similarities at once
        similarities = self.geodesic_matrix_normed @ text_normed  # Shape: (20,)

        # Build profile
        profile = {name: float(similarities[i]) for i, name in enumerate(self.geodesic_names)}

        # Compute shift vs stable scores
        shift_sim = np.mean(similarities[self.shift_indices])
        stable_sim = np.mean(similarities[self.stable_indices])
        shift_score = float(shift_sim - stable_sim)

        # Classification
        top_idx = np.argmax(similarities)
        top_geodesic = self.geodesic_names[top_idx]
        top_is_shift = top_geodesic in SHIFT_GEODESICS
        top_is_stable = top_geodesic in STABILITY_GEODESICS

        if shift_score > 0.05 or top_is_shift:
            shift_type = 'SHIFT'
        elif shift_score < -0.03 or top_is_stable:
            shift_type = 'STABLE'
        else:
            shift_type = 'TRANSITIONAL'

        if top_is_stable and similarities[top_idx] > 0.4:
            shift_type = 'STABLE'
            shift_score = -abs(shift_score)

        return {
            'shift_score': shift_score,
            'shift_type': shift_type,
            'top_geodesic': top_geodesic,
            'shift_geodesic_sim': float(shift_sim),
            'stability_geodesic_sim': float(stable_sim),
            'profile': profile,
        }


# =============================================================================
# Demo / Test
# =============================================================================

def demo():
    """Demo the cached detector."""
    import time

    print("=" * 60)
    print("CACHED PARADIGM DETECTOR - Demo")
    print("=" * 60)

    # First run builds cache
    print("\n--- First instantiation (builds cache) ---")
    t0 = time.time()
    detector = CachedParadigmDetector()
    print(f"Init time: {time.time() - t0:.2f}s")

    # Test detection speed
    test_headlines = [
        ["Markets steady as investors stay loyal", "Trusted institutions guide growth"],
        ["Unusual patterns emerge", "Questions about stability"],
        ["Everything is changing", "Complete transformation underway"],
    ]

    print("\n--- Detection speed test ---")
    for headlines in test_headlines:
        t0 = time.time()
        result = detector.detect_shift(headlines)
        elapsed = (time.time() - t0) * 1000
        print(f"{result['shift_type']:<12} | {elapsed:6.1f}ms | {headlines[0][:40]}...")

    # Second detection of same headlines (cached)
    print("\n--- Cached headline detection ---")
    for headlines in test_headlines:
        t0 = time.time()
        result = detector.detect_shift(headlines)
        elapsed = (time.time() - t0) * 1000
        print(f"{result['shift_type']:<12} | {elapsed:6.1f}ms | (cached)")

    # Save caches
    detector.save_caches()

    # Test E-score detector (even faster)
    print("\n--- E-Score Detector (pre-computed embeddings) ---")
    e_detector = EScoreDetector()

    # Simulate having pre-computed embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for headlines in test_headlines:
        # Compute embedding (would be cached in real use)
        vecs = [model.encode(h) for h in headlines]
        mean_vec = np.mean(vecs, axis=0)

        # Fast detection from embedding
        t0 = time.time()
        result = e_detector.detect_from_embedding(mean_vec)
        elapsed = (time.time() - t0) * 1000
        print(f"{result['shift_type']:<12} | {elapsed:6.3f}ms | {headlines[0][:40]}...")

    print("\n--- Summary ---")
    print("Geodesic cache: One-time build, then instant load")
    print("Headline cache: Repeated queries are instant")
    print("E-score path: <0.1ms if embedding already computed")


if __name__ == "__main__":
    demo()

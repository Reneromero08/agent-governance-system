"""
PRIME RADIANT
=============

The database that holds precomputed signal embeddings and historical data.
Named after Asimov's device that holds the Seldon Plan equations.

This is the "memory" of the Psychohistory bot - all computations are
deterministic lookups and formula applications, not LLM reasoning.
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from .signal_vocabulary import (
    get_all_signals,
    get_signal_descriptions,
    SignalDefinition,
    SignalState,
    AssetClass,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MARKET_BOT_DIR = Path(__file__).parent.parent
RADIANT_DIR = MARKET_BOT_DIR / "radiant_cache"
RADIANT_DB = RADIANT_DIR / "prime_radiant.db"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegimeRecord:
    """Historical regime detection record."""
    timestamp: str
    asset: str
    R_value: float
    alpha: float
    Df: float
    gate_status: str
    regime: str
    signals_json: str


@dataclass
class AlphaRecord:
    """Alpha drift tracking record."""
    timestamp: str
    alpha: float
    drift_from_05: float
    warning_level: int  # 0=none, 1=watch, 2=alert, 3=critical


# =============================================================================
# PRIME RADIANT CLASS
# =============================================================================

class PrimeRadiant:
    """
    The Prime Radiant database for Psychohistory trading.

    Stores:
    - Precomputed signal embeddings (one-time)
    - Historical R-values and regimes
    - Alpha trajectory for drift detection

    All computations are deterministic - no LLM reasoning here.
    """

    def __init__(self, db_path: Optional[Path] = None, rebuild: bool = False):
        """
        Initialize Prime Radiant.

        Args:
            db_path: Path to SQLite database. Defaults to radiant_cache/prime_radiant.db
            rebuild: If True, rebuild embeddings even if they exist
        """
        self.db_path = db_path or RADIANT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()

        # Check if embeddings need to be built
        if rebuild or not self._embeddings_exist():
            self._build_embeddings()

        # Load embeddings into memory for fast lookup
        self.signal_embeddings = self._load_embeddings()

    def _init_schema(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Signal embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_embeddings (
                signal_id TEXT PRIMARY KEY,
                category TEXT,
                description TEXT,
                embedding BLOB,
                created_at TEXT
            )
        """)

        # Regime history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset TEXT,
                R_value REAL,
                alpha REAL,
                Df REAL,
                gate_status TEXT,
                regime TEXT,
                signals_json TEXT
            )
        """)

        # Alpha trajectory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpha_trajectory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alpha REAL,
                drift_from_05 REAL,
                warning_level INTEGER
            )
        """)

        # Create indices for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_timestamp
            ON regime_history(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_asset
            ON regime_history(asset)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alpha_timestamp
            ON alpha_trajectory(timestamp)
        """)

        conn.commit()
        conn.close()

    def _embeddings_exist(self) -> bool:
        """Check if embeddings have been computed."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM signal_embeddings")
        count = cursor.fetchone()[0]
        conn.close()

        # Check if we have all signals
        expected = len(get_all_signals())
        return count >= expected

    def _build_embeddings(self):
        """Build embeddings for all signals using SentenceTransformer."""
        print("Building signal embeddings (one-time operation)...")

        # Import here to avoid loading model if not needed
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        signals = get_all_signals()
        descriptions = get_signal_descriptions()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for signal in signals:
            # Embed the description
            embedding = model.encode(signal.description)
            embedding_bytes = embedding.astype(np.float32).tobytes()

            # Store in database
            cursor.execute("""
                INSERT OR REPLACE INTO signal_embeddings
                (signal_id, category, description, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                signal.signal_id,
                signal.category.value,
                signal.description,
                embedding_bytes,
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        print(f"Embedded {len(signals)} signals into Prime Radiant")

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all embeddings into memory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT signal_id, embedding FROM signal_embeddings")

        embeddings = {}
        for row in cursor.fetchall():
            signal_id = row[0]
            embedding = np.frombuffer(row[1], dtype=np.float32)
            embeddings[signal_id] = embedding

        conn.close()
        return embeddings

    # =========================================================================
    # EMBEDDING ACCESS
    # =========================================================================

    def get_embedding(self, signal_id: str) -> Optional[np.ndarray]:
        """Get embedding for a single signal."""
        return self.signal_embeddings.get(signal_id)

    def get_embeddings(self, signal_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple signals."""
        return {
            sid: self.signal_embeddings[sid]
            for sid in signal_ids
            if sid in self.signal_embeddings
        }

    def state_to_vector(self, state: SignalState) -> np.ndarray:
        """
        Convert SignalState to weighted embedding vector.

        This is the key transformation: market state -> semantic vector
        """
        weights = state.to_vector_weights()

        if not weights:
            # No active signals - return zero vector
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Weighted sum of embeddings
        total_weight = sum(weights.values())
        weighted_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        for signal_id, weight in weights.items():
            if signal_id in self.signal_embeddings:
                weighted_vec += weight * self.signal_embeddings[signal_id]

        # Normalize
        weighted_vec /= total_weight

        return weighted_vec

    def states_to_vectors(self, states: List[SignalState]) -> List[np.ndarray]:
        """Convert multiple states to vectors."""
        return [self.state_to_vector(s) for s in states]

    # =========================================================================
    # REGIME HISTORY
    # =========================================================================

    def record_regime(
        self,
        timestamp: str,
        asset: str,
        R_value: float,
        alpha: float,
        Df: float,
        gate_status: str,
        regime: str,
        signals: Dict[str, float],
    ):
        """Record a regime detection result."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO regime_history
            (timestamp, asset, R_value, alpha, Df, gate_status, regime, signals_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            asset,
            R_value,
            alpha,
            Df,
            gate_status,
            regime,
            json.dumps(signals),
        ))

        conn.commit()
        conn.close()

    def get_regime_history(
        self,
        asset: Optional[str] = None,
        limit: int = 100
    ) -> List[RegimeRecord]:
        """Get recent regime history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if asset:
            cursor.execute("""
                SELECT timestamp, asset, R_value, alpha, Df, gate_status, regime, signals_json
                FROM regime_history
                WHERE asset = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (asset, limit))
        else:
            cursor.execute("""
                SELECT timestamp, asset, R_value, alpha, Df, gate_status, regime, signals_json
                FROM regime_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        records = []
        for row in cursor.fetchall():
            records.append(RegimeRecord(
                timestamp=row[0],
                asset=row[1],
                R_value=row[2],
                alpha=row[3],
                Df=row[4],
                gate_status=row[5],
                regime=row[6],
                signals_json=row[7],
            ))

        conn.close()
        return records

    def get_R_history(self, asset: str, limit: int = 100) -> List[float]:
        """Get recent R values for an asset."""
        records = self.get_regime_history(asset, limit)
        return [r.R_value for r in records]

    # =========================================================================
    # ALPHA TRAJECTORY
    # =========================================================================

    def record_alpha(
        self,
        timestamp: str,
        alpha: float,
        warning_level: int = 0
    ):
        """Record alpha value for drift tracking."""
        drift_from_05 = abs(alpha - 0.5)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alpha_trajectory
            (timestamp, alpha, drift_from_05, warning_level)
            VALUES (?, ?, ?, ?)
        """, (timestamp, alpha, drift_from_05, warning_level))

        conn.commit()
        conn.close()

    def get_alpha_history(self, limit: int = 100) -> List[AlphaRecord]:
        """Get recent alpha trajectory."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, alpha, drift_from_05, warning_level
            FROM alpha_trajectory
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        records = []
        for row in cursor.fetchall():
            records.append(AlphaRecord(
                timestamp=row[0],
                alpha=row[1],
                drift_from_05=row[2],
                warning_level=row[3],
            ))

        conn.close()
        return records

    def get_alpha_values(self, limit: int = 100) -> List[float]:
        """Get recent alpha values only."""
        records = self.get_alpha_history(limit)
        return [r.alpha for r in records]

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM signal_embeddings")
        n_embeddings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM regime_history")
        n_regimes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alpha_trajectory")
        n_alphas = cursor.fetchone()[0]

        conn.close()

        return {
            "n_signal_embeddings": n_embeddings,
            "n_regime_records": n_regimes,
            "n_alpha_records": n_alphas,
            "db_path": str(self.db_path),
        }


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRIME RADIANT - Psychohistory Market Bot")
    print("=" * 60)

    # Initialize (will build embeddings on first run)
    radiant = PrimeRadiant()

    # Show stats
    stats = radiant.get_stats()
    print(f"\n--- Database Stats ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test state to vector conversion
    print("\n--- State to Vector Test ---")
    test_state = SignalState(
        signals={
            "trend_up": 0.8,
            "volume_surge": 0.6,
            "bullish_news": 0.7,
        },
        timestamp=datetime.now().isoformat(),
        asset="SPY",
        asset_class=AssetClass.ALL,
    )

    vec = radiant.state_to_vector(test_state)
    print(f"  State: {test_state.signals}")
    print(f"  Vector shape: {vec.shape}")
    print(f"  Vector norm: {np.linalg.norm(vec):.4f}")
    print(f"  Net bias: {test_state.net_bias():.2f}")

    # Test embedding lookup
    print("\n--- Embedding Lookup Test ---")
    for signal_id in ["trend_up", "breakdown", "vol_spike"]:
        emb = radiant.get_embedding(signal_id)
        if emb is not None:
            print(f"  {signal_id}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    print("\n--- Prime Radiant Ready ---")

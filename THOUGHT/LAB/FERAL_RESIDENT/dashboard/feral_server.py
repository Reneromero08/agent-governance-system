#!/usr/bin/env python3
"""
Feral Dashboard Server

FastAPI server with WebSocket support for real-time updates.
Provides API endpoints for:
- Mind state queries
- Chat with Feral
- Daemon control (start/stop/configure)
- Activity log

Run with: python feral_server.py
Open: http://localhost:8420
"""

# Suppress transformers/tokenizers output BEFORE any imports
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import threading

# Add paths - FERAL_RESIDENT is the canonical source
# Now dashboard is inside FERAL_RESIDENT, so adjust paths accordingly
DASHBOARD_DIR = Path(__file__).resolve().parent
FERAL_RESIDENT_PATH = DASHBOARD_DIR.parent  # dashboard/ -> FERAL_RESIDENT/
REPO_ROOT = FERAL_RESIDENT_PATH.parents[2]  # FERAL_RESIDENT -> LAB -> THOUGHT -> repo
CORTEX_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic"
sys.path.insert(0, str(FERAL_RESIDENT_PATH))  # Canonical daemon location
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CORTEX_PATH))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn websockets")
    sys.exit(1)

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Lazy imports - these load transformers, so import inside functions AFTER TUI starts
FeralDaemon = None
ActivityEvent = None
VectorResident = None

def _lazy_import_feral():
    global FeralDaemon, ActivityEvent, VectorResident
    if FeralDaemon is None:
        # Suppress all warnings during import
        import warnings
        import logging
        warnings.filterwarnings('ignore')
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

        from autonomic.feral_daemon import FeralDaemon as FD, ActivityEvent as AE
        from cognition.vector_brain import VectorResident as VR
        FeralDaemon = FD
        ActivityEvent = AE
        VectorResident = VR

# Optional: CORTEX query module for constellation visualization
try:
    import query as cortex_query
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False


# =============================================================================
# Performance Cache Infrastructure
# =============================================================================

class TTLCache:
    """Simple TTL-based cache for expensive computations."""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self.default_ttl = default_ttl

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        ttl = ttl or self.default_ttl
        if time.time() - self._timestamps[key] > ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]

    def set(self, key: str, value: Any):
        """Store value with current timestamp."""
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def invalidate(self, key: str = None):
        """Clear specific key or all cache."""
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()


# Global caches
_constellation_cache = TTLCache(default_ttl=300)  # 5 min default
_config_cache: Dict[str, Any] = {}  # Cached config.json
_config_mtime: float = 0  # Last modified time


def get_cache_config() -> Dict[str, Any]:
    """Get cache settings from config.json (with file mtime caching)."""
    global _config_cache, _config_mtime
    config_path = FERAL_RESIDENT_PATH / "config.json"
    try:
        mtime = config_path.stat().st_mtime
        if mtime > _config_mtime or not _config_cache:
            with open(config_path, 'r') as f:
                _config_cache = json.load(f)
            _config_mtime = mtime
    except Exception:
        pass
    return _config_cache.get('cache', {})


# =============================================================================
# Pydantic Models
# =============================================================================

class ThinkRequest(BaseModel):
    query: str


class ThinkResponse(BaseModel):
    response: str
    E_resonance: float
    E_compression: float
    gate_open: bool
    mind_Df: float
    distance_from_start: float


class BehaviorConfigRequest(BaseModel):
    behavior: str
    enabled: Optional[bool] = None
    interval: Optional[int] = None


class SmasherStartRequest(BaseModel):
    delay_ms: int = 100       # Milliseconds between chunks
    batch_size: int = 10      # Chunks per batch
    batch_pause_ms: int = 500 # Pause between batches
    max_chunks: int = 0       # 0 = unlimited


class DaemonStatus(BaseModel):
    running: bool
    uptime_seconds: float
    thread_id: str
    behaviors: Dict[str, Any]
    activity_count: int


# =============================================================================
# WebSocket Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for broadcasting"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)


# =============================================================================
# Hot Reload File Watcher
# =============================================================================

class HotReloadWatcher:
    """Watches static files and triggers browser reload on changes"""

    def __init__(self, watch_dir: Path, manager_ref):
        self.watch_dir = watch_dir
        self.manager = manager_ref
        self.file_mtimes: Dict[str, float] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop):
        """Start watching in background thread"""
        self._loop = loop
        self.running = True
        self._scan_files()  # Initial scan
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _scan_files(self):
        """Scan all files and record mtimes"""
        for path in self.watch_dir.rglob("*"):
            if path.is_file():
                try:
                    self.file_mtimes[str(path)] = path.stat().st_mtime
                except Exception:
                    pass

    def _watch_loop(self):
        """Polling loop - checks for file changes every 500ms"""
        import time
        while self.running:
            time.sleep(0.5)
            changed = False

            for path in self.watch_dir.rglob("*"):
                if path.is_file():
                    try:
                        mtime = path.stat().st_mtime
                        path_str = str(path)
                        if path_str in self.file_mtimes:
                            if mtime > self.file_mtimes[path_str]:
                                changed = True
                        self.file_mtimes[path_str] = mtime
                    except Exception:
                        pass

            if changed and self._loop:
                # Trigger reload via WebSocket
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast({'type': 'hot_reload'}),
                    self._loop
                )


# =============================================================================
# Global State
# =============================================================================

manager = ConnectionManager()
hot_reload_watcher: Optional[HotReloadWatcher] = None
daemon = None  # Type: FeralDaemon (lazy loaded)
resident = None  # Type: VectorResident (lazy loaded)
tui_state = None  # TUI state for activity updates
broadcast_loop = None  # Main event loop for thread-safe WebSocket broadcasts


def get_resident():
    """Get or create the resident instance"""
    global resident
    _lazy_import_feral()  # Load transformers NOW
    if resident is None:
        db_path = FERAL_RESIDENT_PATH / "data" / "db" / "feral_eternal.db"
        db_path.parent.mkdir(exist_ok=True)
        # Skip paper loading on startup - papers are in cassettes and loaded on demand
        resident = VectorResident(thread_id="eternal", db_path=str(db_path), load_papers=False)
    return resident


def get_daemon():
    """Get or create the daemon instance"""
    global daemon
    _lazy_import_feral()  # Load transformers NOW
    if daemon is None:
        daemon = FeralDaemon(resident=get_resident(), thread_id="eternal")

        # Smash event throttling - batch and rate limit WebSocket broadcasts
        smash_batch = []
        smash_batch_lock = threading.Lock()
        last_smash_broadcast = [0.0]  # Use list for mutable closure
        SMASH_THROTTLE_MS = 100  # Broadcast at most every 100ms
        SMASH_BATCH_SIZE = 10   # Or when batch reaches this size

        # Add WebSocket callback - uses thread-safe broadcast scheduling
        def on_activity(event: ActivityEvent):
            global broadcast_loop
            try:
                # Update TUI if available (thread-safe - just appending to list)
                if tui_state is not None:
                    paper = event.details.get('paper', '')
                    chunk_id = event.details.get('chunk_id', '')
                    if paper and chunk_id:
                        tui_state.add_activity(f"[{event.action}] {paper}:{chunk_id} - {event.summary[:30]}")
                    else:
                        tui_state.add_activity(f"[{event.action}] {event.summary[:50]}")
            except Exception as e:
                if tui_state is not None:
                    tui_state.add_activity(f"[ERROR] {str(e)[:40]}")

            # Thread-safe broadcast helper - uses global broadcast_loop
            def safe_broadcast(msg):
                global broadcast_loop
                loop = broadcast_loop
                if loop is None:
                    # Try to get running loop (works if called from async context)
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(manager.broadcast(msg))
                        return
                    except RuntimeError:
                        pass
                    # DEBUG: Log when loop is None
                    if tui_state is not None:
                        tui_state.add_activity(f"[WS] NO LOOP! Conns={len(manager.active_connections)}")
                    return  # No loop available yet

                # Use the global main loop for thread-safe broadcast
                try:
                    future = asyncio.run_coroutine_threadsafe(manager.broadcast(msg), loop)
                    # DEBUG: Log successful schedule (only for smash_hit)
                    if msg.get('type') == 'smash_hit' and tui_state is not None:
                        conns = len(manager.active_connections)
                        tui_state.add_activity(f"[WS] Broadcast smash_hit to {conns} clients")
                except Exception as e:
                    if tui_state is not None:
                        tui_state.add_activity(f"[WS] ERROR: {str(e)[:30]}")

            # For smash events, skip verbose activity log broadcast (just send smash_hit)
            if event.action != 'smash':
                # Broadcast activity to activity log (non-smash events only)
                safe_broadcast({
                    'type': 'activity',
                    'data': {
                        'timestamp': event.timestamp,
                        'action': event.action,
                        'summary': event.summary,
                        'details': event.details
                    }
                })

            # Broadcast constellation events for dynamic 3D visualization
            if event.details.get('chunk_id'):
                chunk_id = event.details['chunk_id']
                paper = event.details.get('paper', 'papers')
                heading = event.details.get('heading', '')[:25] if event.details.get('heading') else ''
                is_new = event.details.get('is_new_node', False)
                # Use consistent node ID format matching constellation: chunk:{receipt_id}
                node_id = event.details.get('full_node_id') or f"chunk:{chunk_id}"
                similar_to = event.details.get('similar_to')
                similar_E = event.details.get('similar_E', 0)

                if event.action == 'smash':
                    # THROTTLED: Batch smash events and send periodically
                    smash_event = {
                        'node_id': node_id,
                        'paper': paper,
                        'chunk_id': chunk_id,
                        'E': event.details.get('E', 0),
                        'gate_open': event.details.get('gate_open', False),
                        'is_new_node': is_new,
                        'similar_to': similar_to,
                        'similar_E': similar_E,
                        'rate': event.details.get('rate', 0)
                    }

                    now = time.time() * 1000  # ms
                    should_flush = False
                    batch_to_send = []  # Initialize before the with block

                    with smash_batch_lock:
                        smash_batch.append(smash_event)
                        elapsed = now - last_smash_broadcast[0]
                        if len(smash_batch) >= SMASH_BATCH_SIZE or elapsed >= SMASH_THROTTLE_MS:
                            should_flush = True
                            batch_to_send = smash_batch.copy()
                            smash_batch.clear()
                            last_smash_broadcast[0] = now

                    if should_flush and batch_to_send:
                        # Send latest hit for UI update + batch stats
                        latest = batch_to_send[-1]
                        safe_broadcast({
                            'type': 'smash_hit',
                            'data': {
                                **latest,
                                'batch_size': len(batch_to_send),
                                'batch': batch_to_send  # Include full batch for bulk graph updates
                            }
                        })
                elif is_new:
                    safe_broadcast({
                        'type': 'node_discovered',
                        'data': {
                            'node_id': node_id,
                            'label': heading or f"chunk-{chunk_id}",
                            'similar_to': similar_to,
                            'activity_type': event.action,
                            'paper': paper
                        }
                    })
                else:
                    safe_broadcast({
                        'type': 'node_activated',
                        'data': {
                            'node_id': node_id,
                            'similar_to': similar_to,
                            'activity_type': event.action
                        }
                    })

        daemon.add_callback(on_activity)

        # Store batch clear function on daemon for stop_smasher to call
        def clear_smash_batch():
            with smash_batch_lock:
                smash_batch.clear()
        daemon._clear_smash_batch = clear_smash_batch

    return daemon


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global hot_reload_watcher, broadcast_loop

    # Initialize resident and daemon
    get_resident()
    get_daemon()

    # Set the global event loop for thread-safe WebSocket broadcasts
    broadcast_loop = asyncio.get_running_loop()

    # Start hot reload watcher for live editing
    static_dir = Path(__file__).parent / "static"
    hot_reload_watcher = HotReloadWatcher(static_dir, manager)
    hot_reload_watcher.start(broadcast_loop)

    yield

    # Shutdown
    if hot_reload_watcher:
        hot_reload_watcher.stop()
    if daemon and daemon.running:
        await daemon.stop()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Feral Dashboard",
    description="Control center for Feral Resident",
    version="2.0.0",
    lifespan=lifespan
)

# Static files - caching controlled by config.json cache.dev_mode
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.middleware("http")
async def static_cache_middleware(request, call_next):
    """Apply caching headers based on config.json cache settings.

    In dev_mode (default): no-cache for hot reload
    In production: long-lived cache for performance
    """
    response = await call_next(request)
    path = request.url.path

    # Only apply to static assets
    if path.startswith('/static/'):
        cache_cfg = get_cache_config()
        dev_mode = cache_cfg.get('dev_mode', True)  # Default to dev mode for safety

        if dev_mode:
            # Development: no caching for hot reload
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        else:
            # Production: aggressive caching (configurable max-age)
            max_age = cache_cfg.get('static_max_age', 86400)  # Default 1 day
            if path.endswith('.js') or path.endswith('.css'):
                response.headers["Cache-Control"] = f"public, max-age={max_age}, immutable"
            elif path.endswith('.html'):
                # HTML should revalidate more often
                response.headers["Cache-Control"] = "public, max-age=300, must-revalidate"
            else:
                response.headers["Cache-Control"] = f"public, max-age={max_age}"

    return response


# =============================================================================
# Routes: Cache Management
# =============================================================================

@app.get("/api/cache/status")
async def get_cache_status():
    """Get cache status and stats for all caches."""
    cache_cfg = get_cache_config()

    # Try to get embedding cache stats
    embedding_stats = None
    try:
        from model_client import get_cache_stats
        embedding_stats = get_cache_stats()
    except ImportError:
        pass

    return {
        'ok': True,
        'config': cache_cfg,
        'constellation': {
            'cached': bool(_constellation_cache.get('constellation:500')),
            'ttl_sec': cache_cfg.get('constellation_ttl_sec', 300)
        },
        'embedding': embedding_stats,
        'static_assets': {
            'dev_mode': cache_cfg.get('dev_mode', True),
            'max_age_sec': cache_cfg.get('static_max_age', 86400) if not cache_cfg.get('dev_mode', True) else 0
        }
    }


@app.post("/api/cache/invalidate")
async def invalidate_cache(target: str = "constellation"):
    """Invalidate cached data.

    Args:
        target: Which cache to invalidate ('constellation', 'embedding', 'all')
    """
    if target == "constellation":
        _constellation_cache.invalidate()
        return {'ok': True, 'message': 'Constellation cache invalidated'}
    elif target == "embedding":
        try:
            from model_client import clear_embedding_cache
            clear_embedding_cache()
            return {'ok': True, 'message': 'Embedding cache cleared'}
        except ImportError:
            return {'ok': False, 'error': 'Embedding cache not available'}
    elif target == "all":
        _constellation_cache.invalidate()
        try:
            from model_client import clear_embedding_cache
            clear_embedding_cache()
        except ImportError:
            pass
        return {'ok': True, 'message': 'All caches invalidated'}
    else:
        return {'ok': False, 'error': f'Unknown cache target: {target}'}


# =============================================================================
# Routes: Status
# =============================================================================

@app.get("/")
async def root():
    """Serve the dashboard"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    """Get mind state and resident status"""
    r = get_resident()
    status = r.status

    return {
        'ok': True,
        'mind': {
            'Df': status['mind_Df'],
            'hash': status['mind_hash'],
            'distance_from_start': status['distance_from_start'],
        },
        'interactions': status['interaction_count'],
        'db_stats': status['db_stats'],
        'reasoner_stats': status['reasoner_stats'],
        'version': r.VERSION
    }


@app.get("/api/evolution")
async def get_evolution():
    """Get mind evolution history"""
    r = get_resident()
    evolution = r.mind_evolution

    return {
        'ok': True,
        'current_Df': evolution['current_Df'],
        'distance_from_start': evolution['distance_from_start'],
        'interaction_count': evolution['interaction_count'],
        'Df_history': evolution.get('Df_history', [])[-50:],  # Last 50
        'distance_history': evolution.get('distance_history', [])[-50:]
    }


@app.get("/api/constellation")
async def get_constellation(max_nodes: int = 500, bypass_cache: bool = False):
    """Get document constellation graph showing GOD TIER papers and their headings.

    Structure:
    - Root "FERAL" node
    - Paper nodes (folders) for each paper_id
    - Heading nodes (pages) for each section
    - Similarity edges between chunks

    Caching:
    - Results cached for constellation_ttl_sec (default 5 min)
    - Use bypass_cache=true to force refresh
    """
    import sqlite3
    import numpy as np
    import json

    # Check cache first (unless bypassed)
    cache_key = f"constellation:{max_nodes}"
    cache_cfg = get_cache_config()
    ttl = cache_cfg.get('constellation_ttl_sec', 300)

    if not bypass_cache:
        cached = _constellation_cache.get(cache_key, ttl=ttl)
        if cached:
            cached['from_cache'] = True
            return cached

    db_path = FERAL_RESIDENT_PATH / "data" / "db" / "feral_eternal.db"

    nodes = []
    edges = []
    vector_embeddings = {}
    paper_nodes = set()  # Track paper folders

    try:
        # Root node
        nodes.append({"id": "root", "label": "FERAL CONSTELLATION", "group": "folder"})

        with sqlite3.connect(str(db_path)) as conn:
            # Get paper chunks from receipts (paper_load operation)
            # Sample across ALL papers by using ROW_NUMBER to get N chunks per paper
            chunks_per_paper = max(3, max_nodes // 100)  # At least 3 chunks per paper
            cursor = conn.execute("""
                WITH ranked_chunks AS (
                    SELECT r.receipt_id, r.output_hash, r.metadata, v.vec_blob, r.created_at,
                           json_extract(r.metadata, '$.paper_id') as paper_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY json_extract(r.metadata, '$.paper_id')
                               ORDER BY r.created_at
                           ) as rn
                    FROM receipts r
                    LEFT JOIN vectors v ON v.vec_sha256 LIKE r.output_hash || '%'
                    WHERE r.operation = 'paper_load'
                )
                SELECT receipt_id, output_hash, metadata, vec_blob
                FROM ranked_chunks
                WHERE rn <= ?
                ORDER BY paper_id, created_at
            """, (chunks_per_paper,))

            rows = cursor.fetchall()

            # Track heading hierarchy per paper: paper_id -> {level: last_node_id}
            heading_parents = {}

            for receipt_id, output_hash, metadata_json, vec_blob in rows:
                if not metadata_json:
                    continue

                meta = json.loads(metadata_json)
                paper_id = meta.get('paper_id', 'unknown')
                heading = meta.get('heading', '').strip()

                # Create paper folder node if not exists
                paper_node_id = f"paper:{paper_id}"
                if paper_node_id not in paper_nodes:
                    paper_nodes.add(paper_node_id)
                    nodes.append({
                        "id": paper_node_id,
                        "label": paper_id,
                        "group": "folder",
                        "paper_id": paper_id
                    })
                    edges.append({"from": "root", "to": paper_node_id, "type": "hierarchy"})
                    # Initialize heading hierarchy for this paper
                    heading_parents[paper_id] = {0: paper_node_id}  # Level 0 = paper itself

                # Parse heading level from # count
                heading_level = 0
                if heading.startswith('#'):
                    stripped = heading.lstrip('#')
                    heading_level = len(heading) - len(stripped)

                # Create heading node (chunk)
                # Clean heading for label
                label = heading.lstrip('#').strip()[:40]
                if not label:
                    label = "content"

                chunk_node_id = f"chunk:{receipt_id}"
                nodes.append({
                    "id": chunk_node_id,
                    "label": label,
                    "group": "page",
                    "paper_id": paper_id,
                    "heading": heading,
                    "level": heading_level
                })

                # Find appropriate parent based on heading level
                # A ### (level 3) should connect to the most recent ## (level 2)
                # If no ## exists yet, connect to the paper node
                parent_node_id = paper_node_id
                if paper_id in heading_parents:
                    # Look for closest ancestor (any level < current)
                    for check_level in range(heading_level - 1, -1, -1):
                        if check_level in heading_parents[paper_id]:
                            parent_node_id = heading_parents[paper_id][check_level]
                            break

                # Connect chunk to its hierarchical parent
                edges.append({"from": parent_node_id, "to": chunk_node_id, "type": "hierarchy"})

                # Update hierarchy tracker: this node becomes the parent for deeper levels
                if paper_id in heading_parents:
                    heading_parents[paper_id][heading_level] = chunk_node_id
                    # Clear any deeper levels (new section resets subsections)
                    for deeper in list(heading_parents[paper_id].keys()):
                        if deeper > heading_level:
                            del heading_parents[paper_id][deeper]

                # Store vector for similarity computation
                if vec_blob:
                    try:
                        embedding = np.frombuffer(vec_blob, dtype=np.float32)
                        if len(embedding) > 0:
                            vector_embeddings[chunk_node_id] = embedding
                    except Exception:
                        pass

        # Compute similarity edges between chunks
        if len(vector_embeddings) > 1:
            similarity_edges = compute_similarity_edges(vector_embeddings, threshold=0.0, max_edges=1000)
            edges.extend(similarity_edges)

        # Load resident-decided links from database
        resident_link_edges = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("""
                SELECT source_hash, target_hash, link_type, strength
                FROM resident_links
                ORDER BY created_at DESC
                LIMIT 500
            """)

            # Build hash -> node_id mapping
            hash_to_node = {}
            for node in nodes:
                if node.get('id', '').startswith('chunk:'):
                    # Get receipt_id from node id
                    receipt_id = node['id'].replace('chunk:', '')
                    hash_to_node[receipt_id] = node['id']

            for source_hash, target_hash, link_type, strength in cursor.fetchall():
                # Map hashes to node IDs (prefix match)
                source_node = None
                target_node = None
                for h, nid in hash_to_node.items():
                    if source_hash.startswith(h) or h.startswith(source_hash[:8]):
                        source_node = nid
                    if target_hash.startswith(h) or h.startswith(target_hash[:8]):
                        target_node = nid

                if source_node and target_node and source_node != target_node:
                    resident_link_edges.append({
                        "from": source_node,
                        "to": target_node,
                        "type": link_type,  # mind_projected, co_retrieval, entanglement
                        "weight": float(strength) if strength else 1.0
                    })

        edges.extend(resident_link_edges)

        # Edge type color legend (for frontend)
        edge_colors = {
            'hierarchy': '#008f11',      # Dark green - structural
            'similarity': '#64ffff',      # Cyan - cosine similarity
            'mind_projected': '#ff6b6b',  # Coral red - resident's perspective
            'co_retrieval': '#ffd93d',    # Gold - retrieved together
            'entanglement': '#c77dff'     # Purple - quantum bound
        }

        result = {
            'ok': True,
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges),
            'similarity_edge_count': len([e for e in edges if e.get('type') == 'similarity']),
            'resident_link_count': len(resident_link_edges),
            'edge_colors': edge_colors,
            'from_cache': False
        }

        # Store in cache
        _constellation_cache.set(cache_key, result)

        return result

    except Exception as e:
        import traceback
        return {
            'ok': False,
            'error': str(e),
            'trace': traceback.format_exc(),
            'nodes': [{"id": "error", "label": "Error loading data", "group": "folder"}],
            'edges': []
        }


def compute_similarity_edges(embeddings: Dict[str, Any], threshold: float = 0.7, max_edges: int = 100) -> List[Dict]:
    """Compute cosine similarity edges between chunks - OPTIMIZED with matrix ops"""
    import numpy as np

    node_ids = list(embeddings.keys())
    n = len(node_ids)

    if n < 2:
        return []

    # Stack all embeddings into matrix
    embedding_matrix = np.vstack([embeddings[nid] for nid in node_ids])

    # Normalize for cosine similarity
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embedding_matrix / norms

    # OPTIMIZED: Use matrix multiplication for all similarities at once
    # This is O(n^2) in memory but much faster than nested loops
    similarity_matrix = normalized @ normalized.T

    # Get upper triangle indices (avoid duplicates and self-similarity)
    i_indices, j_indices = np.triu_indices(n, k=1)

    # Get similarity values for upper triangle
    similarities = similarity_matrix[i_indices, j_indices]

    # Filter by threshold BEFORE sorting (much faster)
    mask = similarities >= threshold
    filtered_i = i_indices[mask]
    filtered_j = j_indices[mask]
    filtered_sim = similarities[mask]

    # If too many edges, take top ones
    if len(filtered_sim) > max_edges:
        # Get indices of top max_edges similarities
        top_indices = np.argpartition(filtered_sim, -max_edges)[-max_edges:]
        filtered_i = filtered_i[top_indices]
        filtered_j = filtered_j[top_indices]
        filtered_sim = filtered_sim[top_indices]

    # Build edge list
    similarity_edges = [
        {
            "from": node_ids[i],
            "to": node_ids[j],
            "type": "similarity",
            "weight": float(sim)
        }
        for i, j, sim in zip(filtered_i, filtered_j, filtered_sim)
    ]

    # Sort by weight descending
    similarity_edges.sort(key=lambda x: x["weight"], reverse=True)
    return similarity_edges


@app.get("/api/activity")
async def get_activity(limit: int = 50):
    """Get recent activity log"""
    d = get_daemon()

    activities = list(d.activity_log)[-limit:]

    return {
        'ok': True,
        'activities': [
            {
                'timestamp': a.timestamp,
                'action': a.action,
                'summary': a.summary,
                'details': a.details
            }
            for a in activities
        ]
    }


@app.post("/api/resident/link")
async def create_resident_link(
    source_hash: str,
    target_hash: str,
    link_type: str,
    strength: float = 1.0
):
    """Create a resident-decided link between two chunks.

    link_type must be one of:
    - mind_projected: Similarity from resident's perspective
    - co_retrieval: Chunks retrieved together
    - entanglement: Chunks bound during reasoning
    """
    import sqlite3

    if link_type not in ['mind_projected', 'co_retrieval', 'entanglement']:
        return {'ok': False, 'error': f'Invalid link_type: {link_type}'}

    db_path = FERAL_RESIDENT_PATH / "data" / "db" / "feral_eternal.db"

    try:
        # Get mind hash from resident
        r = get_resident()
        mind_hash = r.mind_hash

        with sqlite3.connect(str(db_path)) as conn:
            import uuid
            from datetime import datetime, timezone
            import json

            link_id = str(uuid.uuid4())[:8]
            now = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """
                INSERT INTO resident_links
                (link_id, source_hash, target_hash, link_type, strength, mind_hash, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (link_id, source_hash, target_hash, link_type, strength,
                 mind_hash, json.dumps({}), now)
            )
            conn.commit()

        return {
            'ok': True,
            'link_id': link_id,
            'link_type': link_type,
            'strength': strength
        }

    except Exception as e:
        return {'ok': False, 'error': str(e)}


@app.post("/api/resident/compute_mind_projected")
async def compute_mind_projected_similarity(max_links: int = 100):
    """Compute mind-projected similarity and store as resident links.

    Projects all chunks through the current mind state and finds
    pairs that are similar from the resident's perspective.
    """
    import sqlite3
    import numpy as np
    import json

    db_path = FERAL_RESIDENT_PATH / "data" / "db" / "feral_eternal.db"

    try:
        r = get_resident()
        if not r.mind_state:
            return {'ok': False, 'error': 'Resident has no mind state yet'}

        mind_vector = r.mind_state.vector
        mind_hash = r.mind_hash

        with sqlite3.connect(str(db_path)) as conn:
            # Get paper chunks with vectors
            cursor = conn.execute("""
                SELECT r.output_hash, v.vec_blob
                FROM receipts r
                JOIN vectors v ON v.vec_sha256 LIKE r.output_hash || '%'
                WHERE r.operation = 'paper_load'
                LIMIT 500
            """)

            chunks = []
            for output_hash, vec_blob in cursor.fetchall():
                if vec_blob:
                    vec = np.frombuffer(vec_blob, dtype=np.float32)
                    chunks.append((output_hash, vec))

            if len(chunks) < 2:
                return {'ok': False, 'error': 'Not enough chunks'}

            # Project each chunk through mind state
            # projected = chunk - (chunk · mind) * mind (orthogonal component)
            # Or simpler: weighted by relevance to mind
            mind_norm = mind_vector / np.linalg.norm(mind_vector)

            projected = []
            for hash_, vec in chunks:
                # Weight by E (relevance to mind)
                E = np.dot(vec / np.linalg.norm(vec), mind_norm)
                # Project: blend original with mind-weighted version
                weighted = vec * (0.5 + 0.5 * E)  # Boost mind-relevant chunks
                projected.append((hash_, weighted, E))

            # Find pairs with high mind-projected similarity
            links_created = 0
            for i in range(len(projected)):
                for j in range(i + 1, len(projected)):
                    if links_created >= max_links:
                        break

                    h1, v1, E1 = projected[i]
                    h2, v2, E2 = projected[j]

                    # Only link if both are relevant to mind
                    if E1 < 0.3 or E2 < 0.3:
                        continue

                    # Compute projected similarity
                    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

                    if sim > 0.7:  # High projected similarity
                        import uuid
                        from datetime import datetime, timezone

                        link_id = str(uuid.uuid4())[:8]
                        now = datetime.now(timezone.utc).isoformat()

                        conn.execute(
                            """
                            INSERT INTO resident_links
                            (link_id, source_hash, target_hash, link_type, strength, mind_hash, context, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (link_id, h1[:16], h2[:16], 'mind_projected', float(sim),
                             mind_hash, json.dumps({'E1': float(E1), 'E2': float(E2)}), now)
                        )
                        links_created += 1

                if links_created >= max_links:
                    break

            conn.commit()

        return {
            'ok': True,
            'links_created': links_created,
            'mind_hash': mind_hash
        }

    except Exception as e:
        import traceback
        return {'ok': False, 'error': str(e), 'trace': traceback.format_exc()}


@app.get("/api/debug/tui")
async def get_tui_debug():
    """Debug: get TUI state and WebSocket status"""
    return {
        'ok': True,
        'tui_state_exists': tui_state is not None,
        'instance_id': tui_state.instance_id if tui_state else None,
        'event_count': tui_state.event_count if tui_state else 0,
        'activity_count': len(tui_state.activity) if tui_state else 0,
        'activity': list(tui_state.activity) if tui_state else [],
        'ws_connections': len(manager.active_connections),
        'broadcast_loop_set': broadcast_loop is not None,
        'broadcast_loop_running': broadcast_loop.is_running() if broadcast_loop else False
    }


# =============================================================================
# Routes: Chat
# =============================================================================

@app.post("/api/think", response_model=ThinkResponse)
async def think(request: ThinkRequest):
    """Send a query to the resident"""
    r = get_resident()

    result = r.think(request.query)

    # Broadcast the thought
    await manager.broadcast({
        'type': 'thought',
        'data': {
            'query': request.query,
            'response': result.response,
            'E': result.E_resonance,
            'gate_open': result.gate_open
        }
    })

    # Broadcast mind update
    await manager.broadcast({
        'type': 'mind_update',
        'data': {
            'Df': result.mind_Df,
            'distance': result.distance_from_start
        }
    })

    return ThinkResponse(
        response=result.response,
        E_resonance=result.E_resonance,
        E_compression=result.E_compression,
        gate_open=result.gate_open,
        mind_Df=result.mind_Df,
        distance_from_start=result.distance_from_start
    )


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent interaction history"""
    r = get_resident()
    interactions = r.get_recent_interactions(limit=limit)

    return {
        'ok': True,
        'interactions': [
            {
                'input': i.get('input', ''),
                'output': i.get('output', ''),
                'E': i.get('E_resonance', 0),
                'Df': i.get('mind_Df', 0),
                'timestamp': i.get('created_at', '')
            }
            for i in interactions
        ]
    }


# =============================================================================
# Routes: Daemon Control
# =============================================================================

@app.get("/api/daemon/status")
async def get_daemon_status():
    """Get daemon status"""
    d = get_daemon()
    return {
        'ok': True,
        **d.status
    }


@app.post("/api/daemon/start")
async def start_daemon():
    """Start the daemon"""
    d = get_daemon()

    if d.running:
        return {'ok': True, 'message': 'Daemon already running'}

    await d.start()

    return {'ok': True, 'message': 'Daemon started'}


@app.post("/api/daemon/stop")
async def stop_daemon():
    """Stop the daemon"""
    d = get_daemon()

    if not d.running:
        return {'ok': True, 'message': 'Daemon not running'}

    await d.stop()

    return {'ok': True, 'message': 'Daemon stopped'}


@app.post("/api/daemon/config")
async def configure_daemon(request: BehaviorConfigRequest):
    """Configure a daemon behavior and persist to config.json"""
    d = get_daemon()

    try:
        # Update in-memory state
        d.configure_behavior(
            request.behavior,
            enabled=request.enabled,
            interval=request.interval
        )

        # Persist to config.json so settings survive refresh
        config_path = FERAL_RESIDENT_PATH / "config.json"

        # Read current config
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}

        # Ensure behaviors section exists
        if 'behaviors' not in config:
            config['behaviors'] = {}

        # Ensure this behavior exists
        if request.behavior not in config['behaviors']:
            config['behaviors'][request.behavior] = {}

        # Update values
        if request.enabled is not None:
            config['behaviors'][request.behavior]['enabled'] = request.enabled
        if request.interval is not None:
            config['behaviors'][request.behavior]['interval'] = max(10, request.interval)

        # Write back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {'ok': True, 'message': f'Configured {request.behavior}', 'persisted': True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Routes: Particle Smasher (Burst Mode)
# =============================================================================

@app.post("/api/smasher/start")
async def start_smasher(request: SmasherStartRequest):
    """Start the Particle Smasher - rapid paper chunk processing"""
    d = get_daemon()

    if d.smasher_config.enabled:
        return {'ok': False, 'message': 'Smasher already running'}

    await d.start_smasher(
        delay_ms=request.delay_ms,
        batch_size=request.batch_size,
        batch_pause_ms=request.batch_pause_ms,
        max_chunks=request.max_chunks
    )

    return {
        'ok': True,
        'message': 'Particle Smasher ENGAGED',
        'config': {
            'delay_ms': request.delay_ms,
            'batch_size': request.batch_size,
            'batch_pause_ms': request.batch_pause_ms,
            'max_chunks': request.max_chunks
        }
    }


@app.post("/api/smasher/stop")
async def stop_smasher():
    """Stop the Particle Smasher immediately (force mode)"""
    d = get_daemon()

    if not d.smasher_config.enabled and d._smasher_task is None:
        return {'ok': True, 'message': 'Smasher not running'}

    # Clear pending WebSocket batch to prevent queued events from broadcasting
    if hasattr(d, '_clear_smash_batch'):
        d._clear_smash_batch()

    # Force immediate stop - don't wait for current chunk
    await d.stop_smasher(force=True)

    return {
        'ok': True,
        'message': 'Particle Smasher DISENGAGED',
        'stats': {
            'chunks_processed': d.smasher_stats.chunks_processed,
            'chunks_absorbed': d.smasher_stats.chunks_absorbed,
            'chunks_rejected': d.smasher_stats.chunks_rejected,
            'rate': d.smasher_stats.chunks_per_second
        }
    }


@app.get("/api/smasher/status")
async def get_smasher_status():
    """Get Particle Smasher status and stats"""
    d = get_daemon()

    return {
        'ok': True,
        'active': d.smasher_config.enabled,
        'config': {
            'delay_ms': d.smasher_config.delay_ms,
            'batch_size': d.smasher_config.batch_size,
            'batch_pause_ms': d.smasher_config.batch_pause_ms,
            'max_chunks': d.smasher_config.max_chunks
        },
        'stats': {
            'chunks_processed': d.smasher_stats.chunks_processed,
            'chunks_absorbed': d.smasher_stats.chunks_absorbed,
            'chunks_rejected': d.smasher_stats.chunks_rejected,
            'chunks_per_second': d.smasher_stats.chunks_per_second,
            'elapsed_seconds': d.smasher_stats.elapsed_seconds
        }
    }


class SmasherConfigUpdate(BaseModel):
    delay_ms: Optional[int] = None
    batch_size: Optional[int] = None
    batch_pause_ms: Optional[int] = None


@app.post("/api/smasher/config")
async def update_smasher_config(request: SmasherConfigUpdate):
    """Update smasher config LIVE - writes to config.json"""
    config_path = FERAL_RESIDENT_PATH / "config.json"

    # Read current config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    # Ensure smasher section exists
    if 'smasher' not in config:
        config['smasher'] = {}

    # Update values
    if request.delay_ms is not None:
        config['smasher']['delay_ms'] = max(10, request.delay_ms)
    if request.batch_size is not None:
        config['smasher']['batch_size'] = max(1, request.batch_size)
    if request.batch_pause_ms is not None:
        config['smasher']['batch_pause_ms'] = max(0, request.batch_pause_ms)

    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return {
        'ok': True,
        'config': config['smasher']
    }


# =============================================================================
# Routes: Memory Pruning
# =============================================================================

class PruneRequest(BaseModel):
    threshold: Optional[float] = None  # Default: 1/(2π)
    dry_run: bool = True  # Default to preview mode


@app.post("/api/prune")
async def prune_memories(request: PruneRequest):
    """
    Prune low-E memories below threshold.

    Default threshold is 1/(2π) ≈ 0.159 (Q46 Law 3 maximum).
    Set dry_run=false to actually delete.

    Usage:
        # Preview what would be pruned
        curl -X POST http://localhost:8420/api/prune -H "Content-Type: application/json" -d '{"dry_run": true}'

        # Actually prune
        curl -X POST http://localhost:8420/api/prune -H "Content-Type: application/json" -d '{"dry_run": false}'
    """
    r = get_resident()

    result = r.prune_low_E_memories(
        threshold=request.threshold,
        dry_run=request.dry_run
    )

    return {
        'ok': True,
        **result
    }


@app.get("/api/memory/stats")
async def get_memory_stats():
    """Get memory statistics including count and E distribution."""
    import math
    r = get_resident()
    mind_state = r.store.get_mind_state()

    # Get all interactions
    interactions = r.store.db.get_thread_interactions(r.thread_id, limit=10000)

    threshold = 1.0 / (2.0 * math.pi)
    above_threshold = 0
    below_threshold = 0
    E_values = []

    for interaction in interactions:
        if not interaction.input_text:
            continue
        try:
            input_state = r.store.embed(interaction.input_text, store=False)
            if mind_state is not None:
                E = input_state.E_with(mind_state)
                E_values.append(E)
                if E >= threshold:
                    above_threshold += 1
                else:
                    below_threshold += 1
        except:
            continue

    return {
        'ok': True,
        'total_memories': len(interactions),
        'above_threshold': above_threshold,
        'below_threshold': below_threshold,
        'threshold': threshold,
        'threshold_name': '1/(2π)',
        'E_mean': sum(E_values) / len(E_values) if E_values else 0,
        'E_min': min(E_values) if E_values else 0,
        'E_max': max(E_values) if E_values else 0
    }


# =============================================================================
# Routes: Live Config (writes to config.json)
# =============================================================================

@app.get("/api/config")
async def get_config():
    """Get current config.json"""
    config_path = FERAL_RESIDENT_PATH / "config.json"

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return {'ok': True, 'config': config, 'path': str(config_path)}
    except FileNotFoundError:
        return {'ok': False, 'error': 'config.json not found', 'path': str(config_path)}
    except json.JSONDecodeError as e:
        return {'ok': False, 'error': f'Invalid JSON: {e}', 'path': str(config_path)}


@app.post("/api/config")
async def update_config(updates: Dict[str, Any]):
    """Update config.json - daemon reads on next cycle"""
    config_path = FERAL_RESIDENT_PATH / "config.json"

    # Read current config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    # Deep merge updates
    def deep_merge(base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(config, updates)

    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return {'ok': True, 'config': config}


# =============================================================================
# WebSocket
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)

    # Send initial state
    r = get_resident()
    d = get_daemon()

    await websocket.send_json({
        'type': 'init',
        'data': {
            'mind': {
                'Df': r.status['mind_Df'],
                'distance': r.status['distance_from_start']
            },
            'daemon': d.status
        }
    })

    try:
        while True:
            # Keep connection alive, receive any messages
            data = await websocket.receive_text()

            # Handle ping/pong
            if data == 'ping':
                await websocket.send_text('pong')

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Main
# =============================================================================

class TUIState:
    """Holds TUI state for the static display"""
    _instance_counter = 0

    def __init__(self):
        TUIState._instance_counter += 1
        self.instance_id = TUIState._instance_counter
        self.activity: list[str] = []
        self.server_status = "Initializing..."
        self.max_activity = 20
        self.event_count = 0  # Track total events received

    def add_activity(self, msg: str):
        self.event_count += 1
        self.activity.append(msg)
        if len(self.activity) > self.max_activity:
            self.activity.pop(0)

    def set_server_status(self, status: str):
        self.server_status = status


def make_tui(state: TUIState):
    """Create full-screen static TUI layout - all green on black"""
    # Banner (ASCII-safe)
    banner = Text()
    banner.append("FERAL\n", style="bold green")
    banner.append("Geometric Cognition Engine\n", style="green")
    banner.append("v2.1.0-q46", style="dim green")

    banner_panel = Panel(
        banner,
        border_style="green",
        box=box.SQUARE,
        padding=(0, 2)
    )

    # Server Info table
    info_table = Table(box=None, show_header=False, padding=(0, 1))
    info_table.add_column("Key", style="dim green")
    info_table.add_column("Value", style="green")
    info_table.add_row("Dashboard", "http://localhost:8420")
    info_table.add_row("Static Dir", str(STATIC_DIR.relative_to(REPO_ROOT)))
    info_table.add_row("Database", "FERAL_RESIDENT/data/db/feral_eternal.db")
    info_table.add_row("Hot Reload", "Enabled")
    info_table.add_row("Status", state.server_status)
    info_table.add_row("Events", str(state.event_count))
    info_table.add_row("Instance", str(state.instance_id))
    info_table.add_row("Stop Smasher", "S key")
    info_table.add_row("Exit", "Ctrl+C")

    info_panel = Panel(
        info_table,
        title="[green]Server Info[/]",
        border_style="green",
        box=box.SQUARE
    )

    # Activity panel (packets being smashed)
    activity_text = Text()
    for line in state.activity:
        activity_text.append(f"  {line}\n", style="green")
    if not state.activity:
        activity_text.append("  Waiting for activity...\n", style="dim green")

    activity_panel = Panel(
        activity_text,
        title="[green]Activity[/]",
        border_style="green",
        box=box.SQUARE
    )

    # Build two-column layout
    layout = Layout()

    # Main split: left column, right column
    layout.split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2)
    )

    # Left column: banner on top, info below
    layout["left"].split_column(
        Layout(name="banner"),
        Layout(name="info")
    )

    # Right column: activity (full height)
    layout["right"].update(activity_panel)

    # Update left sections
    layout["left"]["banner"].update(banner_panel)
    layout["left"]["info"].update(info_panel)

    return layout


def main():
    """Run the server with full-screen static TUI"""
    global tui_state
    import io
    import os
    import signal
    import time

    if not RICH_AVAILABLE:
        uvicorn.run(app, host="0.0.0.0", port=8420, log_level="error")
        return

    # TUI state (global so activity callbacks can update it)
    tui = TUIState()
    tui_state = tui

    # Save ORIGINAL stdout/stderr before ANY redirects
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout/stderr - suppress all library output
    class NullIO(io.StringIO):
        def write(self, s):
            return len(s)
        def flush(self):
            pass

    # Redirect EVERYTHING
    sys.stdout = NullIO()
    sys.stderr = NullIO()
    sys.__stdout__ = NullIO()
    sys.__stderr__ = NullIO()

    # Create console with ORIGINAL stdout so TUI still renders
    console = Console(file=original_stdout, force_terminal=True)

    # Clear screen immediately
    os.system('cls' if os.name == 'nt' else 'clear')

    with Live(make_tui(tui), console=console, refresh_per_second=10, screen=True) as live:
        try:
            # Force render FIRST
            live.refresh()
            time.sleep(0.1)

            # Check if model server is running (fast boot!)
            tui.set_server_status("Initializing...")
            try:
                import requests
                resp = requests.get("http://localhost:8421/health", timeout=1)
                model_server_running = resp.status_code == 200
            except:
                model_server_running = False

            if model_server_running:
                tui.add_activity("[init] Model server detected (fast boot!)")
            else:
                tui.add_activity("[init] Loading transformers locally...")
            live.update(make_tui(tui))
            live.refresh()

            get_resident()

            tui.add_activity("[init] Creating FeralDaemon...")
            live.update(make_tui(tui))
            live.refresh()
            get_daemon()

            tui.add_activity("[init] Starting uvicorn...")
            live.update(make_tui(tui))
            live.refresh()

            # Start server
            server_thread = threading.Thread(
                target=uvicorn.run,
                kwargs={"app": app, "host": "0.0.0.0", "port": 8420, "log_level": "critical"},
                daemon=True
            )
            server_thread.start()

            # Clear init messages, show ready state
            tui.activity = []
            tui.set_server_status("Running")
            live.update(make_tui(tui))

            # Check for keyboard input (Windows-specific for now)
            try:
                import msvcrt
                has_msvcrt = True
            except ImportError:
                has_msvcrt = False

            # Keep TUI alive - refresh every 0.5s to show activity updates
            while True:
                # Check for 's' key to stop smasher
                if has_msvcrt and msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if key == 's':
                        # Force stop smasher
                        d = get_daemon()
                        if d.smasher_config.enabled or d._smasher_task:
                            tui.add_activity("[TUI] EMERGENCY STOP triggered (S key)")
                            d.smasher_config.enabled = False
                            d._force_stop = True
                            if d._smasher_task:
                                d._smasher_task.cancel()
                                d._smasher_task = None

                time.sleep(0.5)
                live.update(make_tui(tui))
                live.refresh()

        except KeyboardInterrupt:
            tui.set_server_status("Shutting down...")
            live.update(make_tui(tui))
            time.sleep(0.5)
        finally:
            # Restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sys.__stdout__ = original_stdout
            sys.__stderr__ = original_stderr


if __name__ == "__main__":
    main()

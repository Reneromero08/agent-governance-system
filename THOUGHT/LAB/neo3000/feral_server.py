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

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import threading

# Add paths - FERAL_RESIDENT is the canonical source
REPO_ROOT = Path(__file__).resolve().parents[3]
NEO3000_DIR = Path(__file__).resolve().parent
FERAL_RESIDENT_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"
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

from feral_daemon import FeralDaemon, ActivityEvent
from vector_brain import VectorResident

# Optional: CORTEX query module for constellation visualization
try:
    import query as cortex_query
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False
    print("WARNING: CORTEX query module not available. Constellation will be disabled.")


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
        print(f"[HOT RELOAD] Watching {self.watch_dir} for changes...")

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
                                print(f"[HOT RELOAD] Changed: {path.name}")
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
daemon: Optional[FeralDaemon] = None
resident: Optional[VectorResident] = None


def get_resident() -> VectorResident:
    """Get or create the resident instance"""
    global resident
    if resident is None:
        db_path = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT" / "data" / "feral_eternal.db"
        db_path.parent.mkdir(exist_ok=True)
        resident = VectorResident(thread_id="eternal", db_path=str(db_path))
    return resident


def get_daemon() -> FeralDaemon:
    """Get or create the daemon instance"""
    global daemon
    if daemon is None:
        daemon = FeralDaemon(resident=get_resident(), thread_id="eternal")

        # Add WebSocket callback
        def on_activity(event: ActivityEvent):
            # Broadcast activity to activity log
            asyncio.create_task(manager.broadcast({
                'type': 'activity',
                'data': {
                    'timestamp': event.timestamp,
                    'action': event.action,
                    'summary': event.summary,
                    'details': event.details
                }
            }))

            # NEW: Broadcast constellation events for dynamic 3D visualization
            if event.details.get('chunk_id'):
                chunk_id = event.details['chunk_id']
                paper = event.details.get('paper', 'papers')
                heading = event.details.get('heading', '')[:25] if event.details.get('heading') else ''
                is_new = event.details.get('is_new_node', False)

                # Use full IDs if provided (smasher sends these), otherwise construct
                node_id = event.details.get('full_node_id') or f"chunk:{paper}:{chunk_id}"
                # Use SEMANTIC similar_to for positioning (not sequential source)
                similar_to = event.details.get('similar_to')
                similar_E = event.details.get('similar_E', 0)

                # Particle Smasher mode: lightweight flash events (no camera follow)
                if event.action == 'smash':
                    asyncio.create_task(manager.broadcast({
                        'type': 'smash_hit',
                        'data': {
                            'node_id': node_id,
                            'E': event.details.get('E', 0),
                            'gate_open': event.details.get('gate_open', False),
                            'is_new_node': is_new,
                            'similar_to': similar_to,  # Semantic anchor for positioning
                            'similar_E': similar_E,     # How similar (for edge weight)
                            'rate': event.details.get('rate', 0)
                        }
                    }))
                elif is_new:
                    # New node discovered - spawn animation
                    asyncio.create_task(manager.broadcast({
                        'type': 'node_discovered',
                        'data': {
                            'node_id': node_id,
                            'label': heading or f"chunk-{chunk_id}",
                            'similar_to': similar_to,
                            'activity_type': event.action,
                            'paper': paper
                        }
                    }))
                else:
                    # Existing node activated - highlight
                    asyncio.create_task(manager.broadcast({
                        'type': 'node_activated',
                        'data': {
                            'node_id': node_id,
                            'similar_to': similar_to,
                            'activity_type': event.action
                        }
                    }))

        daemon.add_callback(on_activity)

    return daemon


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global hot_reload_watcher

    print(f"[FERAL] Starting server...")
    print(f"[FERAL] Dashboard: http://localhost:8420")

    # Initialize resident and daemon
    get_resident()
    get_daemon()

    # Start hot reload watcher for live editing
    static_dir = Path(__file__).parent / "static"
    hot_reload_watcher = HotReloadWatcher(static_dir, manager)
    hot_reload_watcher.start(asyncio.get_event_loop())

    yield

    # Shutdown
    print(f"[FERAL] Shutting down...")
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

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
async def get_constellation(include_similarity: bool = True, similarity_threshold: float = 0.7):
    """Get document constellation graph from cassette network with optional cosine similarity edges"""
    import sqlite3
    import numpy as np

    CASSETTES_DIR = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"
    nodes = []
    edges = []
    chunk_embeddings = {}  # Store embeddings for similarity calculation

    try:
        # Root node
        nodes.append({"id": "root", "label": "CASSETTES", "group": "folder"})

        # Scan cassette databases
        for db_file in CASSETTES_DIR.glob("*.db"):
            cassette_name = db_file.stem
            cassette_id = f"cassette:{cassette_name}"

            # Cassette node
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                    chunk_count = cursor.fetchone()[0]

                    nodes.append({
                        "id": cassette_id,
                        "label": f"{cassette_name} ({chunk_count})",
                        "group": "folder",
                        "path": str(db_file)
                    })
                    edges.append({"from": "root", "to": cassette_id, "type": "hierarchy"})

                    # Sample chunks from this cassette (max 50 per cassette for performance)
                    # Track which chunk_ids we actually load so similarity only uses those
                    loaded_chunk_ids = []
                    cursor = conn.execute("""
                        SELECT c.chunk_id, f.path, c.header_text
                        FROM chunks c
                        JOIN files f ON c.file_id = f.file_id
                        LIMIT 50
                    """)
                    for row in cursor.fetchall():
                        chunk_id, source_path, header_text = row
                        loaded_chunk_ids.append(chunk_id)
                        node_id = f"chunk:{cassette_name}:{chunk_id}"
                        # Create short label from header or filename
                        if header_text:
                            label = header_text[:25]
                        elif source_path:
                            label = source_path.split('/')[-1].split('\\')[-1][:25]
                        else:
                            label = f"chunk-{chunk_id}"
                        nodes.append({
                            "id": node_id,
                            "label": label,
                            "group": "page",
                            "path": source_path or ""
                        })
                        edges.append({"from": cassette_id, "to": node_id, "type": "hierarchy"})

                    # Load embeddings ONLY for chunks we actually added to the graph
                    if include_similarity and loaded_chunk_ids:
                        try:
                            placeholders = ','.join('?' * len(loaded_chunk_ids))
                            cursor = conn.execute(f"""
                                SELECT c.chunk_id, g.vector_blob
                                FROM chunks c
                                JOIN geometric_index g ON c.chunk_hash = g.doc_id
                                WHERE c.chunk_id IN ({placeholders})
                            """, loaded_chunk_ids)
                            for row in cursor.fetchall():
                                chunk_id, vector_blob = row
                                if vector_blob:
                                    node_id = f"chunk:{cassette_name}:{chunk_id}"
                                    # geometric_index stores 384-dim vectors (1536 bytes / 4 = 384 floats)
                                    embedding = np.frombuffer(vector_blob, dtype=np.float32)
                                    chunk_embeddings[node_id] = embedding
                        except Exception as e:
                            # geometric_index table might not exist or join failed
                            print(f"[CONSTELLATION] Embedding load error for {cassette_name}: {e}")
                            pass

            except Exception as e:
                # Skip broken databases
                continue

        # Compute cosine similarity edges
        if include_similarity and len(chunk_embeddings) > 1:
            similarity_edges = compute_similarity_edges(chunk_embeddings, similarity_threshold)
            edges.extend(similarity_edges)

        return {'ok': True, 'nodes': nodes, 'edges': edges}

    except Exception as e:
        return {'ok': False, 'error': str(e), 'nodes': [], 'edges': []}


def compute_similarity_edges(embeddings: Dict[str, Any], threshold: float = 0.7, max_edges: int = 100) -> List[Dict]:
    """Compute cosine similarity edges between chunks"""
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

    # Compute similarity matrix (upper triangle only to avoid duplicates)
    similarity_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(normalized[i], normalized[j])
            if sim >= threshold:
                similarity_edges.append({
                    "from": node_ids[i],
                    "to": node_ids[j],
                    "type": "similarity",
                    "weight": float(sim)
                })

    # Sort by weight and limit
    similarity_edges.sort(key=lambda x: x["weight"], reverse=True)
    return similarity_edges[:max_edges]


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
    """Configure a daemon behavior"""
    d = get_daemon()

    try:
        d.configure_behavior(
            request.behavior,
            enabled=request.enabled,
            interval=request.interval
        )
        return {'ok': True, 'message': f'Configured {request.behavior}'}
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
    """Stop the Particle Smasher"""
    d = get_daemon()

    if not d.smasher_config.enabled:
        return {'ok': True, 'message': 'Smasher not running'}

    await d.stop_smasher()

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

def main():
    """Run the server"""
    print("=" * 60)
    print("  FERAL DASHBOARD SERVER")
    print("=" * 60)
    print(f"  URL: http://localhost:8420")
    print(f"  Static: {STATIC_DIR}")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8420,
        log_level="info"
    )


if __name__ == "__main__":
    main()

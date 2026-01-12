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
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

# Add paths (order matters: local first, then FERAL_RESIDENT for dependencies)
REPO_ROOT = Path(__file__).resolve().parents[3]
NEO3000_DIR = Path(__file__).resolve().parent
CORTEX_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic"
sys.path.insert(0, str(REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NEO3000_DIR))  # Local feral_daemon.py takes priority
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
# Global State
# =============================================================================

manager = ConnectionManager()
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
                source_node_id = event.details.get('source_node_id')
                is_new = event.details.get('is_new_node', False)

                # Build full node ID matching constellation format
                node_id = f"chunk:{paper}:{chunk_id}"
                source_id = f"chunk:{paper}:{source_node_id}" if source_node_id else None

                if is_new:
                    # New node discovered - spawn animation
                    asyncio.create_task(manager.broadcast({
                        'type': 'node_discovered',
                        'data': {
                            'node_id': node_id,
                            'label': heading or f"chunk-{chunk_id}",
                            'source_id': source_id,
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
                            'source_id': source_id,
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
    print(f"[FERAL] Starting server...")
    print(f"[FERAL] Dashboard: http://localhost:8420")

    # Initialize resident and daemon
    get_resident()
    get_daemon()

    yield

    # Shutdown
    print(f"[FERAL] Shutting down...")
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
async def get_constellation():
    """Get document constellation graph from cassette network"""
    import sqlite3

    CASSETTES_DIR = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"
    nodes = []
    edges = []

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
                    edges.append({"from": "root", "to": cassette_id})

                    # Sample chunks from this cassette (max 50 per cassette for performance)
                    cursor = conn.execute("""
                        SELECT c.chunk_id, f.path, c.header_text
                        FROM chunks c
                        JOIN files f ON c.file_id = f.file_id
                        LIMIT 50
                    """)
                    for row in cursor.fetchall():
                        chunk_id, source_path, header_text = row
                        # Create short label from header or filename
                        if header_text:
                            label = header_text[:25]
                        elif source_path:
                            label = source_path.split('/')[-1].split('\\')[-1][:25]
                        else:
                            label = f"chunk-{chunk_id}"
                        nodes.append({
                            "id": f"chunk:{cassette_name}:{chunk_id}",
                            "label": label,
                            "group": "page",
                            "path": source_path or ""
                        })
                        edges.append({"from": cassette_id, "to": f"chunk:{cassette_name}:{chunk_id}"})

            except Exception as e:
                # Skip broken databases
                continue

        return {'ok': True, 'nodes': nodes, 'edges': edges}

    except Exception as e:
        return {'ok': False, 'error': str(e), 'nodes': [], 'edges': []}


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

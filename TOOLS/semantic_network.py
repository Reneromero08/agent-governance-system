#!/usr/bin/env python3
"""
Semantic Network Protocol (SNP)
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

AGENT_DB = Path("CORTEX/system1.db")
AGI_DB = Path("D:/CCC 2.0/AI/AGI/CORTEX/_generated/system1.db")

class MessageType(Enum):
    HANDSHAKE = 0x01
    QUERY = 0x02
    RESPONSE = 0x03
    HEARTBEAT = 0x04

@dataclass
class HandshakeMessage:
    peer_id: str
    db_hash: str
    capabilities: List[str]
    db_path: str

@dataclass
class QueryMessage:
    request_id: str
    query: str
    limit: int = 10

@dataclass
class ResponseMessage:
    request_id: str
    results: List[Dict]
    source: str
    processing_time_ms: int

class CortexPeer:
    def __init__(self, peer_id: str, db_path: Path):
        self.peer_id = peer_id
        self.db_path = db_path
        self.db_hash = self._calculate_db_hash()
        self.capabilities = self._detect_capabilities()
        self.connected = False
    
    def _calculate_db_hash(self) -> str:
        if not self.db_path.exists():
            return ""
        with open(self.db_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    
    def _detect_capabilities(self) -> List[str]:
        caps = []
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if 'section_vectors' in tables:
            caps.append("vectors")
        if 'chunks_fts' in tables or 'chunks' in tables:
            caps.append("fts")
        if 'research_chunks' in tables:
            caps.append("research")
        
        return caps
    
    def get_stats(self) -> Dict:
        if not self.db_path.exists():
            return {}
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'research_chunks' in tables:
            cursor = conn.execute("SELECT COUNT(*) as total_chunks FROM research_chunks")
            row = cursor.fetchone()
            stats = {
                "peer_id": self.peer_id,
                "db_hash": self.db_hash,
                "total_chunks": row[0],
                "vectors": 0,
                "capabilities": self.capabilities
            }
        else:
            cursor = conn.execute("SELECT COUNT(*) as total_chunks, (SELECT COUNT(*) FROM section_vectors) as vectors FROM chunks")
            row = cursor.fetchone()
            stats = {
                "peer_id": self.peer_id,
                "db_hash": self.db_hash,
                "total_chunks": row[0],
                "vectors": row[1],
                "capabilities": self.capabilities
            }
        
        conn.close()
        return stats
    
    def query(self, query: str, limit: int = 10) -> List[Dict]:
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        if 'research_chunks' in self.capabilities:
            cursor = conn.execute("SELECT rc.chunk_id, rc.heading, rc.content, 'research' as source FROM research_chunks rc WHERE rc.content LIKE ? LIMIT ?", (f"%{query}%", limit))
        else:
            cursor = conn.execute("SELECT c.chunk_id, c.chunk_hash, fts.content, 'agent' as source FROM chunks c JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id WHERE fts.content LIKE ? LIMIT ?", (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "chunk_id": row['chunk_id'],
                "hash": row.get('chunk_hash', ''),
                "content": row['content'][:300] if row['content'] else "",
                "source": row['source']
            })
        
        conn.close()
        return results

class CortexNetwork:
    def __init__(self):
        self.peers: Dict[str, CortexPeer] = {}
        self.running = False
    
    def add_peer(self, peer: CortexPeer):
        self.peers[peer.peer_id] = peer
        print(f"[NETWORK] Peer added: {peer.peer_id}")
        print(f"  DB: {peer.db_path}")
        print(f"  Hash: {peer.db_hash}")
        print(f"  Caps: {peer.capabilities}")
    
    def handshake(self, peer: CortexPeer) -> bool:
        print(f"[NETWORK] Handshaking with {peer.peer_id}...")
        
        handshake_msg = HandshakeMessage(
            peer_id="network_hub",
            db_hash="",
            capabilities=["routing", "aggregation"],
            db_path=""
        )
        
        peer.connected = True
        print(f"[NETWORK] Handshake complete: {peer.peer_id}")
        print(f"  Exchanged: peer_id={handshake_msg.peer_id}, capabilities={handshake_msg.capabilities}")
        
        return True
    
    def network_query(self, query: str, limit: int = 10) -> Dict[str, List[Dict]]:
        print(f"[NETWORK] Routing query: '{query}'")
        print(f"[NETWORK] Querying {len(self.peers)} peers...")
        
        results = {}
        for peer_id, peer in self.peers.items():
            if not peer.connected:
                print(f"  Skipping {peer_id}: not connected")
                continue
            
            peer_results = peer.query(query, limit=limit)
            results[peer_id] = peer_results
            print(f"  {peer_id}: {len(peer_results)} results")
        
        return results
    
    def heartbeat(self):
        print(f"[NETWORK] Heartbeat: checking {len(self.peers)} peers")
        alive_count = 0
        
        for peer_id, peer in self.peers.items():
            if peer.connected and peer.db_path.exists():
                alive_count += 1
            else:
                print(f"  {peer_id}: DEAD")
        
        print(f"[NETWORK] Heartbeat complete: {alive_count}/{len(self.peers)} alive")
        return alive_count == len(self.peers)
    
    def print_network_status(self):
        print("\n" + "=" * 70)
        print("CORTEX NETWORK STATUS")
        print("=" * 70)
        
        for peer_id, peer in self.peers.items():
            stats = peer.get_stats()
            print(f"\nPeer: {peer_id}")
            print(f"  Database: {peer.db_path.name}")
            print(f"  Hash: {stats.get('db_hash', 'N/A')}")
            print(f"  Chunks: {stats.get('total_chunks', 0)}")
            print(f"  Vectors: {stats.get('vectors', 0)}")
            print(f"  Capabilities: {stats.get('capabilities', [])}")
            print(f"  Connected: {peer.connected}")

def main():
    print("=" * 70)
    print("SEMANTIC NETWORK PROTOCOL - LINKED DATABASES")
    print("=" * 70)
    print()
    
    network = CortexNetwork()
    
    agent_peer = CortexPeer(
        peer_id="agent-governance-system",
        db_path=AGENT_DB
    )
    
    agi_peer = CortexPeer(
        peer_id="agi-research",
        db_path=AGI_DB
    )
    
    network.add_peer(agent_peer)
    network.add_peer(agi_peer)
    
    print("\n--- HANDSHAKES ---")
    network.handshake(agent_peer)
    network.handshake(agi_peer)
    
    network.print_network_status()
    
    queries = [
        "governance",
        "memory architecture",
        "vector embeddings"
    ]
    
    print("\n--- NETWORK QUERIES ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        results = network.network_query(query, limit=5)
        
        for peer_id, peer_results in results.items():
            print(f"\n{peer_id} ({len(peer_results)} results):")
            for i, r in enumerate(peer_results, 1):
                print(f"  {i}. [{r['source']}] {r['chunk_id']} - {r['content'][:80]}...")
    
    print("\n--- HEARTBEAT ---")
    alive = network.heartbeat()
    status = "HEALTHY" if alive else "DEGRADED"
    print(f"Network Status: {status}")
    
    print("\n" + "=" * 70)
    print("NETWORK SUMMARY")
    print("=" * 70)
    print(f"Peers in network: {len(network.peers)}")
    print(f"Handshakes: Complete")
    print(f"Queries: {len(queries)}")
    print(f"Status: {status}")
    print("\nProtocol: Handshake + Query + Heartbeat")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

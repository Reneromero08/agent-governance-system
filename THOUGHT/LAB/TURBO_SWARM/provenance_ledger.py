#!/usr/bin/env python3
"""
Provenance Ledger

Immutable ledger for tracking execution provenance:
- Swarm tasks and results
- Who ran what (agent/human provenance)
- Cryptographic verification (Merkle roots)

Note: This is NOT Kahneman's "System 2" (slow thinking).
      System 2 thinking is implemented via hierarchical chunk navigation.
      This is purely an audit/provenance layer.
"""

import sqlite3
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

# Configuration
DB_PATH = Path(__file__).resolve().parent / "provenance.db"

class ProvenanceLedger:
    """Immutable ledger for governance and provenance."""
    
    def __init__(self, db_path: Path = DB_PATH, writer: Optional[GuardedWriter] = None):
        self.db_path = db_path
        self.writer = writer
        if self.writer is None:
             # Lazy init default writer if we are running in an env that supports it
            repo_root = Path(__file__).resolve().parents[3]
            self.writer = GuardedWriter(
                project_root=repo_root,
                durable_roots=[
                    "LAW/CONTRACTS/_runs",
                    "THOUGHT/LAB/TURBO_SWARM"
                ]
            )
            self.writer.open_commit_gate()

        self.writer.mkdir_durable("THOUGHT/LAB/TURBO_SWARM")
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        
    def _init_schema(self):
        """Initialize ledger schema."""
        self.conn.executescript("""
            -- Runs table: High-level execution sessions
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                merkle_root TEXT
            );
            
            -- Tasks table: Individual steps within a run
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                result_hash TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            
            -- Provenance table: Who authorized what
            CREATE TABLE IF NOT EXISTS provenance (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL, -- User or Agent ID
                action TEXT NOT NULL,
                target_id TEXT NOT NULL, -- Run ID or Task ID
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                signature TEXT -- Optional digital signature
            );
        """)
        self.conn.commit()
        
    def log_run(self, run_id: str, agent_id: str, prompt: str) -> str:
        """Log a new execution run."""
        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, agent_id, prompt_hash) VALUES (?, ?, ?)",
                (run_id, agent_id, prompt_hash)
            )
        return run_id
        
    def log_task(self, task_id: str, run_id: str, description: str, status: str = "PENDING"):
        """Log a task within a run."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO tasks (task_id, run_id, description, status) VALUES (?, ?, ?, ?)",
                (task_id, run_id, description, status)
            )
            
    def update_task_result(self, task_id: str, result: str):
        """Update task with result and calculate hash."""
        result_hash = hashlib.sha256(result.encode('utf-8')).hexdigest()
        
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET status = 'COMPLETED', result_hash = ? WHERE task_id = ?",
                (result_hash, task_id)
            )
            
    def compute_merkle_root(self, run_id: str) -> str:
        """Compute Merkle root for a completed run."""
        cursor = self.conn.execute(
            "SELECT result_hash FROM tasks WHERE run_id = ? ORDER BY task_id",
            (run_id,)
        )
        hashes = [row['result_hash'] for row in cursor if row['result_hash']]
        
        if not hashes:
            return ""
            
        # Simple Merkle Trace (linear chain for now)
        merkle_root = hashes[0]
        for h in hashes[1:]:
            merkle_root = hashlib.sha256((merkle_root + h).encode('utf-8')).hexdigest()
            
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET merkle_root = ? WHERE run_id = ?",
                (merkle_root, run_id)
            )
            
        return merkle_root

    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Demo: Provenance ledger usage."""
    repo_root = Path(__file__).resolve().parents[3]
    writer = GuardedWriter(
        project_root=repo_root,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "THOUGHT/LAB/TURBO_SWARM"
        ]
    )
    writer.open_commit_gate()
    ledger = ProvenanceLedger(writer=writer)
    
    run_id = f"run_{int(time.time())}"
    print(f"Logging Run: {run_id}")
    ledger.log_run(run_id, "agent-007", "Build the Death Star")
    
    task_id = f"task_{int(time.time())}"
    ledger.log_task(task_id, run_id, "Procure Kyber Crystals")
    
    # Simulate work
    ledger.update_task_result(task_id, "Success: Crystals obtained from Ilum.")
    
    # Verify
    root = ledger.compute_merkle_root(run_id)
    print(f"Run Merkle Root: {root}")
    
    ledger.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Message Cassette Database (Phase 3)

SQLite schema and DB-level enforcement for invariants.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .paths import get_system3_db, get_sqlite_connection


class MessageCassetteDB:

    DB_NAME = "cat_chat.db"  # Consolidated DB
    DB_VERSION = 1
    
    def __init__(self, repo_root: Optional[Path] = None, db_path: Optional[Path] = None):
        if db_path is not None:
            self.db_path = db_path
        else:
            self.db_path = get_system3_db(repo_root)
        
        self._conn = None
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = get_sqlite_connection(self.db_path)
        return self._conn
    
    def _init_db(self):
        conn = self._get_conn()
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            INSERT OR IGNORE INTO cassette_meta (key, value)
            VALUES ('schema_version', ?)
        """, (str(self.DB_VERSION),))
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_messages (
                message_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('USER', 'PLANNER', 'SYSTEM', 'WORKER')),
                idempotency_key TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(run_id, idempotency_key)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_run_id 
            ON cassette_messages(run_id, created_at)
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_jobs (
                job_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                intent TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (message_id) REFERENCES cassette_messages(message_id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_message_id 
            ON cassette_jobs(message_id, ordinal)
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_steps (
                step_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'LEASED', 'COMMITTED')),
                lease_owner TEXT,
                lease_expires_at TEXT,
                fencing_token INTEGER DEFAULT 0,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (job_id) REFERENCES cassette_jobs(job_id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_job_ordinal 
            ON cassette_steps(job_id, ordinal)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_status_expires 
            ON cassette_steps(status, lease_expires_at)
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_receipts (
                receipt_id TEXT PRIMARY KEY,
                step_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                fencing_token INTEGER NOT NULL,
                outcome TEXT NOT NULL CHECK(outcome IN ('SUCCESS', 'FAILURE', 'ABORTED')),
                receipt_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (step_id) REFERENCES cassette_steps(step_id),
                FOREIGN KEY (job_id) REFERENCES cassette_jobs(job_id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_receipts_step_id 
            ON cassette_receipts(step_id)
        """)
        
        self._create_triggers(conn)
        conn.commit()
    
    def _create_triggers(self, conn: sqlite3.Connection):
        conn.execute("DROP TRIGGER IF EXISTS tr_messages_append_only_update")
        conn.execute("""
            CREATE TRIGGER tr_messages_append_only_update
            BEFORE UPDATE ON cassette_messages
            BEGIN
                SELECT RAISE(ABORT, 'cassette_messages is append-only: UPDATE forbidden');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_messages_append_only_delete")
        conn.execute("""
            CREATE TRIGGER tr_messages_append_only_delete
            BEFORE DELETE ON cassette_messages
            BEGIN
                SELECT RAISE(ABORT, 'cassette_messages is append-only: DELETE forbidden');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_receipts_append_only_update")
        conn.execute("""
            CREATE TRIGGER tr_receipts_append_only_update
            BEFORE UPDATE ON cassette_receipts
            BEGIN
                SELECT RAISE(ABORT, 'cassette_receipts is append-only: UPDATE forbidden');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_receipts_append_only_delete")
        conn.execute("""
            CREATE TRIGGER tr_receipts_append_only_delete
            BEFORE DELETE ON cassette_receipts
            BEGIN
                SELECT RAISE(ABORT, 'cassette_receipts is append-only: DELETE forbidden');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_steps_fsm_illegal_1")
        conn.execute("""
            CREATE TRIGGER tr_steps_fsm_illegal_1
            BEFORE UPDATE OF status ON cassette_steps
            WHEN NEW.status = 'COMMITTED' AND OLD.status = 'PENDING'
            BEGIN
                SELECT RAISE(ABORT, 'Illegal FSM transition: PENDING -> COMMITTED');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_steps_fsm_illegal_2")
        conn.execute("""
            CREATE TRIGGER tr_steps_fsm_illegal_2
            BEFORE UPDATE OF status ON cassette_steps
            WHEN NEW.status = 'PENDING' AND OLD.status = 'LEASED'
            BEGIN
                SELECT RAISE(ABORT, 'Illegal FSM transition: LEASED -> PENDING');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_steps_fsm_illegal_3")
        conn.execute("""
            CREATE TRIGGER tr_steps_fsm_illegal_3
            BEFORE UPDATE OF status ON cassette_steps
            WHEN NEW.status = 'LEASED' AND OLD.status = 'COMMITTED'
            BEGIN
                SELECT RAISE(ABORT, 'Illegal FSM transition: COMMITTED -> LEASED');
            END
        """)
        
        conn.execute("DROP TRIGGER IF EXISTS tr_steps_lease_prevent_direct_set")
        conn.execute("""
            CREATE TRIGGER tr_steps_lease_prevent_direct_set
            BEFORE UPDATE OF lease_owner, lease_expires_at, fencing_token ON cassette_steps
            WHEN (NEW.lease_owner <> OLD.lease_owner OR 
                   NEW.lease_expires_at <> OLD.lease_expires_at OR 
                   NEW.fencing_token <> OLD.fencing_token) AND 
                   NOT (OLD.status = 'PENDING' AND NEW.status = 'LEASED')
            BEGIN
                SELECT RAISE(ABORT, 'Lease fields can only be set during PENDING -> LEASED transition');
            END
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cassette_job_budgets (
                job_id TEXT PRIMARY KEY,
                bytes_consumed INTEGER DEFAULT 0,
                symbols_consumed INTEGER DEFAULT 0,
                FOREIGN KEY (job_id) REFERENCES cassette_jobs(job_id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_job_budgets_job_id 
            ON cassette_job_budgets(job_id)
        """)
    
    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

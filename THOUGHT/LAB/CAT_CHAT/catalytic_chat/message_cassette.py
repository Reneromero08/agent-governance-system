#!/usr/bin/env python3
"""
Message Cassette API (Phase 3)

Minimal Python API layer over DB-enforced invariants.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from catalytic_chat.message_cassette_db import MessageCassetteDB


class MessageCassetteError(Exception):
    pass


def _generate_id(prefix: str, *parts: str) -> str:
    seed = ":".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(seed.encode()).hexdigest()[:16]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageCassette:
    
    def __init__(self, repo_root: Optional[Path] = None, db_path: Optional[Path] = None):
        self._db = MessageCassetteDB(repo_root=repo_root, db_path=db_path)
    
    def _get_conn(self) -> sqlite3.Connection:
        return self._db._get_conn()
    
    def post_message(
        self,
        payload: Dict[str, Any],
        run_id: str,
        source: str,
        idempotency_key: Optional[str] = None
    ) -> Tuple[str, str]:
        conn = self._get_conn()
        
        if source not in ("USER", "PLANNER", "SYSTEM", "WORKER"):
            raise MessageCassetteError(f"Invalid source: {source}")
        
        payload_json = json.dumps(payload)
        intent = payload.get("intent", "")
        
        try:
            message_id = _generate_id("msg", run_id, idempotency_key or "")
            job_id = _generate_id("job", message_id)
            
            conn.execute("""
                INSERT INTO cassette_messages 
                (message_id, run_id, source, idempotency_key, payload_json)
                VALUES (?, ?, ?, ?, ?)
            """, (message_id, run_id, source, idempotency_key, payload_json))
            
            conn.execute("""
                INSERT INTO cassette_jobs 
                (job_id, message_id, intent, ordinal)
                VALUES (?, ?, ?, 1)
            """, (job_id, message_id, intent))
            
            step_id = _generate_id("step", job_id, "1")
            conn.execute("""
                INSERT INTO cassette_steps 
                (step_id, job_id, ordinal, status, payload_json)
                VALUES (?, ?, 1, 'PENDING', ?)
            """, (step_id, job_id, payload_json))
            
            conn.commit()
            return (message_id, job_id)
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and "run_id" in str(e):
                cursor = conn.execute("""
                    SELECT m.message_id, j.job_id 
                    FROM cassette_messages m
                    JOIN cassette_jobs j ON m.message_id = j.message_id
                    WHERE m.run_id = ? AND m.idempotency_key = ?
                """, (run_id, idempotency_key))
                row = cursor.fetchone()
                if row:
                    return (row["message_id"], row["job_id"])
            raise MessageCassetteError(f"Failed to post message: {e}")
    
    def claim_step(
        self,
        run_id: str,
        worker_id: str,
        ttl_seconds: int = 300
    ) -> Dict[str, Any]:
        conn = self._get_conn()
        
        lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        
        conn.execute("BEGIN IMMEDIATE")
        
        try:
            cursor = conn.execute("""
                SELECT s.step_id, s.job_id, j.message_id, s.ordinal, s.fencing_token
                FROM cassette_steps s
                JOIN cassette_jobs j ON s.job_id = j.job_id
                JOIN cassette_messages m ON j.message_id = m.message_id
                WHERE s.status = 'PENDING' AND m.run_id = ?
                ORDER BY m.created_at ASC, j.ordinal ASC, s.ordinal ASC
                LIMIT 1
            """, (run_id,))
            
            row = cursor.fetchone()
            if row is None:
                conn.rollback()
                raise MessageCassetteError(f"No pending steps available for run_id: {run_id}")
            
            step_id = row["step_id"]
            job_id = row["job_id"]
            message_id = row["message_id"]
            ordinal = row["ordinal"]
            current_token = row["fencing_token"]
            new_token = current_token + 1
            
            conn.execute("""
                UPDATE cassette_steps
                SET status = 'LEASED',
                    lease_owner = ?,
                    lease_expires_at = ?,
                    fencing_token = ?
                WHERE step_id = ?
            """, (worker_id, lease_expires_at, new_token, step_id))
            
            conn.commit()
            
            cursor = conn.execute("""
                SELECT payload_json FROM cassette_steps WHERE step_id = ?
            """, (step_id,))
            payload_json = cursor.fetchone()["payload_json"]
            
            return {
                "step_id": step_id,
                "job_id": job_id,
                "message_id": message_id,
                "ordinal": ordinal,
                "payload": json.loads(payload_json),
                "fencing_token": new_token,
                "lease_expires_at": lease_expires_at
            }
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise MessageCassetteError(f"Claim failed: {e}")
    
    def complete_step(
        self,
        run_id: str,
        step_id: str,
        worker_id: str,
        fencing_token: int,
        receipt_payload: Dict[str, Any],
        outcome: str
    ) -> str:
        conn = self._get_conn()
        
        if outcome not in ("SUCCESS", "FAILURE", "ABORTED"):
            raise MessageCassetteError(f"Invalid outcome: {outcome}")
        
        receipt_json = json.dumps(receipt_payload)
        receipt_id = _generate_id("rcpt", step_id, outcome)
        
        try:
            cursor = conn.execute("""
                SELECT s.step_id, s.job_id, s.status, s.lease_owner, s.lease_expires_at, s.fencing_token,
                       m.run_id
                FROM cassette_steps s
                JOIN cassette_jobs j ON s.job_id = j.job_id
                JOIN cassette_messages m ON j.message_id = m.message_id
                WHERE s.step_id = ?
            """, (step_id,))
            
            row = cursor.fetchone()
            if row is None:
                raise MessageCassetteError(f"Step not found: {step_id}")
            
            if row["run_id"] != run_id:
                raise MessageCassetteError(f"Step run_id mismatch: expected {run_id}, got {row['run_id']}")
            
            if row["status"] != "LEASED":
                raise MessageCassetteError(f"Step not leased: {step_id} (status: {row['status']})")
            
            if row["lease_owner"] != worker_id:
                raise MessageCassetteError(f"Step leased by different worker: {row['lease_owner']}")
            
            if row["fencing_token"] != fencing_token:
                raise MessageCassetteError(f"Fencing token mismatch: expected {row['fencing_token']}, got {fencing_token}")
            
            lease_expires = row["lease_expires_at"]
            if lease_expires:
                lease_time = datetime.fromisoformat(lease_expires.replace("Z", "+00:00"))
                if lease_time < datetime.now(timezone.utc):
                    raise MessageCassetteError(f"Step lease expired: {step_id}")
            
            conn.execute("""
                INSERT INTO cassette_receipts 
                (receipt_id, step_id, job_id, worker_id, fencing_token, outcome, receipt_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (receipt_id, step_id, row["job_id"], worker_id, fencing_token, outcome, receipt_json))
            
            conn.execute("""
                UPDATE cassette_steps
                SET status = 'COMMITTED'
                WHERE step_id = ?
            """, (step_id,))
            
            conn.commit()
            return receipt_id
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise MessageCassetteError(f"Complete failed: {e}")
    
    def verify_cassette(self, run_id: Optional[str] = None) -> None:
        conn = self._get_conn()
        
        issues = []
        
        cursor = conn.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        if fk_enabled != 1:
            issues.append(f"Foreign keys are not enabled (PRAGMA foreign_keys = {fk_enabled}, expected 1)")
        
        required_tables = ["cassette_messages", "cassette_jobs", "cassette_steps", "cassette_receipts"]
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        for table in required_tables:
            if table not in existing_tables:
                issues.append(f"Required table missing: {table}")
        
        required_triggers = [
            "tr_messages_append_only_update",
            "tr_messages_append_only_delete",
            "tr_receipts_append_only_update",
            "tr_receipts_append_only_delete",
            "tr_steps_fsm_illegal_1",
            "tr_steps_fsm_illegal_2",
            "tr_steps_fsm_illegal_3",
            "tr_steps_lease_prevent_direct_set"
        ]
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        existing_triggers = {row[0] for row in cursor.fetchall()}
        for trigger in required_triggers:
            if trigger not in existing_triggers:
                issues.append(f"Required trigger missing: {trigger}")
        
        where_clause = ""
        params: tuple = ()
        if run_id:
            where_clause = "WHERE m.run_id = ?"
            params = (run_id,)
        
        cursor = conn.execute(f"""
            SELECT COUNT(*) as count
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            {where_clause}
        """, params)
        total_steps = cursor.fetchone()["count"]
        
        cursor = conn.execute(f"""
            SELECT COUNT(*) as count
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            {where_clause} AND s.status = 'PENDING'
        """, params)
        pending_steps = cursor.fetchone()["count"]
        
        cursor = conn.execute(f"""
            SELECT COUNT(*) as count
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            {where_clause} AND s.status = 'LEASED'
        """, params)
        leased_steps = cursor.fetchone()["count"]
        
        cursor = conn.execute(f"""
            SELECT COUNT(*) as count
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            {where_clause} AND s.status = 'COMMITTED'
        """, params)
        committed_steps = cursor.fetchone()["count"]
        
        cursor = conn.execute(f"""
            SELECT s.step_id, s.lease_owner, s.lease_expires_at
            FROM cassette_steps s
            JOIN cassette_jobs j ON s.job_id = j.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            {where_clause} AND s.status = 'LEASED' AND s.lease_expires_at < datetime('now')
        """, params)
        for row in cursor.fetchall():
            issues.append(f"Step {row['step_id']} (owner: {row['lease_owner']}) has expired lease")
        
        if issues:
            import sys
            print(f"FAIL: {len(issues)} issue(s) found", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
            raise MessageCassetteError(f"Verification failed with {len(issues)} issue(s)")
        else:
            import sys
            print("PASS: All invariants verified", file=sys.stderr)
    
    def close(self):
        self._db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

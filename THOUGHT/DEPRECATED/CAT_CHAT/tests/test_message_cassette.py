import pytest
import json
import sqlite3
from pathlib import Path
from catalytic_chat.message_cassette_db import MessageCassetteDB
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError


@pytest.fixture
def cassette_db(tmp_path):
    db_path = tmp_path / "test_cassette.db"
    db = MessageCassetteDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def cassette(tmp_path):
    db_path = tmp_path / "test_cassette.db"
    mc = MessageCassette(db_path=db_path)
    yield mc
    mc.close()


def test_messages_append_only_trigger_blocks_update_delete(cassette_db):
    conn = cassette_db._get_conn()
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("""
            UPDATE cassette_messages SET payload_json = '{"updated": true}' 
            WHERE message_id = 'msg_test'
        """)
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM cassette_messages WHERE message_id = 'msg_test'")


def test_receipts_append_only_trigger_blocks_update_delete(cassette_db, cassette):
    conn = cassette_db._get_conn()
    
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    step_id = result["step_id"]
    
    cassette.complete_step(
        run_id="test_run",
        step_id=step_id,
        worker_id="worker1",
        fencing_token=result["fencing_token"],
        receipt_payload={"output": "done"},
        outcome="SUCCESS"
    )
    
    cursor = conn.execute("SELECT receipt_id FROM cassette_receipts LIMIT 1")
    receipt_id = cursor.fetchone()["receipt_id"]
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("""
            UPDATE cassette_receipts SET receipt_json = '{"updated": true}' 
            WHERE receipt_id = ?
        """, (receipt_id,))
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM cassette_receipts WHERE receipt_id = ?", (receipt_id,))


def test_fsm_illegal_transition_rejected_by_trigger(cassette_db):
    conn = cassette_db._get_conn()
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    conn.execute("""
        INSERT INTO cassette_jobs 
        (job_id, message_id, intent, ordinal)
        VALUES ('job_test', 'msg_test', 'test_intent', 1)
    """)
    
    conn.execute("""
        INSERT INTO cassette_steps 
        (step_id, job_id, ordinal, status, payload_json)
        VALUES ('step_test', 'job_test', 1, 'PENDING', '{"test": true}')
    """)
    
    with pytest.raises(sqlite3.IntegrityError, match="PENDING -> COMMITTED"):
        conn.execute("UPDATE cassette_steps SET status = 'COMMITTED' WHERE step_id = 'step_test'")
    
    conn.execute("UPDATE cassette_steps SET status = 'LEASED', lease_owner='worker', lease_expires_at='2099-01-01' WHERE step_id = 'step_test'")
    
    with pytest.raises(sqlite3.IntegrityError, match="LEASED -> PENDING"):
        conn.execute("UPDATE cassette_steps SET status = 'PENDING' WHERE step_id = 'step_test'")
    
    conn.execute("UPDATE cassette_steps SET status = 'COMMITTED' WHERE step_id = 'step_test'")
    
    with pytest.raises(sqlite3.IntegrityError, match="COMMITTED -> LEASED"):
        conn.execute("UPDATE cassette_steps SET status = 'LEASED' WHERE step_id = 'step_test'")


def test_claim_deterministic_order(cassette):
    for i in range(3):
        cassette.post_message(
            payload={"intent": f"test{i}"},
            run_id="test_run",
            source="USER",
            idempotency_key=f"key{i}"
        )
    
    claims = []
    for i in range(3):
        result = cassette.claim_step("test_run", f"worker{i}", ttl_seconds=60)
        claims.append(result["step_id"])
    
    assert len(claims) == len(set(claims))
    
    cursor = cassette._get_conn().execute("""
        SELECT s.step_id, m.created_at, j.ordinal as job_ordinal, s.ordinal as step_ordinal
        FROM cassette_steps s
        JOIN cassette_jobs j ON s.job_id = j.job_id
        JOIN cassette_messages m ON j.message_id = m.message_id
        WHERE s.step_id IN (?, ?, ?)
        ORDER BY m.created_at ASC, j.ordinal ASC, s.ordinal ASC
    """, (*claims,))
    
    ordered_steps = [row["step_id"] for row in cursor.fetchall()]
    
    assert set(claims) == set(ordered_steps)
    
    for step_id in ordered_steps:
        assert step_id in claims


def test_complete_rejects_stale_token(cassette):
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    step_id = result["step_id"]
    current_token = result["fencing_token"]
    
    with pytest.raises(MessageCassetteError, match="Fencing token mismatch"):
        cassette.complete_step(
            run_id="test_run",
            step_id=step_id,
            worker_id="worker1",
            fencing_token=current_token - 1,
            receipt_payload={"output": "done"},
            outcome="SUCCESS"
        )


def test_complete_rejects_expired_lease(cassette):
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=1)
    step_id = result["step_id"]
    
    import time
    time.sleep(1.5)
    
    with pytest.raises(MessageCassetteError, match="lease expired"):
        cassette.complete_step(
            run_id="test_run",
            step_id=step_id,
            worker_id="worker1",
            fencing_token=result["fencing_token"],
            receipt_payload={"output": "done"},
            outcome="SUCCESS"
        )


def test_receipt_requires_existing_step_job_fk(cassette_db):
    conn = cassette_db._get_conn()
    
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute("""
            INSERT INTO cassette_receipts 
            (receipt_id, step_id, job_id, worker_id, fencing_token, outcome, receipt_json)
            VALUES ('rcpt_bad', 'step_nonexistent', 'job_nonexistent', 'worker', 1, 'SUCCESS', '{}')
        """)
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    conn.execute("""
        INSERT INTO cassette_jobs 
        (job_id, message_id, intent, ordinal)
        VALUES ('job_test', 'msg_test', 'test_intent', 1)
    """)
    
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute("""
            INSERT INTO cassette_receipts 
            (receipt_id, step_id, job_id, worker_id, fencing_token, outcome, receipt_json)
            VALUES ('rcpt_bad_step', 'step_nonexistent', 'job_test', 'worker', 1, 'SUCCESS', '{}')
        """)


def test_post_message_idempotency(cassette):
    payload = {"intent": "test", "data": "value"}
    idempotency_key = "test_key"
    
    message_id1, job_id1 = cassette.post_message(
        payload=payload,
        run_id="test_run",
        source="USER",
        idempotency_key=idempotency_key
    )
    
    message_id2, job_id2 = cassette.post_message(
        payload=payload,
        run_id="test_run",
        source="USER",
        idempotency_key=idempotency_key
    )
    
    assert message_id1 == message_id2
    assert job_id1 == job_id2


def test_claim_fencing_token_increments(cassette):
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result1 = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    assert result1["fencing_token"] == 1
    
    cassette.complete_step(
        run_id="test_run",
        step_id=result1["step_id"],
        worker_id="worker1",
        fencing_token=result1["fencing_token"],
        receipt_payload={"output": "done"},
        outcome="SUCCESS"
    )
    
    cursor = cassette._get_conn().execute("""
        SELECT fencing_token FROM cassette_steps WHERE step_id = ?
    """, (result1["step_id"],))
    
    assert cursor.fetchone()["fencing_token"] == 1


def test_verify_cassette_no_issues(cassette):
    cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    cassette.verify_cassette(run_id="test_run")


def test_claim_no_pending_steps_raises(cassette):
    with pytest.raises(MessageCassetteError, match="No pending steps"):
        cassette.claim_step("empty_run", "worker1", ttl_seconds=60)


def test_complete_invalid_outcome_raises(cassette):
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    
    with pytest.raises(MessageCassetteError, match="Invalid outcome"):
        cassette.complete_step(
            run_id="test_run",
            step_id=result["step_id"],
            worker_id="worker1",
            fencing_token=result["fencing_token"],
            receipt_payload={"output": "done"},
            outcome="INVALID"
        )


def test_post_invalid_source_raises(cassette):
    with pytest.raises(MessageCassetteError, match="Invalid source"):
        cassette.post_message(
            payload={"intent": "test"},
            run_id="test_run",
            source="INVALID"
        )


def test_steps_delete_allowed_when_persisting_design(cassette_db):
    conn = cassette_db._get_conn()
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    conn.execute("""
        INSERT INTO cassette_jobs 
        (job_id, message_id, intent, ordinal)
        VALUES ('job_test', 'msg_test', 'test_intent', 1)
    """)
    
    conn.execute("""
        INSERT INTO cassette_steps 
        (step_id, job_id, ordinal, status, payload_json)
        VALUES ('step_test', 'job_test', 1, 'PENDING', '{"test": true}')
    """)
    
    steps_before = conn.execute("SELECT COUNT(*) as count FROM cassette_steps").fetchone()["count"]
    
    conn.execute("DELETE FROM cassette_steps WHERE step_id = 'step_test'")
    
    steps_after = conn.execute("SELECT COUNT(*) as count FROM cassette_steps").fetchone()["count"]
    
    assert steps_after == steps_before - 1


def test_messages_update_delete_blocked_by_triggers(cassette_db):
    conn = cassette_db._get_conn()
    
    message_id, job_id = MessageCassette(db_path=cassette_db.db_path).post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("""
            UPDATE cassette_messages SET payload_json = '{"hacked": true}' 
            WHERE message_id = ?
        """, (message_id,))
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM cassette_messages WHERE message_id = ?", (message_id,))


def test_receipts_update_delete_blocked_by_triggers(cassette_db):
    cassette = MessageCassette(db_path=cassette_db.db_path)
    
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    step_id = result["step_id"]
    
    receipt_id = cassette.complete_step(
        run_id="test_run",
        step_id=step_id,
        worker_id="worker1",
        fencing_token=result["fencing_token"],
        receipt_payload={"output": "done"},
        outcome="SUCCESS"
    )
    
    conn = cassette_db._get_conn()
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("""
            UPDATE cassette_receipts SET receipt_json = '{"hacked": true}' 
            WHERE receipt_id = ?
        """, (receipt_id,))
    
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM cassette_receipts WHERE receipt_id = ?", (receipt_id,))


def test_illegal_fsm_transition_blocked(cassette_db):
    conn = cassette_db._get_conn()
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    conn.execute("""
        INSERT INTO cassette_jobs 
        (job_id, message_id, intent, ordinal)
        VALUES ('job_test', 'msg_test', 'test_intent', 1)
    """)
    
    conn.execute("""
        INSERT INTO cassette_steps 
        (step_id, job_id, ordinal, status, payload_json)
        VALUES ('step_test', 'job_test', 1, 'PENDING', '{"test": true}')
    """)
    
    with pytest.raises(sqlite3.IntegrityError, match="PENDING -> COMMITTED"):
        conn.execute("UPDATE cassette_steps SET status = 'COMMITTED' WHERE step_id = 'step_test'")
    
    conn.execute("UPDATE cassette_steps SET status = 'LEASED', lease_owner='worker', lease_expires_at='2099-01-01' WHERE step_id = 'step_test'")
    
    with pytest.raises(sqlite3.IntegrityError, match="LEASED -> PENDING"):
        conn.execute("UPDATE cassette_steps SET status = 'PENDING' WHERE step_id = 'step_test'")
    
    conn.execute("UPDATE cassette_steps SET status = 'COMMITTED' WHERE step_id = 'step_test'")
    
    with pytest.raises(sqlite3.IntegrityError, match="COMMITTED -> LEASED"):
        conn.execute("UPDATE cassette_steps SET status = 'LEASED' WHERE step_id = 'step_test'")


def test_lease_direct_set_blocked(cassette_db):
    conn = cassette_db._get_conn()
    
    conn.execute("""
        INSERT INTO cassette_messages 
        (message_id, run_id, source, payload_json)
        VALUES ('msg_test', 'test_run', 'USER', '{"test": true}')
    """)
    
    conn.execute("""
        INSERT INTO cassette_jobs 
        (job_id, message_id, intent, ordinal)
        VALUES ('job_test', 'msg_test', 'test_intent', 1)
    """)
    
    conn.execute("""
         INSERT INTO cassette_steps 
         (step_id, job_id, ordinal, status, payload_json)
         VALUES ('step_test', 'job_test', 1, 'PENDING', '{"test": true}')
     """)
    
    conn.execute("UPDATE cassette_steps SET status = 'LEASED' WHERE step_id = 'step_test'")
    
    with pytest.raises(sqlite3.IntegrityError, match="Lease fields can only be set"):
        conn.execute("""
            UPDATE cassette_steps 
            SET lease_owner = 'hacker', lease_expires_at = '2099-01-01', fencing_token = 999
            WHERE step_id = 'step_test'
        """)


def test_complete_fails_on_stale_token_even_if_lease_owner_matches(cassette_db):
    cassette = MessageCassette(db_path=cassette_db.db_path)
    
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=60)
    step_id = result["step_id"]
    current_token = result["fencing_token"]
    
    with pytest.raises(MessageCassetteError, match="Fencing token mismatch"):
        cassette.complete_step(
            run_id="test_run",
            step_id=step_id,
            worker_id="worker1",
            fencing_token=current_token - 1,
            receipt_payload={"output": "done"},
            outcome="SUCCESS"
        )


def test_complete_fails_on_expired_lease(cassette_db):
    cassette = MessageCassette(db_path=cassette_db.db_path)
    
    message_id, job_id = cassette.post_message(
        payload={"intent": "test"},
        run_id="test_run",
        source="USER"
    )
    
    result = cassette.claim_step("test_run", "worker1", ttl_seconds=1)
    step_id = result["step_id"]
    
    import time
    time.sleep(1.5)
    
    with pytest.raises(MessageCassetteError, match="lease expired"):
        cassette.complete_step(
            run_id="test_run",
            step_id=step_id,
            worker_id="worker1",
            fencing_token=result["fencing_token"],
            receipt_payload={"output": "done"},
            outcome="SUCCESS"
        )


def test_verify_cassette_checks_foreign_keys_enabled(cassette):
    import sys
    from io import StringIO
    
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    
    try:
        cassette.verify_cassette(run_id=None)
        stderr_output = sys.stderr.getvalue()
        assert "foreign_keys" not in stderr_output.lower()
    finally:
        sys.stderr = old_stderr

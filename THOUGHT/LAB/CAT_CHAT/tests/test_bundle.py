#!/usr/bin/env python3
"""
Bundle System Tests (Phase 5)
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path

import pytest

from catalytic_chat.bundle import BundleBuilder, BundleVerifier, BundleError, _sha256, _canonical_json
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError, _generate_id
from catalytic_chat.section_indexer import SectionIndexer, build_index
from catalytic_chat.symbol_registry import SymbolRegistry


@pytest.fixture
def repo_root(tmp_path):
    """Create a minimal test repository structure."""
    src_dir = tmp_path / "THOUGHT" / "LAB" / "CAT_CHAT"
    src_dir.mkdir(parents=True)

    law_dir = tmp_path / "LAW" / "CANON"
    law_dir.mkdir(parents=True)

    cortex_dir = tmp_path / "CORTEX" / "_generated"
    cortex_dir.mkdir(parents=True)

    (law_dir / "CONTRACT.md").write_text("""
# Test Document

This is a test document for symbol extraction.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
""")

    return tmp_path


@pytest.fixture
def indexed_repo(repo_root):
    """Build index for test repository."""
    build_index(repo_root=repo_root, substrate_mode="sqlite")
    return repo_root


def create_completed_job(repo_root, run_id, request_id, intent):
    """Create a completed job for testing."""
    cassette = MessageCassette(repo_root=repo_root)
    section_indexer = SectionIndexer(repo_root=repo_root)

    sections = section_indexer.build_full_index()
    if not sections:
        raise ValueError("No sections found in index")

    section_id = sections[0].section_id

    request = {
        "run_id": run_id,
        "request_id": request_id,
        "intent": intent,
        "inputs": {
            "files": [],
            "notes": []
        },
        "budgets": {
            "max_steps": 2,
            "max_bytes": 10000000,
            "max_symbols": 10
        }
    }

    try:
        conn = cassette._get_conn()

        message_id = _generate_id("msg", run_id, request_id)
        job_id = _generate_id("job", message_id)

        conn.execute("""
            INSERT INTO cassette_messages
            (message_id, run_id, source, idempotency_key, payload_json)
            VALUES (?, ?, 'USER', ?, ?)
        """, (message_id, run_id, request_id, json.dumps(request)))

        conn.execute("""
            INSERT INTO cassette_jobs
            (job_id, message_id, intent, ordinal)
            VALUES (?, ?, ?, 1)
        """, (job_id, message_id, intent))

        conn.execute("""
            INSERT INTO cassette_job_budgets
            (job_id, bytes_consumed, symbols_consumed)
            VALUES (?, 0, 0)
        """, (job_id,))

        step1_id = _generate_id("step", job_id, "1")
        step1_payload = {
            "step_id": step1_id,
            "ordinal": 1,
            "op": "READ_SECTION",
            "refs": {"section_id": section_id},
            "constraints": {"slice": "head(10)"}
        }

        conn.execute("""
            INSERT INTO cassette_steps
            (step_id, job_id, ordinal, status, payload_json)
            VALUES (?, ?, 1, 'PENDING', ?)
        """, (step1_id, job_id, json.dumps(step1_payload)))

        conn.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_builder = BundleBuilder(repo_root=indexed_repo)

            try:
                with pytest.raises(BundleError, match="Job completeness gate failed"):
                    bundle_builder.build(run_id, job_id, Path(tmpdir))
            finally:
                bundle_builder.close()

    finally:
        cassette.close()


def test_canonical_json_produces_deterministic_output():
    """Test canonical JSON function produces deterministic output."""
    data = {
        "b": 2,
        "a": 1,
        "c": {
            "z": "value",
            "y": "other"
        }
    }

    json1 = _canonical_json(data)
    json2 = _canonical_json(data)

    assert json1 == json2
    assert json1 == '{"a":1,"b":2,"c":{"y":"other","z":"value"}}'


def test_sha256_produces_consistent_hash():
    """Test SHA256 function produces consistent hash."""
    content = "test content"
    hash1 = _sha256(content)
    hash2 = _sha256(content)

    assert hash1 == hash2
    assert len(hash1) == 64

    import hashlib
    expected = hashlib.sha256(content.encode('utf-8')).hexdigest()
    assert hash1 == expected

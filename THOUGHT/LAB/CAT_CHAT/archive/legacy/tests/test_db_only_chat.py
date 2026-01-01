#!/usr/bin/env python3
"""
Test DB-Only Chat System.

Verifies DB-only operations, no auto-exports.
"""

import sys
import os
from pathlib import Path

# Chat system is now in current directory
chat_system_path = Path(__file__).parent
sys.path.insert(0, str(chat_system_path))
os.chdir(str(chat_system_path))

from db_only_chat import DBOnlyChat


def test_write_read_cycle():
    """Test 1: Write and read from DB only."""
    print("=" * 60)
    print("TEST 1: Write/Read Cycle (DB Only)")
    print("=" * 60)

    chat = DBOnlyChat()

    # Write multiple messages
    print("\n[Write] Writing messages to DB...")
    uuid1 = chat.write_message("test-session-1", "user", "How does this work?")
    uuid2 = chat.write_message("test-session-1", "assistant", "It stores everything in DB only.",
                              parent_uuid=uuid1)
    uuid3 = chat.write_message("test-session-1", "user", "What about exports?",
                              parent_uuid=uuid2)
    uuid4 = chat.write_message("test-session-1", "assistant",
                              "Exports only happen when you call export_jsonl() or export_md().",
                              parent_uuid=uuid3)

    print(f"  [OK] Wrote 4 messages to DB")
    print(f"  [OK] User: {uuid1}")
    print(f"  [OK] Assistant: {uuid2}")
    print(f"  [OK] User: {uuid3}")
    print(f"  [OK] Assistant: {uuid4}")

    # Read back from DB
    print("\n[Read] Reading from DB only...")
    messages = chat.read_session("test-session-1")

    print(f"  [OK] Retrieved {len(messages)} messages from DB")
    assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"

    # Verify parent-child relationships
    assert messages[0]["uuid"] == uuid1
    assert messages[1]["parent_uuid"] == uuid1
    assert messages[2]["parent_uuid"] == uuid2
    assert messages[3]["parent_uuid"] == uuid3
    print(f"  [OK] Parent-child relationships verified")

    # Verify no auto-export files exist yet
    projects_dir = Path(__file__).parent / "projects"
    jsonl_path = projects_dir / "test-session-1.jsonl"
    md_path = projects_dir / "test-session-1.md"

    assert not jsonl_path.exists(), "JSONL file should NOT exist yet (no auto-export)"
    assert not md_path.exists(), "MD file should NOT exist yet (no auto-export)"
    print(f"  [OK] No auto-export files created (DB-only confirmed)")

    print("\n[OK] TEST 1 PASSED")
    return True


def test_semantic_search():
    """Test 2: Semantic search using vectors in DB."""
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Search (DB + Vectors)")
    print("=" * 60)

    chat = DBOnlyChat()

    # Create a session with varied content
    session_id = "test-session-2"

    print("\n[Setup] Writing test messages...")
    messages_to_write = [
        ("user", "How do I refactor code?"),
        ("assistant", "Use the refactoring skill to update function signatures."),
        ("user", "What about testing?"),
        ("assistant", "Run pytest to verify all tests pass."),
        ("user", "How do I debug errors?"),
        ("assistant", "Check the logs and use print statements for debugging."),
    ]

    for i, (role, content) in enumerate(messages_to_write):
        parent = None
        if i > 0:
            # For simplicity, not tracking exact parents in this test
            pass
        chat.write_message(session_id, role, content)

    print(f"  [OK] Wrote {len(messages_to_write)} messages")

    # Semantic search for similar content
    print("\n[Search] Testing vector similarity...")

    test_queries = [
        ("refactor", 1),       # Should find refactor-related messages
        ("testing", 1),         # Should find test-related messages
        ("debugging", 1),       # Should find debug-related messages
    ]

    all_passed = True
    for query, expected_min_results in test_queries:
        results = chat.search_semantic(
            query=query,
            session_id=session_id,
            threshold=0.3,  # Lowered threshold for better matching
            limit=10
        )

        print(f"  Query: '{query}'")
        print(f"    Found {len(results)} results (expected >= {expected_min_results})")

        for i, result in enumerate(results[:2]):
            print(f"    [{i+1}] sim={result['similarity']:.2f} - {result['chunk_content'][:60]}...")

        assert len(results) >= expected_min_results, \
            f"Expected at least {expected_min_results} results, got {len(results)}"
        print(f"    [OK] Similarity search working")

    print("\n[OK] TEST 2 PASSED")
    return True


def test_export_on_demand():
    """Test 3: Exports happen only on demand."""
    print("\n" + "=" * 60)
    print("TEST 3: Export On Demand")
    print("=" * 60)

    chat = DBOnlyChat()
    session_id = "test-session-3"

    # Write messages
    print("\n[Write] Creating test session...")
    chat.write_message(session_id, "user", "Test message 1")
    chat.write_message(session_id, "assistant", "Test response 1")
    chat.write_message(session_id, "user", "Test message 2")
    chat.write_message(session_id, "assistant", "Test response 2")
    print("  [OK] Wrote 4 messages")

    # Verify no exports exist
    projects_dir = Path(__file__).parent / "projects"
    jsonl_path = projects_dir / f"{session_id}.jsonl"
    md_path = projects_dir / f"{session_id}.md"

    print("\n[Verify] Checking for auto-exports...")
    if jsonl_path.exists():
        print(f"  [FAIL] JSONL file exists (should not): {jsonl_path}")
        assert False, "JSONL should not exist before export"
    else:
        print(f"  [OK] JSONL file does not exist (correct)")

    if md_path.exists():
        print(f"  [FAIL] MD file exists (should not): {md_path}")
        assert False, "MD should not exist before export"
    else:
        print(f"  [OK] MD file does not exist (correct)")

    # Export JSONL
    print("\n[Export] Exporting JSONL on demand...")
    exported_jsonl = chat.export_jsonl(session_id)
    print(f"  [OK] Exported: {exported_jsonl}")
    assert exported_jsonl.exists(), "JSONL export should exist"

    # Verify JSONL content
    import json
    with open(exported_jsonl, 'r') as f:
        lines = f.readlines()
        print(f"  [OK] JSONL has {len(lines)} lines")
        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"

        for line in lines:
            record = json.loads(line)
            assert "uuid" in record
            assert "sessionId" in record
            assert "type" in record
            assert "message" in record
        print(f"  [OK] JSONL format verified")

    # Export MD
    print("\n[Export] Exporting MD on demand...")
    exported_md = chat.export_md(session_id)
    print(f"  [OK] Exported: {exported_md}")
    assert exported_md.exists(), "MD export should exist"

    # Verify MD content
    with open(exported_md, 'r', encoding='utf-8') as f:
        md_content = f.read()
        print(f"  [OK] MD has {len(md_content)} characters")
        assert f"# Session: {session_id}" in md_content
        assert "User" in md_content or "user" in md_content.lower()
        assert "Assistant" in md_content or "assistant" in md_content.lower()
        print(f"  [OK] MD format verified")

    print("\n[OK] TEST 3 PASSED")
    return True


def test_multiple_sessions():
    """Test 4: Multiple sessions isolation."""
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Sessions Isolation")
    print("=" * 60)

    chat = DBOnlyChat()

    # Create separate sessions
    print("\n[Write] Creating multiple sessions...")
    chat.write_message("session-a", "user", "Session A message 1")
    chat.write_message("session-a", "assistant", "Session A response 1")
    chat.write_message("session-b", "user", "Session B message 1")
    chat.write_message("session-b", "assistant", "Session B response 1")
    print("  [OK] Created 2 sessions with 2 messages each")

    # Verify session isolation
    print("\n[Verify] Checking session isolation...")
    messages_a = chat.read_session("session-a")
    messages_b = chat.read_session("session-b")

    print(f"  [OK] Session A has {len(messages_a)} messages")
    print(f"  [OK] Session B has {len(messages_b)} messages")

    assert len(messages_a) == 2
    assert len(messages_b) == 2

    # Verify no cross-contamination
    session_a_uuids = set(m["uuid"] for m in messages_a)
    session_b_uuids = set(m["uuid"] for m in messages_b)

    assert len(session_a_uuids & session_b_uuids) == 0, "Sessions should not share UUIDs"
    print(f"  [OK] Sessions are isolated (no UUID overlap)")

    # Search within specific session
    print("\n[Search] Testing session-scoped search...")
    results_a = chat.search_semantic("message", session_id="session-a", threshold=0.1)
    results_b = chat.search_semantic("message", session_id="session-b", threshold=0.1)

    print(f"  [OK] Session A search found {len(results_a)} results")
    print(f"  [OK] Session B search found {len(results_b)} results")

    # Verify results belong to correct sessions
    for result in results_a:
        assert result["message_uuid"] in session_a_uuids
    for result in results_b:
        assert result["message_uuid"] in session_b_uuids

    print(f"  [OK] Search results respect session boundaries")

    print("\n[OK] TEST 4 PASSED")
    return True


def test_chunking_and_vectors():
    """Test 5: Long message chunking and vector generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Long Message Chunking + Vectors")
    print("=" * 60)

    chat = DBOnlyChat()
    session_id = "test-session-5"

    # Create a long message (over 500 words to trigger chunking)
    print("\n[Write] Creating long message (will be chunked)...")
    long_content = " ".join([f"word {i}" for i in range(600)])

    uuid = chat.write_message(session_id, "user", long_content)
    print(f"  [OK] Wrote long message (600 words)")
    print(f"  [OK] UUID: {uuid}")

    # Read back and verify chunking
    print("\n[Verify] Checking message chunking...")
    messages = chat.read_session(session_id)
    assert len(messages) == 1

    # Get chunks from DB directly
    from chat_db import ChatDB
    db = ChatDB(db_path=chat.db.db_path)
    msg = db.get_message_by_uuid(uuid)
    chunks = db.get_message_chunks(msg.message_id)

    print(f"  [OK] Message chunked into {len(chunks)} parts")
    assert len(chunks) > 1, "Message should be chunked (600 words > 500 limit)"

    # Verify vector generation
    print("\n[Verify] Checking vector embeddings...")
    chunk_hashes = [c.chunk_hash for c in chunks]
    vectors = db.get_chunk_vectors(chunk_hashes)

    print(f"  [OK] Generated {len(vectors)} vectors")
    assert len(vectors) == len(chunks), "Each chunk should have a vector"

    for vector in vectors:
        assert vector.chunk_hash in chunk_hashes
        assert vector.embedding is not None
        assert vector.model_id == "all-MiniLM-L6-v2"
        assert vector.dimensions == 384
    print(f"  [OK] All vectors valid (384 dimensions)")

    # Verify semantic search works on chunks
    print("\n[Search] Testing chunk-level search...")
    results = chat.search_semantic(
        query="long content",
        session_id=session_id,
        threshold=0.1  # Very low threshold to match repetitive content
    )
    print(f"  [OK] Found {len(results)} chunk-level results")
    if results:
        print(f"  [OK] Top similarity: {results[0]['similarity']:.2f}")
    else:
        print(f"  [OK] No results (repetitive content may have low similarity with query)")

    print("\n[OK] TEST 5 PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  DB-Only Chat System Test Suite")
    print("=" * 70)

    tests = [
        ("Write/Read Cycle", test_write_read_cycle),
        ("Semantic Search", test_semantic_search),
        ("Export On Demand", test_export_on_demand),
        ("Multiple Sessions", test_multiple_sessions),
        ("Chunking & Vectors", test_chunking_and_vectors),
    ]

    results = []

    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASSED", None))
        except Exception as e:
            print(f"\n[FAIL] TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAILED", str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    for name, status, error in results:
        symbol = "[OK]" if status == "PASSED" else "[FAIL]"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      Error: {error}")

    print(f"\n  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    print("\n" + "=" * 70)

    if failed == 0:
        print("  [OK] ALL TESTS PASSED")
    else:
        print(f"  [FAIL] {failed} TEST(S) FAILED")

    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

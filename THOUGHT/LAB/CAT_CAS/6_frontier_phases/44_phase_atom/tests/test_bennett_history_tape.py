"""Shared BennettHistoryTape lifecycle verification test.

Proves:
  1. Untouched tape verification fails (tautology guard).
  2. record_operation mutates SHA-256 hash.
  3. uncompute restores SHA-256 hash.
  4. verify passes only after mutation and full restoration.
  5. verify fails if history_stack is not empty (dirty).
  6. Multiple operations uncompute correctly in LIFO order.
"""

import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from catalytic_tape import BennettHistoryTape


def test_untouched_tape_fails():
    """Verify that a never-modified tape is rejected."""
    t = BennettHistoryTape(size_mb=1)
    try:
        t.verify()
        assert False, "Should have raised RuntimeError for untouched tape"
    except RuntimeError:
        pass  # expected


def test_record_mutates_hash():
    """Verify that record_operation actually changes the tape hash."""
    t = BennettHistoryTape(size_mb=1)
    h0 = hashlib.sha256(t.tape).hexdigest()
    t.record_operation(("test", 42))
    h1 = hashlib.sha256(t.tape).hexdigest()
    assert h0 != h1, "Hash should change after record_operation"


def test_mutation_flag():
    """Verify was_modified flag is set on non-zero data."""
    t = BennettHistoryTape(size_mb=1)
    assert t.was_modified is False
    t.record_operation(("flag_test", 12345))
    assert t.was_modified is True


def test_zeros_dont_set_flag():
    """Verify was_modified stays False if no non-zero bytes are XORed.
    This is a structural property of the guard, tested with empty string
    which encodes to zero-length bytes (no mutation occurs)."""
    t = BennettHistoryTape(size_mb=1)
    t.record_operation("")
    assert t.was_modified is False, "Empty data should not set was_modified"


def test_uncompute_restores_hash():
    """Verify that uncompute restores the original tape hash."""
    t = BennettHistoryTape(size_mb=1)
    h_initial = hashlib.sha256(t.tape).hexdigest()
    t.record_operation(("payload_A", 42))
    t.record_operation(("payload_B", 99))
    h_dirty = hashlib.sha256(t.tape).hexdigest()
    assert h_dirty != h_initial, "Hash should change after writes"
    t.uncompute()
    h_restored = hashlib.sha256(t.tape).hexdigest()
    assert h_restored == h_initial, "Hash should match initial after uncompute"


def test_verify_passes_after_full_lifecycle():
    """Verify passes only after mutation + full uncompute."""
    t = BennettHistoryTape(size_mb=1)
    t.record_operation(("lifecycle_test", 777))
    t.uncompute()
    assert t.verify() is True


def test_verify_fails_with_dirty_stack():
    """Verify that incomplete uncompute is rejected."""
    t = BennettHistoryTape(size_mb=1)
    t.record_operation(("dirty_test", 888))
    assert len(t.history_stack) > 0
    try:
        t.verify()
        assert False, "Should have raised ValueError for non-empty history stack"
    except ValueError:
        pass  # expected


def test_multiple_operations_lifo():
    """Verify multiple operations uncompute in correct LIFO order."""
    t = BennettHistoryTape(size_mb=1)
    h_initial = hashlib.sha256(t.tape).hexdigest()
    for i in range(5):
        t.record_operation(("lifo_step", i))
    t.uncompute()
    h_restored = hashlib.sha256(t.tape).hexdigest()
    assert h_restored == h_initial, "LIFO uncompute should restore hash"
    assert len(t.history_stack) == 0, "History stack should be empty"
    assert t.verify() is True


def test_history_stack_cleared():
    """Verify history stack is fully cleared after uncompute."""
    t = BennettHistoryTape(size_mb=1)
    t.record_operation(("a", 1))
    t.record_operation(("b", 2))
    t.record_operation(("c", 3))
    assert len(t.history_stack) == 3
    t.uncompute()
    assert len(t.history_stack) == 0


def test_bytes_written_counter():
    """Verify bytes_written tracks total bytes."""
    t = BennettHistoryTape(size_mb=1)
    assert t.bytes_written == 0
    t.record_operation(("counter_test", 42))
    assert t.bytes_written > 0

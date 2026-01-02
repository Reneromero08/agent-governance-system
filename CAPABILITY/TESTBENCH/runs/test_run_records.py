"""
Tests for Z.2.3 – Immutable run artifacts

Validates:
- Deterministic hashing
- Roundtrip correctness
- Immutability (re-putting identical data yields same hash)
- Invalid data is rejected
- Corruption is detected
"""

import pytest
import json
from unittest.mock import patch

from CAPABILITY.RUNS.records import (
    put_task_spec,
    put_status,
    put_output_hashes,
    load_task_spec,
    load_status,
    load_output_hashes,
    RunRecordException,
    InvalidInputException,
    _canonical_encode,
    _canonical_decode,
)

from CAPABILITY.CAS.cas import ObjectNotFoundException, CorruptObjectException


# ============================================================================
# Canonical encoding tests
# ============================================================================

class TestCanonicalEncoding:
    """Test canonical encoding/decoding"""

    def test_encode_simple_dict(self):
        """Simple dict encodes deterministically"""
        obj = {"key": "value"}
        data = _canonical_encode(obj)
        assert data == b'{"key":"value"}'

    def test_encode_sorted_keys(self):
        """Keys are sorted for deterministic encoding"""
        obj = {"z": 1, "a": 2, "m": 3}
        data = _canonical_encode(obj)
        assert data == b'{"a":2,"m":3,"z":1}'

    def test_encode_nested(self):
        """Nested structures are encoded correctly"""
        obj = {"outer": {"z": 1, "a": 2}}
        data = _canonical_encode(obj)
        assert data == b'{"outer":{"a":2,"z":1}}'

    def test_encode_deterministic(self):
        """Same input produces same output"""
        obj = {"b": 2, "a": 1, "c": 3}
        data1 = _canonical_encode(obj)
        data2 = _canonical_encode(obj)
        assert data1 == data2

    def test_encode_rejects_non_serializable(self):
        """Non-JSON-serializable objects raise exception"""
        obj = {"func": lambda x: x}
        with pytest.raises(InvalidInputException, match="not JSON-serializable"):
            _canonical_encode(obj)

    def test_decode_valid_json(self):
        """Valid JSON decodes correctly"""
        data = b'{"key":"value"}'
        obj = _canonical_decode(data)
        assert obj == {"key": "value"}

    def test_decode_invalid_utf8(self):
        """Invalid UTF-8 raises exception"""
        data = b'\xff\xfe'
        with pytest.raises(RunRecordException, match="Failed to decode JSON"):
            _canonical_decode(data)

    def test_decode_invalid_json(self):
        """Invalid JSON raises exception"""
        data = b'{invalid json}'
        with pytest.raises(RunRecordException, match="Failed to decode JSON"):
            _canonical_decode(data)


# ============================================================================
# TASK_SPEC tests
# ============================================================================

class TestTaskSpec:
    """Test task specification records"""

    def test_put_and_load_simple_spec(self):
        """Simple task spec roundtrip"""
        spec = {"action": "test", "params": {"x": 1}}
        hash1 = put_task_spec(spec)

        # Validate hash format
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)

        # Load and verify
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_put_deterministic(self):
        """Same spec produces same hash"""
        spec = {"action": "test", "params": {"x": 1}}
        hash1 = put_task_spec(spec)
        hash2 = put_task_spec(spec)
        assert hash1 == hash2

    def test_put_key_order_independent(self):
        """Key order does not affect hash"""
        spec1 = {"z": 3, "a": 1, "m": 2}
        spec2 = {"a": 1, "m": 2, "z": 3}
        hash1 = put_task_spec(spec1)
        hash2 = put_task_spec(spec2)
        assert hash1 == hash2

    def test_put_rejects_non_dict(self):
        """Non-dict input is rejected"""
        with pytest.raises(InvalidInputException, match="must be a dict"):
            put_task_spec("not a dict")

    def test_put_rejects_empty_dict(self):
        """Empty dict is rejected"""
        with pytest.raises(InvalidInputException, match="cannot be empty"):
            put_task_spec({})

    def test_put_rejects_non_serializable(self):
        """Non-serializable content is rejected"""
        spec = {"func": lambda x: x}
        with pytest.raises(InvalidInputException, match="not JSON-serializable"):
            put_task_spec(spec)

    def test_load_rejects_invalid_hash(self):
        """Invalid hash format is rejected"""
        with pytest.raises(InvalidInputException, match="Invalid hash format"):
            load_task_spec("not-a-hash")

    def test_load_rejects_non_string(self):
        """Non-string hash is rejected"""
        with pytest.raises(InvalidInputException, match="must be a string"):
            load_task_spec(123)

    def test_load_missing_object(self):
        """Missing object raises exception"""
        fake_hash = "0" * 64
        with pytest.raises(RunRecordException, match="not found"):
            load_task_spec(fake_hash)

    def test_load_validates_dict_type(self):
        """Loaded object must be a dict"""
        # Store a non-dict object directly in CAS
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode(["not", "a", "dict"])
        hash1 = cas_put(data)

        # Attempt to load as task spec should fail
        with pytest.raises(RunRecordException, match="must be a dict"):
            load_task_spec(hash1)

    def test_immutability(self):
        """Re-putting identical spec yields same hash"""
        spec = {"action": "test", "data": [1, 2, 3]}
        hash1 = put_task_spec(spec)
        hash2 = put_task_spec(spec)
        hash3 = put_task_spec(spec)
        assert hash1 == hash2 == hash3


# ============================================================================
# STATUS tests
# ============================================================================

class TestStatus:
    """Test status records"""

    def test_put_and_load_simple_status(self):
        """Simple status roundtrip"""
        status = {"state": "SUCCESS"}
        hash1 = put_status(status)

        # Validate hash format
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)

        # Load and verify
        loaded = load_status(hash1)
        assert loaded == status

    def test_put_and_load_with_error(self):
        """Status with error details"""
        status = {
            "state": "FAILURE",
            "error_code": "ERR_TIMEOUT",
            "message": "Operation timed out"
        }
        hash1 = put_status(status)
        loaded = load_status(hash1)
        assert loaded == status

    def test_put_deterministic(self):
        """Same status produces same hash"""
        status = {"state": "RUNNING", "progress": 50}
        hash1 = put_status(status)
        hash2 = put_status(status)
        assert hash1 == hash2

    def test_put_key_order_independent(self):
        """Key order does not affect hash"""
        status1 = {"state": "SUCCESS", "code": 0}
        status2 = {"code": 0, "state": "SUCCESS"}
        hash1 = put_status(status1)
        hash2 = put_status(status2)
        assert hash1 == hash2

    def test_put_rejects_non_dict(self):
        """Non-dict input is rejected"""
        with pytest.raises(InvalidInputException, match="must be a dict"):
            put_status("not a dict")

    def test_put_rejects_empty_dict(self):
        """Empty dict is rejected"""
        with pytest.raises(InvalidInputException, match="cannot be empty"):
            put_status({})

    def test_put_requires_state_field(self):
        """Status must contain 'state' field"""
        with pytest.raises(InvalidInputException, match="must contain 'state' field"):
            put_status({"code": 0})

    def test_load_rejects_invalid_hash(self):
        """Invalid hash format is rejected"""
        with pytest.raises(InvalidInputException, match="Invalid hash format"):
            load_status("not-a-hash")

    def test_load_missing_object(self):
        """Missing object raises exception"""
        fake_hash = "1" * 64
        with pytest.raises(RunRecordException, match="not found"):
            load_status(fake_hash)

    def test_load_validates_dict_type(self):
        """Loaded object must be a dict"""
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode("not a dict")
        hash1 = cas_put(data)

        with pytest.raises(RunRecordException, match="must be a dict"):
            load_status(hash1)

    def test_load_validates_state_field(self):
        """Loaded status must contain 'state' field"""
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode({"no_state": "here"})
        hash1 = cas_put(data)

        with pytest.raises(RunRecordException, match="must contain 'state' field"):
            load_status(hash1)

    def test_immutability(self):
        """Re-putting identical status yields same hash"""
        status = {"state": "PENDING", "queue_position": 5}
        hash1 = put_status(status)
        hash2 = put_status(status)
        hash3 = put_status(status)
        assert hash1 == hash2 == hash3

    def test_different_states_different_hashes(self):
        """Different states produce different hashes"""
        status1 = {"state": "PENDING"}
        status2 = {"state": "RUNNING"}
        hash1 = put_status(status1)
        hash2 = put_status(status2)
        assert hash1 != hash2


# ============================================================================
# OUTPUT_HASHES tests
# ============================================================================

class TestOutputHashes:
    """Test output hashes records"""

    def test_put_and_load_empty_list(self):
        """Empty list roundtrip"""
        hashes = []
        hash1 = put_output_hashes(hashes)

        # Validate hash format
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)

        # Load and verify
        loaded = load_output_hashes(hash1)
        assert loaded == hashes

    def test_put_and_load_single_hash(self):
        """Single hash roundtrip"""
        hashes = ["a" * 64]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes

    def test_put_and_load_multiple_hashes(self):
        """Multiple hashes roundtrip"""
        hashes = [
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes

    def test_put_deterministic(self):
        """Same list produces same hash"""
        hashes = ["1" * 64, "2" * 64]
        hash1 = put_output_hashes(hashes)
        hash2 = put_output_hashes(hashes)
        assert hash1 == hash2

    def test_put_order_matters(self):
        """Order affects hash"""
        hashes1 = ["a" * 64, "b" * 64]
        hashes2 = ["b" * 64, "a" * 64]
        hash1 = put_output_hashes(hashes1)
        hash2 = put_output_hashes(hashes2)
        assert hash1 != hash2

    def test_put_preserves_order(self):
        """Order is preserved in roundtrip"""
        hashes = ["3" * 64, "1" * 64, "2" * 64]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes
        assert loaded[0] == "3" * 64
        assert loaded[1] == "1" * 64
        assert loaded[2] == "2" * 64

    def test_put_rejects_non_list(self):
        """Non-list input is rejected"""
        with pytest.raises(InvalidInputException, match="must be a list"):
            put_output_hashes("not a list")

    def test_put_rejects_non_string_element(self):
        """Non-string elements are rejected"""
        with pytest.raises(InvalidInputException, match="must be a string"):
            put_output_hashes([123])

    def test_put_rejects_invalid_hash_format(self):
        """Invalid hash format is rejected"""
        with pytest.raises(InvalidInputException, match="invalid format"):
            put_output_hashes(["not-a-hash"])

    def test_put_rejects_wrong_length_hash(self):
        """Wrong length hashes are rejected"""
        with pytest.raises(InvalidInputException, match="invalid format"):
            put_output_hashes(["a" * 63])  # Too short

    def test_put_rejects_uppercase_hash(self):
        """Uppercase hashes are rejected"""
        with pytest.raises(InvalidInputException, match="invalid format"):
            put_output_hashes(["A" * 64])

    def test_put_rejects_invalid_hex(self):
        """Non-hex characters are rejected"""
        with pytest.raises(InvalidInputException, match="invalid format"):
            put_output_hashes(["g" * 64])

    def test_load_rejects_invalid_hash(self):
        """Invalid hash format is rejected"""
        with pytest.raises(InvalidInputException, match="Invalid hash format"):
            load_output_hashes("not-a-hash")

    def test_load_missing_object(self):
        """Missing object raises exception"""
        fake_hash = "2" * 64
        with pytest.raises(RunRecordException, match="not found"):
            load_output_hashes(fake_hash)

    def test_load_validates_list_type(self):
        """Loaded object must be a list"""
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode({"not": "a list"})
        hash1 = cas_put(data)

        with pytest.raises(RunRecordException, match="must be a list"):
            load_output_hashes(hash1)

    def test_load_validates_element_types(self):
        """Loaded elements must be strings"""
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode([123, 456])
        hash1 = cas_put(data)

        with pytest.raises(RunRecordException, match="must be a string"):
            load_output_hashes(hash1)

    def test_load_validates_element_format(self):
        """Loaded elements must be valid hashes"""
        from CAPABILITY.CAS.cas import cas_put
        data = _canonical_encode(["not-a-hash"])
        hash1 = cas_put(data)

        with pytest.raises(RunRecordException, match="invalid format"):
            load_output_hashes(hash1)

    def test_immutability(self):
        """Re-putting identical list yields same hash"""
        hashes = ["f" * 64, "e" * 64, "d" * 64]
        hash1 = put_output_hashes(hashes)
        hash2 = put_output_hashes(hashes)
        hash3 = put_output_hashes(hashes)
        assert hash1 == hash2 == hash3


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests across all record types"""

    def test_task_spec_with_real_data(self):
        """Task spec with realistic data structure"""
        spec = {
            "task_id": "task-123",
            "action": "process_data",
            "inputs": {
                "file": "data.csv",
                "options": {
                    "format": "csv",
                    "delimiter": ",",
                    "header": True
                }
            },
            "resources": {
                "cpu": 2,
                "memory": "4GB"
            }
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_status_lifecycle(self):
        """Status records for different lifecycle states"""
        states = [
            {"state": "PENDING"},
            {"state": "RUNNING", "started_at": "2026-01-02T00:00:00Z"},
            {"state": "SUCCESS", "completed_at": "2026-01-02T00:05:00Z"},
        ]

        hashes = [put_status(s) for s in states]
        loaded = [load_status(h) for h in hashes]

        assert loaded == states
        assert len(set(hashes)) == 3  # All different

    def test_output_hashes_with_cas_artifacts(self):
        """Output hashes referencing actual CAS objects"""
        from CAPABILITY.CAS.cas import cas_put

        # Create some artifacts in CAS
        artifact1 = cas_put(b"output data 1")
        artifact2 = cas_put(b"output data 2")
        artifact3 = cas_put(b"output data 3")

        # Store output hashes list
        output_list = [artifact1, artifact2, artifact3]
        list_hash = put_output_hashes(output_list)

        # Load and verify
        loaded = load_output_hashes(list_hash)
        assert loaded == output_list

    def test_complete_run_record(self):
        """Complete run with spec, status, and outputs"""
        # Task spec
        spec = {"action": "compute", "input": 42}
        spec_hash = put_task_spec(spec)

        # Status
        status = {"state": "SUCCESS"}
        status_hash = put_status(status)

        # Outputs (from actual CAS artifacts)
        from CAPABILITY.CAS.cas import cas_put
        out1 = cas_put(b"result 1")
        out2 = cas_put(b"result 2")
        outputs_hash = put_output_hashes([out1, out2])

        # Verify all are different hashes
        assert len({spec_hash, status_hash, outputs_hash}) == 3

        # Verify roundtrip
        assert load_task_spec(spec_hash) == spec
        assert load_status(status_hash) == status
        assert load_output_hashes(outputs_hash) == [out1, out2]

    def test_determinism_across_sessions(self):
        """Same data produces same hashes even in different calls"""
        spec = {"action": "test", "n": 100}
        status = {"state": "RUNNING"}
        hashes_list = ["0" * 64]

        # First session
        h1_spec = put_task_spec(spec)
        h1_status = put_status(status)
        h1_hashes = put_output_hashes(hashes_list)

        # Second session (simulate different time/context)
        h2_spec = put_task_spec(spec)
        h2_status = put_status(status)
        h2_hashes = put_output_hashes(hashes_list)

        # Must be identical
        assert h1_spec == h2_spec
        assert h1_status == h2_status
        assert h1_hashes == h2_hashes

    def test_unicode_handling(self):
        """Unicode content is handled correctly"""
        spec = {
            "description": "Process emoji data 🚀",
            "locale": "日本語",
            "symbols": "αβγδ"
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_numeric_precision(self):
        """Numeric values maintain precision"""
        spec = {
            "pi": 3.14159265359,
            "large": 9007199254740991,
            "small": 0.0000000001
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_nested_structures(self):
        """Deeply nested structures work correctly"""
        spec = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_large_output_list(self):
        """Large output hash lists are handled"""
        # Create 100 fake hashes
        hashes = [f"{i:064d}" for i in range(100)]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes
        assert len(loaded) == 100


# ============================================================================
# Corruption detection tests
# ============================================================================

class TestCorruptionDetection:
    """Test that corruption is detected and reported"""

    def test_cas_corruption_detected_task_spec(self):
        """CAS corruption is detected when loading task spec"""
        spec = {"action": "test"}
        hash1 = put_task_spec(spec)

        # Simulate corruption by mocking cas_get to raise CorruptObjectException
        with patch('CAPABILITY.RUNS.records.cas_get') as mock_get:
            mock_get.side_effect = CorruptObjectException("Corruption detected")

            with pytest.raises(RunRecordException, match="corrupted"):
                load_task_spec(hash1)

    def test_cas_corruption_detected_status(self):
        """CAS corruption is detected when loading status"""
        status = {"state": "SUCCESS"}
        hash1 = put_status(status)

        with patch('CAPABILITY.RUNS.records.cas_get') as mock_get:
            mock_get.side_effect = CorruptObjectException("Corruption detected")

            with pytest.raises(RunRecordException, match="corrupted"):
                load_status(hash1)

    def test_cas_corruption_detected_output_hashes(self):
        """CAS corruption is detected when loading output hashes"""
        hashes = ["a" * 64]
        hash1 = put_output_hashes(hashes)

        with patch('CAPABILITY.RUNS.records.cas_get') as mock_get:
            mock_get.side_effect = CorruptObjectException("Corruption detected")

            with pytest.raises(RunRecordException, match="corrupted"):
                load_output_hashes(hash1)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_task_spec_with_null_values(self):
        """Null values are handled"""
        spec = {"action": "test", "optional": None}
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_status_with_empty_string(self):
        """Empty string values are allowed"""
        status = {"state": "FAILURE", "message": ""}
        hash1 = put_status(status)
        loaded = load_status(hash1)
        assert loaded == status

    def test_output_hashes_all_zeros(self):
        """All-zero hash is valid"""
        hashes = ["0" * 64]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes

    def test_output_hashes_all_f(self):
        """All-f hash is valid"""
        hashes = ["f" * 64]
        hash1 = put_output_hashes(hashes)
        loaded = load_output_hashes(hash1)
        assert loaded == hashes

    def test_task_spec_with_arrays(self):
        """Arrays in task spec work correctly"""
        spec = {
            "inputs": [1, 2, 3, 4, 5],
            "tags": ["a", "b", "c"]
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded == spec

    def test_boolean_values(self):
        """Boolean values are preserved"""
        spec = {
            "enabled": True,
            "debug": False
        }
        hash1 = put_task_spec(spec)
        loaded = load_task_spec(hash1)
        assert loaded["enabled"] is True
        assert loaded["debug"] is False

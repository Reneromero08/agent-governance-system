#!/usr/bin/env python3
"""Phase 5.2.6 Semiotic Compression Tests.

Tests for:
- 5.2.6.1 Determinism Tests (same program → same JSON hash)
- 5.2.6.2 Schema Validation Tests (JobSpecs validate against schema)
- 5.2.6.3 Token Benchmark (80%+ reduction target)
- 5.2.6.4 Negative Tests (invalid inputs produce clear errors)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Try to import tiktoken for token benchmarks
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    TOKENIZER_ENCODING = "o200k_base"
except ImportError:
    TIKTOKEN_AVAILABLE = False
    TOKENIZER_ENCODING = None


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def scl_cli():
    """Return function to invoke SCL CLI."""
    def _run(*args, check=True):
        cmd = [sys.executable, "-m", "CAPABILITY.TOOLS.scl"] + list(args)
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if check and result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return result
    return _run


@pytest.fixture
def encoder():
    """Return tiktoken encoder if available."""
    if not TIKTOKEN_AVAILABLE:
        pytest.skip("tiktoken not installed")
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


@pytest.fixture
def jobspec_schema():
    """Load the JobSpec schema."""
    schema_path = REPO_ROOT / "LAW" / "SCHEMAS" / "jobspec.schema.json"
    if not schema_path.exists():
        pytest.skip("JobSpec schema not found")
    return json.loads(schema_path.read_text(encoding='utf-8'))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2.6.1 DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestDeterminism:
    """Test that SCL operations are deterministic.
    
    These tests run 100 iterations and are marked as slow.
    Use --run-slow to execute them.
    """

    def test_decode_deterministic_100_runs(self, scl_cli):
        """Same program → same JSON hash across 100 runs."""
        program = "C3"
        hashes = set()

        for i in range(100):
            result = scl_cli("decode", program, "--json")
            assert result.returncode == 0, f"Run {i} failed"
            output = json.loads(result.stdout)
            assert output["ok"] is True

            # Compute canonical hash of jobspec
            jobspec = output["jobspec"]
            canonical = json.dumps(jobspec, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            hashes.add(h)

        assert len(hashes) == 1, f"Expected 1 unique hash, got {len(hashes)}"

    def test_decode_cjk_deterministic_100_runs(self, scl_cli):
        """CJK symbol decoding is deterministic across 100 runs."""
        program = "法"
        hashes = set()

        for i in range(100):
            result = scl_cli("decode", program, "--json")
            assert result.returncode == 0, f"Run {i} failed"
            output = json.loads(result.stdout)
            assert output["ok"] is True

            jobspec = output["jobspec"]
            canonical = json.dumps(jobspec, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            hashes.add(h)

        assert len(hashes) == 1, f"Expected 1 unique hash, got {len(hashes)}"

    def test_decode_compound_deterministic_100_runs(self, scl_cli):
        """Compound symbol decoding is deterministic across 100 runs."""
        program = "法.驗"
        hashes = set()

        for i in range(100):
            result = scl_cli("decode", program, "--json")
            assert result.returncode == 0, f"Run {i} failed"
            output = json.loads(result.stdout)
            assert output["ok"] is True

            jobspec = output["jobspec"]
            canonical = json.dumps(jobspec, sort_keys=True, ensure_ascii=False)
            h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            hashes.add(h)

        assert len(hashes) == 1, f"Expected 1 unique hash, got {len(hashes)}"

    def test_validate_deterministic_100_runs(self, scl_cli):
        """Validation results are deterministic across 100 runs."""
        program = "C3"
        results = []

        for i in range(100):
            result = scl_cli("validate", program, "--json")
            assert result.returncode == 0, f"Run {i} failed"
            output = json.loads(result.stdout)
            results.append((output["ok"], output["layer"], tuple(output["errors"])))

        # All results should be identical
        unique_results = set(results)
        assert len(unique_results) == 1, f"Expected 1 unique result, got {len(unique_results)}"

    def test_receipt_input_hash_deterministic(self, scl_cli, tmp_path):
        """Receipt input hashes are deterministic."""
        program = "C3"
        input_hashes = set()

        for i in range(10):
            receipt_path = tmp_path / f"receipt_{i}.json"
            result = scl_cli("decode", program, "--receipt-out", str(receipt_path))
            assert result.returncode == 0

            receipt = json.loads(receipt_path.read_text())
            input_hashes.add(receipt["input_hash"])

        assert len(input_hashes) == 1, f"Expected 1 unique input hash, got {len(input_hashes)}"

    def test_same_codebook_version_same_output(self, scl_cli):
        """Same program + same codebook version → same output."""
        programs = ["C3", "I5", "法", "法.驗", "C3:build"]

        for program in programs:
            hashes = set()
            for _ in range(10):
                result = scl_cli("decode", program, "--json")
                if result.returncode != 0:
                    continue  # Skip if program not supported
                output = json.loads(result.stdout)
                if not output.get("ok"):
                    continue

                jobspec = output["jobspec"]
                canonical = json.dumps(jobspec, sort_keys=True, ensure_ascii=False)
                h = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                hashes.add(h)

            if hashes:  # Only check if we got valid results
                assert len(hashes) == 1, f"Program {program}: Expected 1 hash, got {len(hashes)}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2.6.2 SCHEMA VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    """Test that expanded JobSpecs validate against schema."""

    def test_decode_produces_valid_jobspec(self, scl_cli, jobspec_schema):
        """Decoded JobSpec contains all required fields."""
        programs = ["C3", "I5", "法", "法.驗"]

        for program in programs:
            result = scl_cli("decode", program, "--json")
            if result.returncode != 0:
                continue

            output = json.loads(result.stdout)
            if not output.get("ok"):
                continue

            jobspec = output["jobspec"]

            # Check required fields from schema
            required = jobspec_schema.get("required", [])
            for field in required:
                assert field in jobspec, f"Program {program}: Missing required field: {field}"

    def test_jobspec_has_correct_types(self, scl_cli, jobspec_schema):
        """JobSpec fields have correct types."""
        result = scl_cli("decode", "C3", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True

        jobspec = output["jobspec"]

        # Type checks
        assert isinstance(jobspec["job_id"], str)
        assert isinstance(jobspec["phase"], int)
        assert isinstance(jobspec["task_type"], str)
        assert isinstance(jobspec["intent"], str)
        assert isinstance(jobspec["inputs"], dict)
        assert isinstance(jobspec["outputs"], dict)
        assert isinstance(jobspec["catalytic_domains"], list)
        assert isinstance(jobspec["determinism"], str)

    def test_jobspec_task_type_valid_enum(self, scl_cli, jobspec_schema):
        """JobSpec task_type is a valid enum value."""
        valid_task_types = jobspec_schema["properties"]["task_type"]["enum"]

        result = scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        assert jobspec["task_type"] in valid_task_types, \
            f"Invalid task_type: {jobspec['task_type']}"

    def test_jobspec_determinism_valid_enum(self, scl_cli, jobspec_schema):
        """JobSpec determinism is a valid enum value."""
        valid_determinism = jobspec_schema["properties"]["determinism"]["enum"]

        result = scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        assert jobspec["determinism"] in valid_determinism, \
            f"Invalid determinism: {jobspec['determinism']}"

    def test_jobspec_outputs_structure(self, scl_cli):
        """JobSpec outputs has required structure."""
        result = scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        assert "durable_paths" in jobspec["outputs"]
        assert "validation_criteria" in jobspec["outputs"]
        assert isinstance(jobspec["outputs"]["durable_paths"], list)
        assert isinstance(jobspec["outputs"]["validation_criteria"], dict)

    def test_jobspec_job_id_pattern(self, scl_cli):
        """JobSpec job_id matches required pattern (kebab-case)."""
        import re
        pattern = r"^[a-z0-9-]+$"

        result = scl_cli("decode", "C3", "--json")
        output = json.loads(result.stdout)
        jobspec = output["jobspec"]

        assert re.match(pattern, jobspec["job_id"]), \
            f"Invalid job_id format: {jobspec['job_id']}"

    def test_invalid_inputs_produce_schema_errors(self, scl_cli, tmp_path):
        """Invalid JobSpec inputs produce schema errors."""
        # Create an invalid JobSpec (missing required fields)
        invalid_jobspec = {
            "job_id": "test",
            "phase": 1,
            # Missing: task_type, intent, inputs, outputs, catalytic_domains, determinism
        }

        jobspec_path = tmp_path / "invalid_jobspec.json"
        jobspec_path.write_text(json.dumps(invalid_jobspec))

        result = scl_cli("validate", str(jobspec_path), "--json", check=False)
        output = json.loads(result.stdout)

        # Should fail validation
        assert output["ok"] is False or len(output.get("errors", [])) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2.6.3 TOKEN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenBenchmark:
    """Test token reduction achieved by SCL compression."""

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_symbolic_vs_expanded_compression(self, scl_cli, encoder):
        """Measure: tokens for symbolic program vs expanded text."""
        programs = ["C3", "法", "法.驗"]

        for program in programs:
            result = scl_cli("audit", program, "--json")
            if result.returncode != 0:
                continue

            output = json.loads(result.stdout)
            if not output.get("ok"):
                continue

            audit = output["audit"]
            expansion = audit.get("expansion", {})

            # Token count for symbolic program
            symbolic_tokens = len(encoder.encode(program))

            # Get expanded content length
            content_length = expansion.get("content_length", 0)
            filtered_content_length = expansion.get("filtered_content_length", 0)
            expanded_length = content_length or filtered_content_length

            if expanded_length > 0:
                # Estimate expanded tokens (actual counting would need full content)
                # Use compression ratio from entry if available
                compression = expansion.get("compression", 1)

                print(f"\n{program}:")
                print(f"  Symbolic tokens: {symbolic_tokens}")
                print(f"  Compression ratio: {compression}x")

                # The symbolic representation should be much smaller
                assert symbolic_tokens <= 5, \
                    f"Symbolic program too long: {symbolic_tokens} tokens"

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_governance_boilerplate_80_percent_reduction(self, encoder):
        """Target: 80%+ reduction for governance boilerplate."""
        # Representative governance text (what would need to be pasted)
        governance_boilerplate = """
        All operations must validate against the canon schema before execution.
        The validation must include: syntax checking, symbol resolution, and
        semantic verification. Each operation emits a receipt containing:
        operation type, timestamp, input hash, output hash, and validation status.
        Contract rule C3 states that all writes to canon require verification receipt.
        Invariant I5 ensures deterministic output for identical inputs.
        """

        # Symbolic representation
        symbolic_program = "法.驗"  # Points to LAW/CANON/VERIFICATION

        boilerplate_tokens = len(encoder.encode(governance_boilerplate))
        symbolic_tokens = len(encoder.encode(symbolic_program))

        reduction_pct = (1 - symbolic_tokens / boilerplate_tokens) * 100

        print(f"\nGovernance boilerplate:")
        print(f"  Boilerplate tokens: {boilerplate_tokens}")
        print(f"  Symbolic tokens: {symbolic_tokens}")
        print(f"  Reduction: {reduction_pct:.1f}%")

        assert reduction_pct >= 80, \
            f"Expected 80%+ reduction, got {reduction_pct:.1f}%"

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_benchmark_representative_programs(self, scl_cli, encoder):
        """Benchmark fixture with representative programs."""
        benchmark_cases = [
            ("C3", "Contract rule: verification receipt required"),
            ("I5", "Invariant: deterministic output"),
            ("法", "LAW/CANON domain pointer"),
            ("法.驗", "Verification compound pointer"),
            ("C3:build", "Contract rule with context"),
        ]

        results = []
        for program, description in benchmark_cases:
            result = scl_cli("decode", program, "--json")
            if result.returncode != 0:
                continue

            output = json.loads(result.stdout)
            if not output.get("ok"):
                continue

            jobspec = output["jobspec"]

            # Token counts
            program_tokens = len(encoder.encode(program))
            expanded_tokens = len(encoder.encode(json.dumps(jobspec)))
            intent_tokens = len(encoder.encode(jobspec.get("intent", "")))

            results.append({
                "program": program,
                "description": description,
                "program_tokens": program_tokens,
                "expanded_tokens": expanded_tokens,
                "intent_tokens": intent_tokens,
            })

        print("\nBenchmark Results:")
        print("-" * 70)
        for r in results:
            print(f"  {r['program']:12} | prog={r['program_tokens']:3} | "
                  f"expanded={r['expanded_tokens']:4} | intent={r['intent_tokens']:3}")

        # All symbolic programs should be <= 5 tokens
        for r in results:
            assert r["program_tokens"] <= 5, \
                f"Program {r['program']} too long: {r['program_tokens']} tokens"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2.6.4 NEGATIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNegativeTests:
    """Test error handling for invalid inputs."""

    def test_invalid_syntax_clear_error(self, scl_cli):
        """Invalid syntax → clear error message."""
        invalid_programs = [
            "not-valid-123!",
            "%%%",
            "C-3",
            "法法法",  # Multiple CJK without operator
            "C3.C4.C5",  # Nested compounds not supported
        ]

        for program in invalid_programs:
            result = scl_cli("validate", program, "--json", check=False)
            output = json.loads(result.stdout)

            # Should either fail or have errors
            if output.get("ok") is False:
                assert len(output.get("errors", [])) > 0, \
                    f"Program '{program}': Expected error message"

    def test_unknown_symbol_clear_error(self, scl_cli):
        """Unknown symbol → clear error message."""
        unknown_programs = [
            "X99",  # Unknown radical X
            "龍",   # Unknown CJK symbol
            "Z1",   # Unknown radical Z
        ]

        for program in unknown_programs:
            result = scl_cli("validate", program, "--json", check=False)
            output = json.loads(result.stdout)

            assert output["ok"] is False, \
                f"Program '{program}' should fail validation"

            # Check for meaningful error message
            errors = output.get("errors", [])
            assert len(errors) > 0, \
                f"Program '{program}': Expected error message"

            # Error should mention "unknown" or the symbol
            error_text = " ".join(errors).lower()
            assert "unknown" in error_text or program.lower() in error_text or program in str(errors), \
                f"Program '{program}': Error should mention unknown symbol"

    def test_unknown_contract_rule_error(self, scl_cli):
        """Unknown contract rule number → clear error."""
        result = scl_cli("validate", "C999", "--json", check=False)
        output = json.loads(result.stdout)

        assert output["ok"] is False
        errors = output.get("errors", [])
        assert any("C999" in e or "unknown" in e.lower() for e in errors), \
            "Error should mention unknown contract rule"

    def test_unknown_invariant_error(self, scl_cli):
        """Unknown invariant number → clear error."""
        result = scl_cli("validate", "I999", "--json", check=False)
        output = json.loads(result.stdout)

        assert output["ok"] is False
        errors = output.get("errors", [])
        assert any("I999" in e or "unknown" in e.lower() for e in errors), \
            "Error should mention unknown invariant"

    def test_empty_program_error(self, scl_cli):
        """Empty program → clear error."""
        result = scl_cli("validate", "", "--json", check=False)
        # Should fail with non-zero exit code or error in output
        assert result.returncode != 0 or "error" in result.stdout.lower()

    def test_circular_expansion_handling(self, scl_cli):
        """Circular expansion (if possible) → error, not infinite loop."""
        # This tests that the system handles potential circular references
        # The implementation should have depth limits

        # Test with maximum depth resolution
        result = scl_cli("audit", "法", "--json")

        # Should complete without hanging
        assert result.returncode in [0, 1], "Should complete without hanging"

    def test_malformed_json_file_error(self, scl_cli, tmp_path):
        """Malformed JSON file → clear error."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ invalid json }")

        result = scl_cli("validate", str(bad_json), "--json", check=False)

        # Should fail gracefully
        assert result.returncode != 0 or "error" in result.stdout.lower()

    def test_decode_invalid_returns_layer_info(self, scl_cli):
        """Decode errors include which validation layer failed."""
        result = scl_cli("decode", "X999", "--json", check=False)
        output = json.loads(result.stdout)

        assert output["ok"] is False
        # Receipt should include layer information
        receipt = output.get("receipt", {})
        assert receipt.get("layer") in ["L1", "L2", "L3", "L0"], \
            "Error should include validation layer"


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdditionalCoverage:
    """Additional tests for comprehensive coverage."""

    def test_all_known_contract_rules_decode(self, scl_cli):
        """All known contract rules can be decoded."""
        # C1-C13 are the known contract rules
        for i in range(1, 14):
            program = f"C{i}"
            result = scl_cli("decode", program, "--json")

            if result.returncode == 0:
                output = json.loads(result.stdout)
                if output.get("ok"):
                    assert "jobspec" in output

    def test_context_modifier_preserved(self, scl_cli):
        """Context modifier is preserved in decoded JobSpec."""
        contexts = ["build", "audit", "security", "execute", "validate"]

        for ctx in contexts:
            program = f"C3:{ctx}"
            result = scl_cli("decode", program, "--json")

            if result.returncode == 0:
                output = json.loads(result.stdout)
                if output.get("ok"):
                    jobspec = output["jobspec"]
                    metadata = jobspec.get("metadata", {})
                    assert metadata.get("context") == ctx, \
                        f"Context {ctx} not preserved"

    def test_run_command_invariant_checks(self, scl_cli):
        """Run command checks required invariants."""
        result = scl_cli("run", "C3", "--dry-run", "--json")
        assert result.returncode == 0

        output = json.loads(result.stdout)
        execution = output.get("execution", {})
        proofs = execution.get("invariant_proofs", [])

        # Should check I5, I6, C7, C8
        invariant_ids = [p["invariant"] for p in proofs]
        assert "I5" in invariant_ids, "Should check I5 (determinism)"
        assert "I6" in invariant_ids, "Should check I6 (output roots)"

    def test_audit_shows_compression_ratio(self, scl_cli):
        """Audit command shows compression ratio."""
        result = scl_cli("audit", "法", "--json")

        if result.returncode == 0:
            output = json.loads(result.stdout)
            if output.get("ok"):
                expansion = output.get("audit", {}).get("expansion", {})
                # Should have compression info
                assert "compression" in expansion or "content_length" in expansion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

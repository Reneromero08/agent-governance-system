#!/usr/bin/env python3
"""
Phase 5.3.2 Tests: GOV_IR_SPEC.md Validation

Tests that validate the Governance IR specification is complete,
well-formed, and that the JSON schema is valid.

Deliverables verified:
    - GOV_IR_SPEC.md exists and is normative
    - All IR node types defined
    - concept_unit counting rules specified
    - JSON schema exists and is valid
    - Canonical JSON normalization documented

Usage:
    pytest CAPABILITY/TESTBENCH/integration/test_phase_5_3_2_gov_ir_spec.py -v
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
GOV_IR_SPEC_PATH = PROJECT_ROOT / "LAW" / "CANON" / "SEMANTIC" / "GOV_IR_SPEC.md"
GOV_IR_SCHEMA_PATH = PROJECT_ROOT / "LAW" / "SCHEMAS" / "gov_ir.schema.json"

sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def spec_content() -> str:
    """Load GOV_IR_SPEC.md content."""
    assert GOV_IR_SPEC_PATH.exists(), f"GOV_IR_SPEC.md not found at {GOV_IR_SPEC_PATH}"
    return GOV_IR_SPEC_PATH.read_text(encoding='utf-8')


@pytest.fixture
def spec_hash(spec_content) -> str:
    """Compute SHA-256 of spec content."""
    return hashlib.sha256(spec_content.encode('utf-8')).hexdigest()


@pytest.fixture
def schema() -> dict:
    """Load gov_ir.schema.json."""
    if not GOV_IR_SCHEMA_PATH.exists():
        pytest.skip(f"Schema not found: {GOV_IR_SCHEMA_PATH}")
    with open(GOV_IR_SCHEMA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE AND METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecExistence:
    """Test spec file exists and has proper metadata."""

    def test_spec_exists(self):
        """GOV_IR_SPEC.md must exist."""
        assert GOV_IR_SPEC_PATH.exists(), "GOV_IR_SPEC.md not found"

    def test_schema_exists(self):
        """gov_ir.schema.json must exist."""
        assert GOV_IR_SCHEMA_PATH.exists(), "gov_ir.schema.json not found"

    def test_spec_not_empty(self, spec_content):
        """Spec must have content."""
        assert len(spec_content) > 1000, "Spec appears too short"

    def test_spec_is_normative(self, spec_content):
        """Spec must be marked as NORMATIVE."""
        assert "Status:** NORMATIVE" in spec_content or "status: normative" in spec_content.lower()

    def test_spec_has_canon_id(self, spec_content):
        """Spec must have Canon ID."""
        assert "SEMANTIC-GOV-IR-001" in spec_content

    def test_spec_has_version(self, spec_content):
        """Spec must declare version."""
        assert re.search(r'\*\*Version:\*\*\s*\d+\.\d+\.\d+', spec_content)

    def test_spec_has_phase_reference(self, spec_content):
        """Spec must reference Phase 5.3.2."""
        assert "5.3.2" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# IR NODE TYPES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIRNodeTypes:
    """Test all IR node types are defined."""

    def test_ir_primitives_section(self, spec_content):
        """IR Primitives section must exist."""
        assert "## 2. IR Primitives" in spec_content

    def test_all_node_types_defined(self, spec_content):
        """All 9 node types must be defined."""
        required_types = [
            "constraint",
            "permission",
            "prohibition",
            "reference",
            "gate",
            "operation",
            "literal",
            "sequence",
            "record",
        ]
        for node_type in required_types:
            assert f'"{node_type}"' in spec_content, f"Missing node type: {node_type}"

    def test_constraint_node_defined(self, spec_content):
        """Constraint node must be fully defined."""
        assert "### 3.1 Constraint Node" in spec_content
        assert '"op": "requires"' in spec_content or '"op"' in spec_content

    def test_permission_node_defined(self, spec_content):
        """Permission node must be fully defined."""
        assert "### 3.2 Permission Node" in spec_content

    def test_prohibition_node_defined(self, spec_content):
        """Prohibition node must be fully defined."""
        assert "### 3.3 Prohibition Node" in spec_content

    def test_reference_node_defined(self, spec_content):
        """Reference node must be fully defined."""
        assert "### 3.4 Reference Node" in spec_content
        # Reference types must be listed
        ref_types = ["path", "canon_version", "tool_id", "artifact_hash", "rule_id", "invariant_id"]
        for rt in ref_types:
            assert rt in spec_content, f"Missing reference type: {rt}"

    def test_gate_node_defined(self, spec_content):
        """Gate node must be fully defined."""
        assert "### 3.5 Gate Node" in spec_content
        # Gate types must be listed
        gate_types = ["test", "restore_proof", "allowlist_check", "hash_verify", "schema_validate"]
        for gt in gate_types:
            assert gt in spec_content, f"Missing gate type: {gt}"

    def test_operation_node_defined(self, spec_content):
        """Operation node must be fully defined."""
        assert "### 3.6 Operation Node" in spec_content
        # Boolean operations must be listed
        ops = ["AND", "OR", "NOT"]
        for op in ops:
            assert f"`{op}`" in spec_content, f"Missing operation: {op}"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDE-EFFECTS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSideEffects:
    """Test side-effects flags are documented."""

    def test_side_effects_section(self, spec_content):
        """Side-Effects section must exist."""
        assert "## 4. Side-Effects Flags" in spec_content

    def test_side_effect_fields(self, spec_content):
        """Required side-effect fields must be documented."""
        fields = ["writes", "deletes", "creates", "modifies_canon", "requires_ceremony", "emits_receipt"]
        for field in fields:
            assert field in spec_content, f"Missing side-effect field: {field}"


# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL JSON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCanonicalJSON:
    """Test canonical JSON normalization is specified."""

    def test_canonical_section(self, spec_content):
        """Canonical JSON Schema section must exist."""
        assert "## 5. Canonical JSON Schema" in spec_content

    def test_normalization_rules(self, spec_content):
        """All normalization rules must be documented."""
        rules = ["N1:", "N2:", "N3:", "N4:"]
        for rule in rules:
            assert rule in spec_content, f"Missing normalization rule: {rule}"

    def test_stable_key_ordering(self, spec_content):
        """Stable key ordering must be specified."""
        assert "Stable Key Ordering" in spec_content
        assert "sort" in spec_content.lower()

    def test_explicit_types(self, spec_content):
        """Explicit types rule must be documented."""
        assert "Explicit Types" in spec_content
        assert "coercion" in spec_content.lower()

    def test_canonical_json_function(self, spec_content):
        """canonical_json function must be provided."""
        assert "def canonical_json" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# EQUALITY DEFINITION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEqualityDefinition:
    """Test equality definition is specified."""

    def test_equality_section(self, spec_content):
        """Equality Definition section must exist."""
        assert "## 6. Equality Definition" in spec_content

    def test_byte_identical_equality(self, spec_content):
        """Byte-identical equality must be defined."""
        assert "Byte-Identical" in spec_content or "byte-identical" in spec_content

    def test_ir_equal_function(self, spec_content):
        """ir_equal function must be provided."""
        assert "def ir_equal" in spec_content

    def test_hash_equality(self, spec_content):
        """Hash equality alternative must be documented."""
        assert "ir_hash_equal" in spec_content or "canonical_hash" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT_UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConceptUnit:
    """Test concept_unit is fully defined."""

    def test_concept_unit_section(self, spec_content):
        """concept_unit Definition section must exist."""
        assert "## 7. concept_unit Definition" in spec_content

    def test_concept_unit_definition(self, spec_content):
        """concept_unit must be formally defined."""
        assert "atomic unit of governance meaning" in spec_content.lower() or "concept_unit" in spec_content

    def test_counting_rules(self, spec_content):
        """Counting rules must be specified."""
        assert "### 7.2 Counting Rules" in spec_content
        assert "def count_concept_units" in spec_content

    def test_per_node_counts(self, spec_content):
        """Per-node concept_unit counts must be documented."""
        # Constraint/permission/prohibition/reference/gate = 1 each
        assert "concept_units:** 1" in spec_content or "1 concept_unit" in spec_content.lower()

    def test_literal_zero_units(self, spec_content):
        """Literals must be 0 concept_units."""
        assert "concept_units:** 0" in spec_content

    def test_cdr_calculation(self, spec_content):
        """CDR calculation must be documented."""
        assert "### 7.3 CDR Calculation" in spec_content
        assert "CDR = concept_units" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """Test JSON schema is valid and complete."""

    def test_schema_valid_json(self, schema):
        """Schema must be valid JSON."""
        assert isinstance(schema, dict)

    def test_schema_has_schema_keyword(self, schema):
        """Schema must have $schema keyword."""
        assert "$schema" in schema

    def test_schema_has_type(self, schema):
        """Schema must define type."""
        assert "type" in schema or "oneOf" in schema or "anyOf" in schema

    def test_schema_has_definitions(self, schema):
        """Schema should have definitions for node types."""
        has_defs = "definitions" in schema or "$defs" in schema or "properties" in schema
        assert has_defs, "Schema should define node types"

    def test_spec_references_schema(self, spec_content):
        """Spec must reference the schema file."""
        assert "gov_ir.schema.json" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMappingFromSources:
    """Test governance source mappings are documented."""

    def test_mapping_section(self, spec_content):
        """Mapping from Governance Sources section must exist."""
        assert "## 9. Mapping from Governance Sources" in spec_content

    def test_contract_rules_mapping(self, spec_content):
        """Contract rules mapping must be documented."""
        assert "### 9.1 Contract Rules" in spec_content
        assert "CONTRACT_RULE_IR" in spec_content or "C3" in spec_content

    def test_invariants_mapping(self, spec_content):
        """Invariants mapping must be documented."""
        assert "### 9.2 Invariants" in spec_content
        assert "INVARIANT_IR" in spec_content or "INV-005" in spec_content

    def test_gates_mapping(self, spec_content):
        """Gates mapping must be documented."""
        assert "### 9.3 Gates" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# SPC INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSPCIntegration:
    """Test SPC integration is documented."""

    def test_spc_integration_section(self, spec_content):
        """SPC Integration section must exist."""
        assert "## 10. Integration with SPC" in spec_content

    def test_decoder_output(self, spec_content):
        """Decoder output specification must be documented."""
        assert "Decoder Output" in spec_content or "decode(" in spec_content

    def test_jobspec_wrapping(self, spec_content):
        """JobSpec wrapping must be documented."""
        assert "JobSpec" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-REFERENCES TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferences:
    """Test internal cross-references are valid."""

    def test_references_spc_spec(self, spec_content):
        """Must reference SPC_SPEC.md."""
        assert "SPC_SPEC" in spec_content

    def test_references_token_receipt_spec(self, spec_content):
        """Must reference TOKEN_RECEIPT_SPEC.md."""
        assert "TOKEN_RECEIPT_SPEC" in spec_content

    def test_references_jobspec_schema(self, spec_content):
        """Must reference jobspec.schema.json."""
        assert "jobspec.schema.json" in spec_content


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT HASH RECEIPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentReceipt:
    """Test content integrity for receipt generation."""

    def test_content_hash_reproducible(self, spec_content, spec_hash):
        """Content hash must be reproducible."""
        content2 = GOV_IR_SPEC_PATH.read_text(encoding='utf-8')
        hash2 = hashlib.sha256(content2.encode('utf-8')).hexdigest()
        assert spec_hash == hash2, "Content hash not reproducible"

    def test_content_hash_format(self, spec_hash):
        """Content hash must be valid SHA-256."""
        assert len(spec_hash) == 64
        assert all(c in '0123456789abcdef' for c in spec_hash)

    def test_changelog_present(self, spec_content):
        """Changelog section must be present."""
        assert "## Changelog" in spec_content

    def test_appendices_present(self, spec_content):
        """Appendices must be present."""
        assert "## Appendix" in spec_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

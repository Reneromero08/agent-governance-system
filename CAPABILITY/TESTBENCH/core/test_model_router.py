#!/usr/bin/env python3
"""Tests for model_router.py

Validates deterministic model selection and fallback chain behavior.
"""

from pathlib import Path
import sys

# Ensure CAPABILITY/TOOLS is in path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS"))

import pytest
from model_router import (
    ModelSpec,
    RouterSelection,
    RouterError,
    InvalidModelError,
    EmptyFallbackChainError,
    parse_model_name,
    validate_model,
    select_model,
    create_router_receipt,
    KNOWN_MODELS,
)


class TestParseModelName:
    """Test model name parsing."""

    def test_parse_simple_name(self):
        assert parse_model_name("Claude Sonnet 4.5") == "Claude Sonnet 4.5"

    def test_parse_with_reasoning_annotation(self):
        name = "Claude Sonnet 4.5 (non-thinking) (Reasoning: Medium)"
        assert parse_model_name(name) == "Claude Sonnet 4.5 (non-thinking)"

    def test_parse_empty_string(self):
        assert parse_model_name("") == ""

    def test_parse_only_reasoning(self):
        assert parse_model_name("(Reasoning: High)") == ""


class TestValidateModel:
    """Test model validation."""

    def test_validate_known_model(self):
        spec = validate_model("Claude Sonnet 4.5 (non-thinking)")
        assert isinstance(spec, ModelSpec)
        assert spec.name == "Claude Sonnet 4.5 (non-thinking)"
        assert spec.model_id == "claude-sonnet-4-5"

    def test_validate_with_reasoning_annotation(self):
        spec = validate_model("Claude Sonnet (Reasoning: High)")
        assert isinstance(spec, ModelSpec)
        assert spec.name == "Claude Sonnet"

    def test_validate_unknown_model(self):
        with pytest.raises(InvalidModelError) as exc_info:
            validate_model("GPT-9000")
        assert "Unknown model" in str(exc_info.value)
        assert "GPT-9000" in str(exc_info.value)

    def test_validate_empty_model(self):
        with pytest.raises(InvalidModelError) as exc_info:
            validate_model("")
        assert "cannot be empty" in str(exc_info.value)

    def test_all_known_models_valid(self):
        """Verify all models in KNOWN_MODELS registry are self-consistent."""
        for model_name in KNOWN_MODELS.keys():
            spec = validate_model(model_name)
            assert spec.name == model_name


class TestSelectModel:
    """Test model selection logic."""

    def test_select_primary_only(self):
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=[],
            selection_index=0,
        )
        assert isinstance(selection, RouterSelection)
        assert selection.selected_model.name == "Claude Sonnet 4.5 (non-thinking)"
        assert selection.selection_index == 0
        assert selection.selection_reason == "primary_model"

    def test_select_first_fallback(self):
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=1,
        )
        assert selection.selected_model.name == "Claude Sonnet"
        assert selection.selection_index == 1
        assert selection.selection_reason == "fallback[0]"

    def test_select_second_fallback(self):
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=2,
        )
        assert selection.selected_model.name == "Gemini Pro"
        assert selection.selection_index == 2
        assert selection.selection_reason == "fallback[1]"

    def test_select_out_of_bounds(self):
        with pytest.raises(IndexError) as exc_info:
            select_model(
                primary_model="Claude Sonnet 4.5 (non-thinking)",
                fallback_chain=["Claude Sonnet"],
                selection_index=5,
            )
        assert "out of range" in str(exc_info.value)

    def test_select_negative_index(self):
        with pytest.raises(IndexError):
            select_model(
                primary_model="Claude Sonnet 4.5 (non-thinking)",
                fallback_chain=[],
                selection_index=-1,
            )

    def test_empty_chain_error(self):
        with pytest.raises(EmptyFallbackChainError) as exc_info:
            select_model(
                primary_model="",
                fallback_chain=[],
                selection_index=0,
            )
        assert "at least one model" in str(exc_info.value)

    def test_invalid_primary_model(self):
        with pytest.raises(InvalidModelError):
            select_model(
                primary_model="NonExistentModel",
                fallback_chain=[],
                selection_index=0,
            )

    def test_invalid_fallback_model(self):
        with pytest.raises(InvalidModelError):
            select_model(
                primary_model="Claude Sonnet 4.5 (non-thinking)",
                fallback_chain=["ValidModel", "InvalidModel"],
                selection_index=0,
            )

    def test_fallback_chain_none_is_empty_list(self):
        """Verify None fallback_chain is treated as empty list."""
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=None,
            selection_index=0,
        )
        assert selection.selected_model.name == "Claude Sonnet 4.5 (non-thinking)"


class TestDeterminism:
    """Test deterministic behavior - same inputs produce same outputs."""

    def test_chain_hash_determinism(self):
        """Same chain should produce same hash."""
        selection1 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=0,
        )
        selection2 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=0,
        )
        assert selection1.fallback_chain_hash == selection2.fallback_chain_hash

    def test_different_chain_different_hash(self):
        """Different chains should produce different hashes."""
        selection1 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet"],
            selection_index=0,
        )
        selection2 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Gemini Pro"],
            selection_index=0,
        )
        assert selection1.fallback_chain_hash != selection2.fallback_chain_hash

    def test_order_matters_for_hash(self):
        """Chain order should affect hash."""
        selection1 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=0,
        )
        selection2 = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Gemini Pro", "Claude Sonnet"],
            selection_index=0,
        )
        assert selection1.fallback_chain_hash != selection2.fallback_chain_hash

    def test_selection_determinism_across_runs(self):
        """Multiple selections with same inputs should be identical."""
        results = []
        for _ in range(10):
            selection = select_model(
                primary_model="Claude Sonnet 4.5 (non-thinking)",
                fallback_chain=["Claude Sonnet", "Gemini Pro"],
                selection_index=1,
            )
            results.append({
                "model_name": selection.selected_model.name,
                "model_id": selection.selected_model.model_id,
                "hash": selection.fallback_chain_hash,
            })

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result == first


class TestCreateRouterReceipt:
    """Test router receipt generation."""

    def test_create_receipt_basic(self):
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet"],
            selection_index=0,
        )

        receipt = create_router_receipt(
            selection=selection,
            task_id="test-task-001",
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet"],
        )

        assert receipt["task_id"] == "test-task-001"
        assert receipt["router_version"] == "1.0"
        assert receipt["primary_model"] == "Claude Sonnet 4.5 (non-thinking)"
        assert receipt["fallback_chain"] == ["Claude Sonnet"]
        assert "selection" in receipt
        assert receipt["selection"]["selection_index"] == 0

    def test_receipt_with_no_fallbacks(self):
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=None,
            selection_index=0,
        )

        receipt = create_router_receipt(
            selection=selection,
            task_id="test-task-002",
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=None,
        )

        assert receipt["fallback_chain"] == []


class TestModelSpecs:
    """Test ModelSpec dataclass behavior."""

    def test_model_spec_frozen(self):
        """ModelSpec should be immutable (frozen=True)."""
        spec = ModelSpec(
            name="Test Model",
            model_id="test-id",
            reasoning_level="Medium",
            thinking_mode=False,
        )

        with pytest.raises(AttributeError):
            spec.name = "Different Name"

    def test_model_spec_to_dict(self):
        spec = ModelSpec(
            name="Test Model",
            model_id="test-id",
            reasoning_level="High",
            thinking_mode=True,
        )

        result = spec.to_dict()
        assert result == {
            "name": "Test Model",
            "model_id": "test-id",
            "reasoning_level": "High",
            "thinking_mode": True,
        }


class TestKnownModelsRegistry:
    """Test the KNOWN_MODELS registry."""

    def test_registry_has_claude_models(self):
        assert "Claude Sonnet 4.5 (non-thinking)" in KNOWN_MODELS
        assert "Claude Sonnet" in KNOWN_MODELS
        assert "Claude Opus 4.5" in KNOWN_MODELS

    def test_registry_has_gpt_models(self):
        assert "GPT-5.2-Codex" in KNOWN_MODELS

    def test_registry_has_gemini_models(self):
        assert "Gemini Pro" in KNOWN_MODELS
        assert "Gemini 3 Pro (Low)" in KNOWN_MODELS

    def test_all_specs_are_frozen(self):
        """All ModelSpec instances should be frozen."""
        for spec in KNOWN_MODELS.values():
            with pytest.raises(AttributeError):
                spec.name = "Modified"


# Integration test combining multiple components
class TestIntegration:
    """Integration tests for full router workflow."""

    def test_full_workflow_primary_success(self):
        """Test complete workflow: validate -> select -> receipt."""
        # 1. Validate models
        primary_spec = validate_model("Claude Sonnet 4.5 (non-thinking)")
        fallback1_spec = validate_model("Claude Sonnet")

        assert primary_spec.name == "Claude Sonnet 4.5 (non-thinking)"
        assert fallback1_spec.name == "Claude Sonnet"

        # 2. Select model
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=0,
        )

        assert selection.selected_model.name == "Claude Sonnet 4.5 (non-thinking)"
        assert selection.selection_reason == "primary_model"

        # 3. Create receipt
        receipt = create_router_receipt(
            selection=selection,
            task_id="integration-test",
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
        )

        assert receipt["task_id"] == "integration-test"
        assert receipt["selection"]["selected_model"]["name"] == "Claude Sonnet 4.5 (non-thinking)"

    def test_full_workflow_fallback_selection(self):
        """Test workflow when falling back to secondary model."""
        # Select fallback[0]
        selection = select_model(
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
            selection_index=1,
        )

        assert selection.selected_model.name == "Claude Sonnet"
        assert selection.selection_reason == "fallback[0]"
        assert selection.selection_index == 1

        # Receipt should record the fallback
        receipt = create_router_receipt(
            selection=selection,
            task_id="fallback-test",
            primary_model="Claude Sonnet 4.5 (non-thinking)",
            fallback_chain=["Claude Sonnet", "Gemini Pro"],
        )

        assert receipt["selection"]["selection_reason"] == "fallback[0]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Tests for Model Weight Registry - Phase 5.1.3

Tests:
1. Model record creation with schema validation
2. Model registration and retrieval
3. ID determinism (name@version -> hash)
4. Description embedding generation
5. Semantic search by description
6. Deduplication by weights hash
7. Registry statistics
8. Integrity verification
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.model_registry import (
    create_model_record,
    register_model,
    get_model,
    get_model_by_id,
    get_model_by_weights_hash,
    search_models,
    list_models,
    get_registry_stats,
    verify_registry,
    _model_id,
    _hash_text,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_model_registry.db"


@pytest.fixture
def temp_weights_file(tmp_path):
    """Create a temporary weights file."""
    weights_path = tmp_path / "test_model.bin"
    # Write some dummy binary data
    weights_path.write_bytes(b"DUMMY_MODEL_WEIGHTS_" * 100)
    return weights_path


class TestModelIdDeterminism:
    """Tests for model ID generation."""

    def test_model_id_from_name_version(self):
        """Test that model ID is deterministic."""
        id1 = _model_id("all-MiniLM-L6-v2", "2.0.0")
        id2 = _model_id("all-MiniLM-L6-v2", "2.0.0")

        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex

    def test_different_versions_different_ids(self):
        """Test that different versions produce different IDs."""
        id1 = _model_id("bert-base", "1.0.0")
        id2 = _model_id("bert-base", "1.1.0")

        assert id1 != id2

    def test_different_names_different_ids(self):
        """Test that different names produce different IDs."""
        id1 = _model_id("bert-base", "1.0.0")
        id2 = _model_id("bert-large", "1.0.0")

        assert id1 != id2

    def test_id_is_hash_of_canonical_string(self):
        """Test that ID matches expected hash."""
        name = "test-model"
        version = "1.0.0"
        expected = _hash_text(f"{name}@{version}")

        assert _model_id(name, version) == expected


class TestCreateModelRecord:
    """Tests for model record creation."""

    def test_create_minimal_record(self):
        """Test creating a record with minimal fields."""
        record = create_model_record(
            name="test-model",
            version="1.0.0",
            description="A test model",
            format="pytorch"
        )

        assert record["name"] == "test-model"
        assert record["version"] == "1.0.0"
        assert record["description"] == "A test model"
        assert record["format"] == "pytorch"
        assert record["id"] == _model_id("test-model", "1.0.0")
        assert record["weights_hash"] is None
        assert record["size_bytes"] is None
        assert record["embedding"] is None

    def test_create_record_with_metadata(self):
        """Test creating a record with full metadata."""
        metadata = {
            "architecture": "transformer",
            "parameters": 22000000,
            "hidden_size": 384,
            "num_layers": 6,
            "license": "MIT",
        }

        record = create_model_record(
            name="minilm",
            version="2.0.0",
            description="Sentence transformer for semantic similarity",
            format="pytorch",
            weights_hash="abc123def456",
            size_bytes=90000000,
            metadata=metadata,
            source="huggingface"
        )

        assert record["weights_hash"] == "abc123def456"
        assert record["size_bytes"] == 90000000
        assert record["metadata"]["architecture"] == "transformer"
        assert record["metadata"]["parameters"] == 22000000
        assert record["source"] == "huggingface"

    def test_record_has_timestamps(self):
        """Test that record has timestamps."""
        record = create_model_record(
            name="test",
            version="1.0",
            description="Test",
            format="onnx"
        )

        assert "created_at" in record
        assert "updated_at" in record
        assert record["created_at"] == record["updated_at"]


class TestModelRegistration:
    """Tests for register_model function."""

    def test_register_model(self, temp_db_path):
        """Test registering a new model."""
        result = register_model(
            name="test-model",
            version="1.0.0",
            description="A test model for embeddings",
            format="pytorch",
            db_path=temp_db_path,
            embed_description=True,
            verbose=False
        )

        assert result["status"] == "registered"
        assert result["name"] == "test-model"
        assert result["version"] == "1.0.0"
        assert result["embedded"] is True
        assert "receipt" in result
        assert result["receipt"]["operation"] == "register"

    def test_register_with_weights_file(self, temp_db_path, temp_weights_file):
        """Test registering with actual weights file."""
        result = register_model(
            name="test-model-with-weights",
            version="1.0.0",
            description="Model with weights",
            format="pytorch",
            weights_path=temp_weights_file,
            db_path=temp_db_path,
            verbose=False
        )

        assert result["status"] == "registered"
        assert result["weights_hash"] is not None
        assert len(result["weights_hash"]) == 64  # SHA-256

    def test_register_updates_existing(self, temp_db_path):
        """Test that re-registering updates the record."""
        # First registration
        result1 = register_model(
            name="update-test",
            version="1.0.0",
            description="Original description",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        # Second registration (same name/version, different description)
        result2 = register_model(
            name="update-test",
            version="1.0.0",
            description="Updated description",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        # Should be same model ID
        assert result1["model_id"] == result2["model_id"]

        # Verify update
        model = get_model("update-test", "1.0.0", db_path=temp_db_path)
        assert model["description"] == "Updated description"


class TestModelRetrieval:
    """Tests for model retrieval functions."""

    def test_get_model_by_name_version(self, temp_db_path):
        """Test retrieving model by name and version."""
        register_model(
            name="retrieval-test",
            version="1.0.0",
            description="Test model",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        model = get_model("retrieval-test", "1.0.0", db_path=temp_db_path)

        assert model is not None
        assert model["name"] == "retrieval-test"
        assert model["version"] == "1.0.0"

    def test_get_model_latest_version(self, temp_db_path):
        """Test retrieving latest version when version not specified."""
        # Register two versions
        register_model(
            name="multi-version",
            version="1.0.0",
            description="Version 1",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="multi-version",
            version="2.0.0",
            description="Version 2",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        # Get without specifying version
        model = get_model("multi-version", db_path=temp_db_path)

        assert model is not None
        # Should return the latest (by created_at)
        assert model["version"] == "2.0.0"

    def test_get_model_by_id(self, temp_db_path):
        """Test retrieving model by ID."""
        result = register_model(
            name="id-test",
            version="1.0.0",
            description="Test",
            format="onnx",
            db_path=temp_db_path,
            verbose=False
        )

        model = get_model_by_id(result["model_id"], db_path=temp_db_path)

        assert model is not None
        assert model["name"] == "id-test"

    def test_get_model_by_weights_hash(self, temp_db_path, temp_weights_file):
        """Test retrieving model by weights hash (deduplication)."""
        result = register_model(
            name="weights-test",
            version="1.0.0",
            description="Test",
            format="pytorch",
            weights_path=temp_weights_file,
            db_path=temp_db_path,
            verbose=False
        )

        # Retrieve by weights hash
        model = get_model_by_weights_hash(
            result["weights_hash"],
            db_path=temp_db_path
        )

        assert model is not None
        assert model["name"] == "weights-test"

    def test_get_nonexistent_model(self, temp_db_path):
        """Test retrieving a model that doesn't exist."""
        model = get_model("nonexistent", "1.0.0", db_path=temp_db_path)
        assert model is None


class TestSemanticSearch:
    """Tests for search_models function."""

    def test_search_returns_results(self, temp_db_path):
        """Test that search returns relevant results."""
        # Register some models
        register_model(
            name="sentence-transformer",
            version="1.0.0",
            description="A transformer model for sentence embeddings and semantic similarity",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="image-classifier",
            version="1.0.0",
            description="A convolutional neural network for image classification",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        results = search_models("sentence embeddings", db_path=temp_db_path, top_k=3)

        assert len(results) > 0
        assert all("similarity" in r for r in results)
        # Sentence transformer should rank higher
        assert results[0]["name"] == "sentence-transformer"

    def test_search_ranking(self, temp_db_path):
        """Test that results are ranked by similarity."""
        # Register models with varying relevance
        register_model(
            name="bert-embeddings",
            version="1.0.0",
            description="BERT model for text embeddings",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="gpt-text",
            version="1.0.0",
            description="GPT model for text generation",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="resnet-images",
            version="1.0.0",
            description="ResNet model for image recognition",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        results = search_models("text embeddings", db_path=temp_db_path, top_k=5)

        # Results should be sorted by similarity descending
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_top_k_limit(self, temp_db_path):
        """Test that top_k limits results."""
        # Register multiple models
        for i in range(5):
            register_model(
                name=f"model-{i}",
                version="1.0.0",
                description=f"Test model number {i} for testing",
                format="pytorch",
                db_path=temp_db_path,
                verbose=False
            )

        results = search_models("test model", db_path=temp_db_path, top_k=2)

        assert len(results) <= 2

    def test_search_format_filter(self, temp_db_path):
        """Test filtering by model format."""
        register_model(
            name="pytorch-model",
            version="1.0.0",
            description="A PyTorch model",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="onnx-model",
            version="1.0.0",
            description="An ONNX model",
            format="onnx",
            db_path=temp_db_path,
            verbose=False
        )

        # Search only ONNX models
        results = search_models(
            "model",
            db_path=temp_db_path,
            format_filter="onnx"
        )

        assert all(r["format"] == "onnx" for r in results)


class TestListModels:
    """Tests for list_models function."""

    def test_list_all_models(self, temp_db_path):
        """Test listing all registered models."""
        # Register some models
        for name in ["model-a", "model-b", "model-c"]:
            register_model(
                name=name,
                version="1.0.0",
                description=f"Description for {name}",
                format="pytorch",
                db_path=temp_db_path,
                verbose=False
            )

        models = list_models(db_path=temp_db_path)

        assert len(models) == 3
        names = [m["name"] for m in models]
        assert "model-a" in names
        assert "model-b" in names
        assert "model-c" in names

    def test_list_filter_by_format(self, temp_db_path):
        """Test filtering list by format."""
        register_model(
            name="pytorch-only",
            version="1.0.0",
            description="PyTorch",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="onnx-only",
            version="1.0.0",
            description="ONNX",
            format="onnx",
            db_path=temp_db_path,
            verbose=False
        )

        pytorch_models = list_models(format_filter="pytorch", db_path=temp_db_path)
        onnx_models = list_models(format_filter="onnx", db_path=temp_db_path)

        assert len(pytorch_models) == 1
        assert len(onnx_models) == 1
        assert pytorch_models[0]["name"] == "pytorch-only"
        assert onnx_models[0]["name"] == "onnx-only"


class TestRegistryStats:
    """Tests for get_registry_stats function."""

    def test_stats_on_empty_registry(self, temp_db_path):
        """Test stats on non-existent registry."""
        stats = get_registry_stats(temp_db_path)
        assert stats["exists"] is False
        assert stats["total_models"] == 0

    def test_stats_after_registration(self, temp_db_path):
        """Test stats after registering models."""
        # Register models with different formats
        register_model(
            name="pytorch-1",
            version="1.0.0",
            description="PyTorch model 1",
            format="pytorch",
            source="huggingface",
            db_path=temp_db_path,
            verbose=False
        )

        register_model(
            name="onnx-1",
            version="1.0.0",
            description="ONNX model 1",
            format="onnx",
            source="local",
            db_path=temp_db_path,
            verbose=False
        )

        stats = get_registry_stats(temp_db_path)

        assert stats["exists"] is True
        assert stats["total_models"] == 2
        assert stats["embedded_models"] == 2
        assert stats["formats"]["pytorch"] == 1
        assert stats["formats"]["onnx"] == 1
        assert stats["sources"]["huggingface"] == 1
        assert stats["sources"]["local"] == 1
        assert stats["receipt_count"] >= 2


class TestRegistryVerification:
    """Tests for verify_registry function."""

    def test_verify_valid_registry(self, temp_db_path):
        """Test verification of valid registry."""
        register_model(
            name="valid-model",
            version="1.0.0",
            description="A valid test model",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        result = verify_registry(db_path=temp_db_path)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_verify_empty_registry(self, temp_db_path):
        """Test verification of non-existent registry."""
        result = verify_registry(db_path=temp_db_path)

        assert result["valid"] is False
        assert "Database does not exist" in result["errors"]


class TestReceipts:
    """Tests for receipt generation."""

    def test_registration_emits_receipt(self, temp_db_path):
        """Test that registration creates a receipt."""
        result = register_model(
            name="receipt-test",
            version="1.0.0",
            description="Test",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        assert "receipt" in result
        assert result["receipt"]["operation"] == "register"
        assert "receipt_hash" in result["receipt"]
        assert len(result["receipt"]["receipt_hash"]) == 64


class TestEmbeddingIntegration:
    """Tests for embedding integration."""

    def test_embedding_stored_with_model(self, temp_db_path):
        """Test that embedding is stored with model."""
        register_model(
            name="embed-test",
            version="1.0.0",
            description="A model for testing embedding storage",
            format="pytorch",
            embed_description=True,
            db_path=temp_db_path,
            verbose=False
        )

        model = get_model("embed-test", "1.0.0", db_path=temp_db_path)

        assert model["embedding"] is not None
        assert model["dimensions"] == 384

    def test_embedding_dimensions(self, temp_db_path):
        """Test that embeddings have correct dimensions."""
        import numpy as np

        register_model(
            name="dim-test",
            version="1.0.0",
            description="Testing embedding dimensions",
            format="pytorch",
            db_path=temp_db_path,
            verbose=False
        )

        model = get_model("dim-test", "1.0.0", db_path=temp_db_path)

        embedding = np.frombuffer(model["embedding"], dtype=np.float32)
        assert len(embedding) == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

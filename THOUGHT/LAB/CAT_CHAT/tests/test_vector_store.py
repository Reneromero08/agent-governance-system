import pytest
from pathlib import Path
from catalytic_chat.experimental.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    store = VectorStore(db_path=db_path)
    yield store
    store.close()


def test_vector_put_get_roundtrip(temp_db):
    content = b"test content"
    vector = [0.1, 0.2, 0.3, 0.4]
    meta = {"key": "value", "num": 42}
    
    vector_id = temp_db.put_vector("test_ns", content, vector, meta)
    
    result = temp_db.get_vector(vector_id)
    
    assert result is not None
    assert result["vector_id"] == vector_id
    assert result["namespace"] == "test_ns"
    assert result["dims"] == 4
    assert result["vector"] == vector
    assert result["meta"] == meta
    assert "created_at" in result


def test_vector_query_topk_deterministic(temp_db):
    import math
    
    query_vector = [1.0, 0.0, 0.0, 0.0]
    
    vectors = [
        (b"a", [1.0, 0.0, 0.0, 0.0], {"id": "a"}),
        (b"b", [0.9, 0.1, 0.0, 0.0], {"id": "b"}),
        (b"c", [0.0, 1.0, 0.0, 0.0], {"id": "c"}),
        (b"d", [0.0, 0.0, 1.0, 0.0], {"id": "d"}),
    ]
    
    for content, vec, meta in vectors:
        temp_db.put_vector("test_ns", content, vec, meta)
    
    results = temp_db.query_topk("test_ns", query_vector, k=3)
    
    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]
    assert results[0]["meta"]["id"] == "a"
    
    for r in results:
        assert "vector_id" in r
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.0


def test_vector_reject_bad_dims(temp_db):
    with pytest.raises(VectorStoreError, match="empty"):
        temp_db.put_vector("test_ns", b"content", [], {})
    
    vid = temp_db.put_vector("test_ns", b"content", [0.1], {})
    assert vid is not None


def test_vector_namespace_isolation(temp_db):
    content1 = b"content 1"
    content2 = b"content 2"
    vector = [0.1, 0.2, 0.3]
    meta1 = {"ns": "one"}
    meta2 = {"ns": "two"}
    
    id1 = temp_db.put_vector("ns1", content1, vector, meta1)
    id2 = temp_db.put_vector("ns2", content2, vector, meta2)
    
    results_ns1 = temp_db.query_topk("ns1", vector, k=10)
    results_ns2 = temp_db.query_topk("ns2", vector, k=10)
    
    assert len(results_ns1) == 1
    assert len(results_ns2) == 1
    assert results_ns1[0]["vector_id"] == id1
    assert results_ns2[0]["vector_id"] == id2
    assert results_ns1[0]["meta"]["ns"] == "one"
    assert results_ns2[0]["meta"]["ns"] == "two"


def test_vector_query_empty_namespace(temp_db):
    results = temp_db.query_topk("empty_ns", [1.0, 0.0], k=5)
    assert results == []


def test_vector_get_nonexistent(temp_db):
    result = temp_db.get_vector("nonexistent_id")
    assert result is None


def test_vector_replace_existing(temp_db):
    content = b"test content"
    vector1 = [0.1, 0.2, 0.3]
    vector2 = [0.9, 0.8, 0.7]
    meta1 = {"v": 1}
    meta2 = {"v": 2}
    
    vector_id = temp_db.put_vector("test_ns", content, vector1, meta1)
    
    same_id = temp_db.put_vector("test_ns", content, vector2, meta2)
    
    assert vector_id == same_id
    
    result = temp_db.get_vector(vector_id)
    assert result["vector"] == vector2
    assert result["meta"] == meta2


def test_vector_delete_namespace(temp_db):
    temp_db.put_vector("ns1", b"a", [1.0], {})
    temp_db.put_vector("ns1", b"b", [2.0], {})
    temp_db.put_vector("ns2", b"c", [3.0], {})
    
    deleted = temp_db.delete_namespace("ns1")
    assert deleted == 2
    
    assert len(temp_db.query_topk("ns1", [1.0], k=10)) == 0
    assert len(temp_db.query_topk("ns2", [1.0], k=10)) == 1


def test_vector_context_manager(tmp_path):
    db_path = tmp_path / "test_ctx.db"
    
    with VectorStore(db_path=db_path) as store:
        vid = store.put_vector("ctx_ns", b"ctx", [1.0], {"ctx": True})
        result = store.get_vector(vid)
        assert result is not None

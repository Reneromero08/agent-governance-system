from pathlib import Path
import sys
import pytest
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.cas_store import CatalyticStore
from CAPABILITY.PRIMITIVES.hash_toolbelt import (
    hash_ast,
    hash_describe,
    hash_read_text,
)
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


def _make_test_writer(project_root: Path) -> GuardedWriter:
    """Create a GuardedWriter configured for test temp directories."""
    writer = GuardedWriter(
        project_root=project_root,
        tmp_roots=["_tmp"],
        durable_roots=["cas"],  # Allow CAS dir
        exclusions=[],  # No exclusions in test mode
    )
    writer.open_commit_gate()  # Tests need durable writes enabled
    return writer


@pytest.fixture
def store(tmp_path):
    test_writer = _make_test_writer(tmp_path)
    s = CatalyticStore(tmp_path / "cas", writer=test_writer)
    return s

def test_hash_ast_identity(store):
    code = "def foo(): pass"
    h = store.put_bytes(code.encode('utf-8'))
    
    # hash_ast returns a JSON string describing the AST outline
    res1 = hash_ast(store=store, hash_hex=h)
    res2 = hash_ast(store=store, hash_hex=h)
    assert res1 == res2
    assert h in res1

def test_hash_read_text_matches(store):
    content = "hello world"
    h = store.put_bytes(content.encode('utf-8'))
    
    res = hash_read_text(store=store, hash_hex=h)
    assert content in res
    assert h in res
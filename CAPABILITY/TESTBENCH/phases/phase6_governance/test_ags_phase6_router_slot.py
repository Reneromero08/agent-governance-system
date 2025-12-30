from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports from new structure
from CAPABILITY.PRIMITIVES import (
    cas_store, hash_toolbelt, ledger, merkle, restore_proof, verify_bundle, fs_guard, skills
)
from CAPABILITY.PIPELINES import (
    pipeline_chain, pipeline_dag, pipeline_runtime, pipeline_verify, swarm_runtime
)

def test_imports_ok():
    # This test primarily checks that the imports above work.
    assert True

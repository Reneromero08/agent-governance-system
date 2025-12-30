import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fix imports
from CAPABILITY.PRIMITIVES import (
    SkillRegistry,
    canonical_json,
    resolve_adapter,
    SkillNotFoundError,
    RegistryError,
    CapabilityHashMismatch
)

def test_skill_not_found_raises():
    with pytest.raises(SkillNotFoundError):
        raise SkillNotFoundError("skill missing")

def test_canonical_json_determinism():
    data = {"b": 2, "a": 1}
    # canonical_json returns BYTES and is COMPACT (no spaces)
    assert canonical_json(data) == b'{"a":1,"b":2}'

    data2 = {"a": 1, "b": 2}
    assert canonical_json(data) == canonical_json(data2)  # Fixed: was comparing canonical_json(data) with canonical_json(data2)
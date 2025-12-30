from pathlib import Path
import sys
import pytest
import json
from typing import Dict, Any, List
from jsonschema import Draft7Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Schemas and Fixtures paths
SCHEMAS_DIR = REPO_ROOT / "LAW" / "SCHEMAS"
FIXTURES_VALID = REPO_ROOT / "LAW" / "CONTRACTS" / "fixtures" / "catalytic" / "phase0" / "valid"
FIXTURES_INVALID = REPO_ROOT / "LAW" / "CONTRACTS" / "fixtures" / "catalytic" / "phase0" / "invalid"

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))

@pytest.fixture
def validators():
    # Helper to load schema if it exists
    def get_v(name):
        p = SCHEMAS_DIR / f"{name}.schema.json"
        if not p.exists():
            pytest.skip(f"Schema {name} missing at {p}")
        return Draft7Validator(_load_json(p))

    return {
        "jobspec": get_v("jobspec"),
        "ledger": get_v("ledger"),
        "proof": get_v("proof"),
        "validation_error": get_v("validation_error")
    }

def choose_validator(doc: Dict[str, Any], vals: Dict[str, Draft7Validator]) -> Draft7Validator:
    if "job_id" in doc and "task_type" in doc:
        return vals["jobspec"]
    if "valid" in doc and "errors" in doc and "warnings" in doc:
        return vals["validation_error"]
    if "RUN_INFO" in doc and "PRE_MANIFEST" in doc and "POST_MANIFEST" in doc:
        return vals["ledger"]
    if "proof_version" in doc and "restoration_result" in doc:
        return vals["proof"]
    pytest.fail(f"Fixture does not match any known schema: {list(doc.keys())}")

def test_valid_fixtures(validators):
    if not FIXTURES_VALID.exists():
        pytest.skip(f"Fixtures dir missing: {FIXTURES_VALID}")
    files = list(FIXTURES_VALID.glob("*.json"))
    if not files:
        pytest.skip("No valid fixtures found")
    for p in files:
        doc = _load_json(p)
        v = choose_validator(doc, validators)
        errors = list(v.iter_errors(doc))
        if errors:
            pytest.fail(f"Valid fixture {p.name} failed: {errors[0].message}")

def test_invalid_fixtures(validators):
    if not FIXTURES_INVALID.exists():
        pytest.skip(f"Fixtures dir missing: {FIXTURES_INVALID}")
    files = list(FIXTURES_INVALID.glob("*.json"))
    if not files:
        pytest.skip("No invalid fixtures found")
    for p in files:
        doc = _load_json(p)
        v = choose_validator(doc, validators)
        errors = list(v.iter_errors(doc))
        if not errors:
            pytest.fail(f"Invalid fixture {p.name} unexpectedly passed")
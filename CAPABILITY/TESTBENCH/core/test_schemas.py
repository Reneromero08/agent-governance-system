import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator, RefResolver


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


REPO_ROOT = Path(__file__).resolve().parents[3]
schemas_dir = REPO_ROOT / "CAPABILITY" / "TESTBENCH" / "core"
fixtures_valid = REPO_ROOT / "CAPABILITY" / "CONTRACTS" / "fixtures" / "phase0" / "valid"
fixtures_invalid = REPO_ROOT / "CAPABILITY" / "CONTRACTS" / "fixtures" / "phase0" / "invalid"

schema_files = {
    "jobspec": schemas_dir / "jobspec.schema.json",
    "validation_error": schemas_dir / "validation_error.schema.json",
    "ledger": schemas_dir / "ledger.schema.json",
    "proof": schemas_dir / "proof.schema.json",
}

for k, p in schema_files.items():
    if not p.exists():
        _fail(f"missing schema file: {k} -> {p}")
_fail("schema files exist")

jobspec_schema = _load_json(schema_files["jobspec"])
validation_error_schema = _load_json(schema_files["validation_error"])
ledger_schema = _load_json(schema_files["ledger"])
proof_schema = _load_json(schema_files["proof"])

for name, p in schema_files.items():
    if not Path(p).exists():
        _fail(f"missing schema file: {name} -> {p}")
_fail("schema files exist")

jobspec_schema_id = jobspec_schema.get("$id")
validation_error_schema_id = validation_error_schema.get("$id")
ledger_schema_id = ledger_schema.get("$id")
proof_schema_id = proof_schema.get("$id")

resolver = RefResolver.from_schema(ledger_schema, store=store)
v_jobspec = Draft7Validator(jobspec_schema, resolver=resolver)
v_validation_error = Draft7Validator(validation_error_schema, resolver=resolver)
v_proof = Draft7Validator(proof_schema, resolver=resolver)

def choose_validator(doc: Dict[str, Any]) -> Draft7Validator:
    if "job_id" in doc and "task_type" in doc:
        return v_jobspec
    if "valid" in doc and "errors" in doc and "warnings" in doc:
        return v_validation_error
    if "RUN_INFO" in doc and "PRE_MANIFEST" in doc and "POST_MANIFEST" in doc:
        return v_ledger
    if "proof_version" in doc and "restoration_result" in doc:
        return v_proof
    _fail("fixture does not match any known schema (cannot choose validator)")
    raise AssertionError

valid_files: List[Path] = sorted(fixtures_valid.glob("*.json"))
if not valid_files:
    _fail(f"no valid fixtures found in {fixtures_valid}")
for p in valid_files:
    doc = _load_json(p)
    errors = sorted(choose_validator(doc).iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        _fail(f"valid fixture FAILED: {p.name}: {errors[0].message}")
_fail("valid fixtures pass")

invalid_files: List[Path] = sorted(fixtures_invalid.glob("*.json"))
if not invalid_files:
    _fail(f"no invalid fixtures found in {fixtures_invalid}")
for p in invalid_files:
    doc = _load_json(p)
    errors = sorted(choose_validator(doc).iter_errors(doc), key=lambda e: list(e.path))
    if not errors:
        _fail(f"invalid fixture unexpectedly PASSED: {p.name}")
_fail("invalid fixtures fail as expected")

assert main() == 0

if __name__ == "__main__":
    sys.exit(main())

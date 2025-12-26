from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, RefResolver


def _canonical_json_line(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


@dataclass(frozen=True)
class _LedgerSchemas:
    ledger: dict[str, Any]
    jobspec: dict[str, Any]


def _load_schemas() -> _LedgerSchemas:
    cat_dpt_root = Path(__file__).resolve().parents[1]
    schemas_dir = cat_dpt_root / "SCHEMAS"
    ledger_schema_path = schemas_dir / "ledger.schema.json"
    jobspec_schema_path = schemas_dir / "jobspec.schema.json"
    ledger_schema = json.loads(ledger_schema_path.read_text(encoding="utf-8"))
    jobspec_schema = json.loads(jobspec_schema_path.read_text(encoding="utf-8"))
    return _LedgerSchemas(ledger=ledger_schema, jobspec=jobspec_schema)


def _build_validator() -> Draft7Validator:
    schemas = _load_schemas()

    cat_dpt_root = Path(__file__).resolve().parents[1]
    schemas_dir = cat_dpt_root / "SCHEMAS"
    ledger_schema_path = (schemas_dir / "ledger.schema.json").resolve()
    jobspec_schema_path = (schemas_dir / "jobspec.schema.json").resolve()

    ledger_uri = ledger_schema_path.as_uri()
    jobspec_uri = jobspec_schema_path.as_uri()

    store: dict[str, Any] = {
        schemas.ledger.get("$id", "ledger.schema.json"): schemas.ledger,
        schemas.jobspec.get("$id", "jobspec.schema.json"): schemas.jobspec,
        "ledger.schema.json": schemas.ledger,
        "ledger.schema.json#": schemas.ledger,
        ledger_uri: schemas.ledger,
        ledger_uri + "#": schemas.ledger,
        "jobspec.schema.json": schemas.jobspec,
        "jobspec.schema.json#": schemas.jobspec,
        jobspec_uri: schemas.jobspec,
        jobspec_uri + "#": schemas.jobspec,
    }

    resolver = RefResolver.from_schema(schemas.ledger, store=store)
    return Draft7Validator(schemas.ledger, resolver=resolver)


_LEDGER_VALIDATOR = _build_validator()


class Ledger:
    """
    Append-only JSONL ledger. Each line MUST be a full object conforming to `SCHEMAS/ledger.schema.json`.

    Notes on determinism:
    - Ledger never generates timestamps. `RUN_INFO.timestamp` must be supplied by the caller deterministically.

    Notes on durability:
    - `append()` calls `os.fsync()` to ensure the appended line is flushed to disk.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._expected_size: int | None = None

    def append(self, record: dict[str, Any]) -> None:
        self._validate_record(record)

        line = _canonical_json_line(record)
        prior_size = self.path.stat().st_size if self.path.exists() else 0

        if self._expected_size is not None and prior_size < self._expected_size:
            raise RuntimeError("attempted non-append write detected (file truncated or rewritten)")

        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = 0
            while written < len(line):
                n = os.write(fd, line[written:])
                if n <= 0:
                    raise RuntimeError("failed to append ledger line")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)

        after_size = self.path.stat().st_size
        if after_size != prior_size + len(line):
            raise RuntimeError("attempted non-append write detected (unexpected size delta)")
        self._expected_size = after_size

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(self.path, "rb") as f:
            for idx, raw_line in enumerate(f, start=1):
                if not raw_line.endswith(b"\n"):
                    raise ValueError(f"partial ledger line at {idx}")
                line = raw_line[:-1]
                if line == b"":
                    raise ValueError(f"empty ledger line at {idx}")
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception as e:
                    raise ValueError(f"invalid JSON at line {idx}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"ledger line {idx} is not an object")
                records.append(obj)
        return records

    def verify_append_only(self) -> bool:
        if not self.path.exists():
            return True

        data = self.path.read_bytes()
        if data == b"":
            return True
        if not data.endswith(b"\n"):
            return False

        for idx, raw_line in enumerate(data.splitlines(keepends=True), start=1):
            if raw_line == b"\n":
                return False
            if not raw_line.endswith(b"\n"):
                return False
            try:
                obj = json.loads(raw_line[:-1].decode("utf-8"))
            except Exception:
                return False
            if not isinstance(obj, dict):
                return False
        return True

    def _validate_record(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        errors = sorted(_LEDGER_VALIDATOR.iter_errors(record), key=lambda e: e.path)
        if errors:
            # Keep deterministic, minimal error reporting (no dependency on validator formatting).
            first = errors[0]
            raise ValueError(f"ledger schema invalid at {list(first.path)}: {first.message}")

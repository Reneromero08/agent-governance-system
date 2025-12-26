from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_HEX64 = "0123456789abcdef"


def _is_hex64(s: str) -> bool:
    return len(s) == 64 and all(ch in _HEX64 for ch in s)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


class _DupeKeyError(ValueError):
    pass


def _loads_no_dupes(raw: str) -> Any:
    def hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                raise _DupeKeyError(k)
            out[k] = v
        return out

    return json.loads(raw, object_pairs_hook=hook)


@dataclass(frozen=True)
class RegistryValidation:
    ok: bool
    code: str
    details: Dict[str, Any]


def validate_capabilities_registry(path: Path) -> RegistryValidation:
    """
    Validates CAPABILITIES.json with fail-closed, deterministic semantics.

    Error codes:
    - REGISTRY_DUPLICATE_HASH: duplicate capability hash key
    - REGISTRY_NONCANONICAL: file bytes not canonical JSON
    - REGISTRY_INVALID: schema/shape invalid
    - REGISTRY_TAMPERED: internal hash/link mismatch
    """
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except Exception as e:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": f"invalid utf-8: {e}"})

    try:
        obj = _loads_no_dupes(raw_text)
    except _DupeKeyError as e:
        return RegistryValidation(False, "REGISTRY_DUPLICATE_HASH", {"where": "object_key", "key": str(e)})
    except Exception as e:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": f"invalid json: {e}"})

    if not isinstance(obj, dict):
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "expected object"})

    allowed_top = {"registry_version", "capabilities"}
    if set(obj.keys()) != allowed_top:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "unexpected top-level fields"})

    if obj.get("registry_version") != "1.0.0":
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "registry_version must be 1.0.0"})

    caps = obj.get("capabilities")
    if not isinstance(caps, dict):
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "capabilities must be object"})

    # Canonical bytes check (sorted keys, stable separators).
    # Allow a single trailing LF at EOF to avoid editor friction.
    canonical = _canonical_json_bytes(obj)
    raw_cmp = raw_bytes[:-1] if raw_bytes.endswith(b"\n") else raw_bytes
    if raw_cmp != canonical:
        return RegistryValidation(False, "REGISTRY_NONCANONICAL", {"path": str(path)})

    # Deterministic ordering: keys must be sorted by capability_hash.
    keys = list(caps.keys())
    if keys != sorted(keys):
        return RegistryValidation(False, "REGISTRY_NONCANONICAL", {"message": "capability hashes not sorted"})

    for cap_hash, entry in caps.items():
        if not (isinstance(cap_hash, str) and _is_hex64(cap_hash)):
            return RegistryValidation(False, "REGISTRY_INVALID", {"message": "invalid capability hash key"})
        if not isinstance(entry, dict):
            return RegistryValidation(False, "REGISTRY_INVALID", {"capability_hash": cap_hash, "message": "entry not object"})
        if set(entry.keys()) != {"adapter_spec_hash", "adapter"}:
            return RegistryValidation(False, "REGISTRY_INVALID", {"capability_hash": cap_hash, "message": "unexpected entry fields"})
        spec_hash = entry.get("adapter_spec_hash")
        adapter = entry.get("adapter")
        if not (isinstance(spec_hash, str) and _is_hex64(spec_hash)):
            return RegistryValidation(False, "REGISTRY_INVALID", {"capability_hash": cap_hash, "message": "invalid adapter_spec_hash"})
        if not isinstance(adapter, dict):
            return RegistryValidation(False, "REGISTRY_INVALID", {"capability_hash": cap_hash, "message": "adapter not object"})
        computed = hashlib.sha256(_canonical_json_bytes(adapter)).hexdigest()
        if computed != cap_hash or spec_hash != computed:
            return RegistryValidation(False, "REGISTRY_TAMPERED", {"capability_hash": cap_hash})

    return RegistryValidation(True, "OK", {"path": str(path), "count": len(caps)})


def validate_capability_pins(path: Path) -> RegistryValidation:
    """
    Validates CAPABILITY_PINS.json with fail-closed, deterministic semantics.

    Error codes:
    - REGISTRY_DUPLICATE_HASH: duplicate entry in allowed_capabilities
    - REGISTRY_NONCANONICAL: file bytes not canonical JSON or list not sorted
    - REGISTRY_INVALID: schema/shape invalid
    """
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except Exception as e:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": f"invalid utf-8: {e}"})

    try:
        obj = _loads_no_dupes(raw_text)
    except _DupeKeyError as e:
        return RegistryValidation(False, "REGISTRY_INVALID", {"where": "object_key", "key": str(e)})
    except Exception as e:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": f"invalid json: {e}"})

    if not isinstance(obj, dict):
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "expected object"})

    allowed_top = {"pins_version", "allowed_capabilities"}
    if set(obj.keys()) != allowed_top:
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "unexpected top-level fields"})

    if obj.get("pins_version") != "1.0.0":
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "pins_version must be 1.0.0"})

    allowed = obj.get("allowed_capabilities")
    if not isinstance(allowed, list):
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "allowed_capabilities must be array"})
    if not all(isinstance(x, str) and _is_hex64(x) for x in allowed):
        return RegistryValidation(False, "REGISTRY_INVALID", {"message": "allowed_capabilities entries must be hex64 strings"})

    canonical = _canonical_json_bytes(obj)
    raw_cmp = raw_bytes[:-1] if raw_bytes.endswith(b"\n") else raw_bytes
    if raw_cmp != canonical:
        return RegistryValidation(False, "REGISTRY_NONCANONICAL", {"path": str(path)})

    if allowed != sorted(allowed):
        return RegistryValidation(False, "REGISTRY_NONCANONICAL", {"message": "allowed_capabilities not sorted"})

    if len(set(allowed)) != len(allowed):
        return RegistryValidation(False, "REGISTRY_DUPLICATE_HASH", {"where": "allowed_capabilities"})

    return RegistryValidation(True, "OK", {"path": str(path), "count": len(allowed)})

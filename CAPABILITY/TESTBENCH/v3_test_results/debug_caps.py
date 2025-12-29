import sys
from pathlib import Path
import json
import hashlib

# Mocking necessary parts from registry_validators.py
_HEX64 = "0123456789abcdefABCDEF"

def _is_hex64(s):
    return len(s) == 64 and all(ch in _HEX64 for ch in s)

def _canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

class _DupeKeyError(ValueError):
    pass

def _loads_no_dupes(raw):
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise _DupeKeyError(k)
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=hook)

def validate(path):
    print(f"Validating {path}...")
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except Exception as e:
        print(f"FAIL: invalid utf-8: {e}")
        return

    try:
        obj = _loads_no_dupes(raw_text)
    except _DupeKeyError as e:
        print(f"FAIL: DUPLICATE_HASH: {e}")
        return
    except Exception as e:
        print(f"FAIL: invalid json: {e}")
        return

    if not isinstance(obj, dict):
        print("FAIL: expected object")
        return

    canonical = _canonical_json_bytes(obj)
    raw_cmp = raw_bytes[:-1] if raw_bytes.endswith(b"\n") else raw_bytes
    if raw_cmp != canonical:
        print("FAIL: NONCANONICAL")
        print(f"Expected len: {len(canonical)}, Got: {len(raw_cmp)}")
        # Show diff if small
        if len(canonical) < 1000:
             print(f"Canonical: {canonical}")
             print(f"Raw: {raw_cmp}")
        return

    caps = obj.get("capabilities")
    keys = list(caps.keys())
    if keys != sorted(keys):
        print("FAIL: NONCANONICAL (unsorted keys)")
        return

    for cap_hash, entry in caps.items():
        adapter = entry.get("adapter")
        spec_hash = entry.get("adapter_spec_hash")
        computed = hashlib.sha256(_canonical_json_bytes(adapter)).hexdigest()
        if computed != cap_hash:
            print(f"FAIL: TAMPERED: computed {computed} != key {cap_hash}")
            return
        if spec_hash != computed:
            print(f"FAIL: TAMPERED: spec_hash {spec_hash} != computed {computed}")
            return

    print("OK")

if __name__ == "__main__":
    validate(Path("LAW/CANON/CAPABILITIES.json"))
    validate(Path("LAW/CANON/CAPABILITY_PINS.json"))

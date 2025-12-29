import json
import hashlib

def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

adapter = {
    "adapter_version": "1.0.0",
    "artifacts": {
        "domain_roots": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "ledger": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "proof": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
    },
    "command": [
        "python",
        "CATALYTIC-DPT/SKILLS/ant-worker/scripts/run.py",
        "CONTRACTS/_runs/_tmp/phase65_registry/task.json",
        "CONTRACTS/_runs/_tmp/phase65_registry/result.json"
    ],
    "deref_caps": {"max_bytes": 1024, "max_depth": 2, "max_matches": 1, "max_nodes": 10},
    "inputs": {
        "CONTRACTS/_runs/_tmp/phase65_registry/in.txt": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    "jobspec": {
        "catalytic_domains": ["CONTRACTS/_runs/_tmp/phase65_registry/domain"],
        "determinism": "deterministic",
        "inputs": {},
        "intent": "capability: ant-worker copy (Phase 6.5)",
        "job_id": "cap-ant-worker-copy-v1",
        "outputs": {
            "durable_paths": ["CONTRACTS/_runs/_tmp/phase65_registry/out.txt"],
            "validation_criteria": {}
        },
        "phase": 6,
        "task_type": "adapter_execution"
    },
    "name": "ant-worker-copy-v1",
    "outputs": {
        "CONTRACTS/_runs/_tmp/phase65_registry/out.txt": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    "side_effects": {
        "clock": False,
        "filesystem_unbounded": False,
        "network": False,
        "nondeterministic": False
    }
}

cap_hash = hashlib.sha256(canonical_json_bytes(adapter)).hexdigest()
print(f"Computed Hash: {cap_hash}")

registry = {
    "registry_version": "1.0.0",
    "capabilities": {
        cap_hash: {
            "adapter_spec_hash": cap_hash,
            "adapter": adapter
        }
    }
}

with open("CATALYTIC-DPT/CAPABILITIES.json", "wb") as f:
    f.write(canonical_json_bytes(registry))
    f.write(b"\n")

pins = {
    "pins_version": "1.0.0",
    "allowed_capabilities": [cap_hash]
}
with open("CATALYTIC-DPT/CAPABILITY_PINS.json", "wb") as f:
    f.write(canonical_json_bytes(pins))
    f.write(b"\n")

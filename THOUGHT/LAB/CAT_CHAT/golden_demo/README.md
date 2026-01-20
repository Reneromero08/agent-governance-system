# CAT_CHAT Golden Demo

This demo shows the core CAT_CHAT deterministic execution pipeline.

## Quick Start (Windows PowerShell)

```powershell
cd "D:\CCC 2.0\AI\agent-governance-system"
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
python THOUGHT\LAB\CAT_CHAT\golden_demo\golden_demo.py
```

## Prerequisites

- Python 3.9+
- jsonschema package (`pip install jsonschema`)

## What the Demo Shows

1. **Bundle Creation**: Deterministic packaging of content into a self-contained bundle
2. **Bundle Verification**: Hash integrity checks (bundle_id, root_hash, artifacts)
3. **Bundle Execution**: Running the bundle and generating a receipt
4. **Receipt Verification**: Verifying the receipt hash computation
5. **Determinism Check**: Re-running to prove identical outputs

## Expected Output

```
============================================================
CAT_CHAT GOLDEN DEMO - Deterministic Execution
============================================================

This demo shows the core CAT_CHAT pipeline:
  - Bundle creation (deterministic packaging)
  - Bundle verification (hash integrity)
  - Bundle execution (receipt generation)
  - Receipt verification (chain integrity)

[1/5] Creating temporary workspace...
  Workspace: C:\Users\...\cat_chat_demo_...

[2/5] Creating demo bundle...
  Bundle ID: a1b2c3d4e5f6...
  Root Hash: f6e5d4c3b2a1...
  Artifacts: 1

[3/5] Verifying bundle integrity...
  Status: SUCCESS
  Bundle ID verified: a1b2c3d4e5f6...
  Root hash verified: f6e5d4c3b2a1...

[4/5] Executing bundle and generating receipt...
  Outcome: SUCCESS
  Receipt Hash: 1234567890ab...
  Receipt Path: ...\receipt.json

[5/5] Verifying receipt integrity...
  Hash Match: True
  Stored:   1234567890ab...
  Computed: 1234567890ab...

============================================================
DEMO COMPLETE - Summary
============================================================
Bundle ID:     a1b2c3d4e5f6...
Root Hash:     f6e5d4c3b2a1...
Receipt Hash:  1234567890ab...
Outcome:       SUCCESS

All verifications PASSED.
The system is deterministic and fail-closed.

============================================================
DETERMINISM CHECK
============================================================
Re-running bundle creation to verify identical hashes...
  Bundle ID: IDENTICAL
  Root Hash: IDENTICAL

Determinism verified - same inputs produce identical outputs.
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All verifications passed |
| 1 | Verification failed |
| 2 | Invalid input (missing dependency, PYTHONPATH) |
| 3 | Internal error |

## Troubleshooting

### ModuleNotFoundError: No module named 'catalytic_chat'

Set PYTHONPATH before running:

```powershell
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
```

### ModuleNotFoundError: No module named 'jsonschema'

Install jsonschema:

```powershell
pip install jsonschema
```

## Files

```
golden_demo/
  golden_demo.py          # Main demo script
  README.md               # This file
  fixtures/
    demo_content.txt      # Sample content
    demo_trust_policy.json  # Example trust policy
```

## Learn More

- [CAT_CHAT Usage Guide](../docs/CAT_CHAT_USAGE_GUIDE.md)
- [Specifications](../docs/specs/SPEC_INDEX.md)
- [Bundle Spec](../docs/specs/BUNDLE_SPEC.md)
- [Receipt Spec](../docs/specs/RECEIPT_SPEC.md)

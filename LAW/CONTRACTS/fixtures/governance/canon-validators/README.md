# Canon Validators Fixture

This fixture validates canon files for compliance with governance rules.

## What it checks

1. **Duplicate rule numbers**: Ensures no duplicate numbered items within the same section
2. **Line count limits**: Warns when files approach 250 lines, errors at 300+ lines (per INV-009)
3. **Authority gradient**: Validates consistency with canon.json

## How to run

```bash
cd "LAW/CONTRACTS/fixtures/governance/canon-validators"
python validate_canon.py
```

Or from repo root:
```bash
python LAW/CONTRACTS/fixtures/governance/canon-validators/validate_canon.py
```

## Exit codes

- `0`: All checks passed
- `1`: Validation errors found

## Output

The script generates `output.json` with validation results:

```json
{
  "status": "passed|failed",
  "checks": ["duplicate_numbers", "line_count", "authority_gradient"],
  "failures": {
    "check_name": ["error messages..."]
  }
}
```

## Current status

As of creation, the following canon files exceed the 300-line limit:
- CATALYTIC/CMP-01_CATALYTIC_MUTATION_PROTOCOL.md (351 lines)
- CATALYTIC/SPECTRUM-04_IDENTITY_SIGNING.md (738 lines)
- CATALYTIC/SPECTRUM-05_VERIFICATION_LAW.md (507 lines)
- CATALYTIC/SPECTRUM-06_RESTORE_RUNNER.md (606 lines)
- DOCUMENT_POLICY.md (402 lines)
- STEWARDSHIP.md (374 lines)

These files should be split per INV-009 requirements.

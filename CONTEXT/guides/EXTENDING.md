# Extending AGS

This guide explains how to extend the Agent Governance System.

## Adding a New Skill

1. **Create the skill directory:**
   ```
   SKILLS/my-skill/
   ├── SKILL.md      # Manifest (required)
   ├── run.py        # Entry point (required)
   ├── validate.py   # Validation script (required)
   └── fixtures/     # Test fixtures (required)
       └── basic/
           ├── input.json
           └── expected.json
   ```

2. **Define the manifest** (`SKILL.md`):
   - Purpose and triggers
   - Inputs and outputs
   - Required canon version
   - Constraints

3. **Implement `run.py`**:
   ```python
   def main(input_path: Path, output_path: Path) -> int:
       payload = json.loads(input_path.read_text())
       # ... process ...
       output_path.write_text(json.dumps(result))
       return 0  # success
   ```

4. **Add fixtures** that demonstrate correct behavior.

5. **Run tests**:
   ```bash
   python CONTRACTS/runner.py
   ```

## Adding a New Invariant

Invariants are frozen in v1.0. To add a new one:

1. File an ADR under `CONTEXT/decisions/`
2. Add to `CANON/INVARIANTS.md` with next number (INV-009, etc.)
3. Update `TOOLS/check_canon_governance.py` FROZEN_INVARIANTS list
4. Add a fixture that validates the invariant
5. Bump minor version

## Adding Contract Fixtures

1. Create directory under `CONTRACTS/fixtures/<category>/<name>/`
2. Add `input.json`, `expected.json`, and `run.py`
3. Runner auto-discovers fixtures

## Querying the Cortex

Skills must use the cortex API, not raw filesystem:

```python
from CORTEX.query import get_by_id, get_by_type

skill = get_by_id("skill:my-skill")
all_skills = get_by_type("skill")
```

# Testing AGS

This guide explains how to run and create tests.

## Running All Tests

```bash
# Run all contract and skill fixtures
python CONTRACTS/runner.py

# Run governance checks
python TOOLS/check_canon_governance.py

# Run critic (code quality)
python TOOLS/critic.py

# Run token linter
python TOOLS/lint_tokens.py
```

## Fixture Structure

Fixtures are the primary testing mechanism. Each fixture has:

```
fixtures/<name>/
├── input.json      # Input to the test
├── expected.json   # Expected output
├── run.py          # Execution script (optional for skills)
└── validate.py     # Custom validation (optional)
```

## Creating a New Fixture

1. **Create the directory:**
   ```bash
   mkdir -p CONTRACTS/fixtures/governance/my-check
   ```

2. **Define input.json:**
   ```json
   {
     "description": "What this tests",
     "test_data": "..."
   }
   ```

3. **Define expected.json:**
   ```json
   {
     "description": "What this tests",
     "test_data": "...",
     "result": "expected value"
   }
   ```

4. **Create run.py:**
   ```python
   def main(input_path, output_path):
       # Read input, process, write output
       return 0
   ```

## Skill Fixtures

Skills have fixtures under their own directory:

```
SKILLS/my-skill/fixtures/basic/
├── input.json
└── expected.json
```

The skill's `run.py` is used automatically.

## Validation

The runner compares actual output to expected.json. For custom validation, add `validate.py`:

```python
def main(actual_path, expected_path):
    actual = json.loads(actual_path.read_text())
    expected = json.loads(expected_path.read_text())
    return 0 if actual == expected else 1
```

## CI Integration

All tests run in CI via `.github/workflows/contracts.yml`.

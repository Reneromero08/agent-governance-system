# Verification: CAT_CHAT Context Window Management (Phase 3.2.1)

Logic implemented in: `THOUGHT/LAB/CAT_CHAT/catalytic_chat/context_assembler.py`
Tests implemented in: `THOUGHT/LAB/CAT_CHAT/tests/test_context_assembler.py`

## How to Run Tests

From the repository root (`D:\CCC 2.0\AI\agent-governance-system`):

```powershell
$env:PYTHONPATH = "THOUGHT\LAB\CAT_CHAT"
python -m pytest -v THOUGHT/LAB/CAT_CHAT/tests/test_context_assembler.py
```

## Verification Highlights

1. **Determinism**: `test_assembler_determinism` verifies that identical inputs produce bit-exact context receipt hashes.
2. **Fail-Closed**: `test_fail_closed_mandatory` verifies that if mandatory items (System + Latest User) exceed the budget, the assembler refuses to return a partial context and sets `success=False`.
3. **Priority & Truncation**: `test_priority_and_truncation` verifies:
    - Mandatory items are included first.
    - Large messages are truncated via HEAD truncation (preserves start, discards end).
    - Recent dialog (Tier 2) is prioritized over explicit expansions (Tier 3).
    - If budget runs out in Tier 2, Tier 3 is skipped.
    - Final ordering is correct (System -> Expansions -> History -> User).
4. **Starvation**: `test_starvation_of_expansions` confirms that high-priority history starves low-priority expansions.
5. **Budgets**: `test_max_messages_limit` confirms that message count cap is enforced strictly even if tokens remain.

## Phase 3.2.1 Design Decisions

### Truncation
- **Mode**: Only HEAD truncation is supported (preserves start of content, discards end).
- **Method**: Character-based binary search approximation (10 iterations).
- **Guarantees**: Deterministic for identical inputs and token_estimator functions. Token estimates are approximate; no token-exact guarantees.
- **Metadata**: 
  - `start_trimmed` is always `False` (HEAD truncation preserves start).
  - `end_trimmed` is `True` if content was truncated, `False` otherwise.

### Expansion Role Assignment
- All expansion items are assigned `role="SYSTEM"` by design.
- This is intentional and not configurable in Phase 3.2.1.

### Priority Tiers
1. **Mandatory**: System prompt (if present) + Latest user message
2. **Recent Dialog**: Assistant/User messages in reverse chronological order
3. **Explicit Expansions**: Symbol expansions referenced by latest user message
4. **Optional Extras**: Additional expansions (only if budget remains)

## Scope Confirmation

- No new directories created (used existing `catalytic_chat` package).
- No persistence (pure in-memory logic).
- No policy invention (strictly followed priority rules).
- All tests pass (5/5).

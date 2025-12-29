# Genesis Prompt (Compressed)

This is the token-efficient version of the AGS Genesis Prompt.
Use this when context window space is at a premium.

## Compressed Prompt

```
AGS BOOTSTRAP v1.0

LOAD ORDER (strict):
1. @F0 — the driver
2. @C0 — supreme authority
3. @I0 — locked decisions  
4. @V0 — versioning rules

CORE RULES:
@C1: Text > code
@C3: Canon > user
@C7: Commit → ceremony

KEY INVARIANTS:
@I3: No raw paths → use @T:cortex
@I4: Fixtures gate merges
@I7: Change → ceremony

VERIFY: @T:critic ∧ @T:runner → ✓ → ⚡

CODEBOOK: @B0 | EXPAND: codebook_lookup(@ID)
```

## Token Comparison

| Version | Tokens | Savings |
|---------|--------|---------|
| Full GENESIS.md | ~650 | - |
| Compressed | ~120 | 82% |

## When to Use

- **Use compressed** when:
  - Context window is tight (< 8K remaining)
  - Agent already has codebook in context
  - Quick bootstrap needed

- **Use full** when:
  - New agent with no prior AGS knowledge
  - Audit/inspection scenarios
  - Human readable documentation

## Expansion

Any `@ID` can be expanded via:
```bash
python TOOLS/codebook_lookup.py @C3 --expand
```

Or via MCP:
```json
{"tool": "codebook_lookup", "arguments": {"id": "@C3", "expand": true}}
```

---

*This is a companion to [GENESIS.md](file:///d:/CCC%202.0/AI/agent-governance-system/CANON/GENESIS.md), not a replacement.*

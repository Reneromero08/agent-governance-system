# Archived Reports: Turn-Level Catalytic System

**Archive Date:** 2026-01-21

## What These Reports Represent

These reports were generated with the **OLD** turn-level catalytic system, which had a fundamental limitation:

- Messages were stored as compressed TURN blobs (user+assistant combined)
- E-scores were computed on turn POINTERS (summaries), not individual messages
- Hydration retrieved entire compressed turns, not specific messages

## Why They're Archived

On 2026-01-21, the system was fixed to be **truly catalytic at the message level**:

- Each user and assistant message is now stored INDIVIDUALLY with its embedding
- E-scores are computed on individual message embeddings
- Hydration can retrieve specific relevant messages

## Report Validity

These reports were **accurate for what was tested** at the time:
- Turn-level compression/hydration worked
- Budget invariants were maintained
- Hash-chained logging was functional

However, they tested a **coarser** system than "catalytic" implies. Re-running these tests now would produce results from the new message-level system.

## Files

- `STRESS_TEST_BRIEF.md` - Original test brief
- `STRESS_TEST_REPORT_2026-01-19.md` - 100% recall on software architecture scenario
- `CHAOS_TEST_REPORT_2026-01-19.md` - Chaos/edge case testing
- `CIPHER_MARATHON_REPORT_2026-01-19.md` - Long-running cipher test
- `CIPHER_THINKING_REPORT_2026-01-20.md` - Cipher with thinking
- `CIPHER_GLM47_REPORT_2026-01-20.md` - Cipher with GLM4-7B

## Re-Running Tests

The test scripts in `../stress_tests/` now automatically use the new message-level catalytic behavior. Running them will produce new results that should be saved to `../reports/` (not this archive).

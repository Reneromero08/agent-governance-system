# Exp 50 L4A Class B Scaffold Report

**Status:** L4A_CLASS_B_SCAFFOLD_PASS. Claim L1.
**Commit:** 671adf3d

---

## Scaffold Provenance

- **Source files committed:** `holo_record.h`, `holo_record.c`, `class_b_pdn_screen.c`
- **Dry-run .holo:** `results/l4a_class_b/scaffold_dry_run.holo` -- generated at runtime, gitignored per `CAT_CAS/.gitignore` line 21 (`**/*.holo`)
- **.holo gitignore policy:** All `.holo` files are intentionally excluded from version control. They are regenerable artifacts produced by the scaffold binary. The source (`.c`/`.h`) is the canonical record.
- **Rebuild confirmed:** Scaffold recompiles cleanly on Phenom II (`gcc -O2 -Wall -Wextra -lm`, 1 harmless warning). Dry-run .holo regenerated identically.
- **Doctrine guards:** All 9 active, verified on rebuild.
- **No collapse contamination:** No verify(x), no AUC, no candidate scoring, no d output, no recovery claim, no orientation claim.

---

## Files in Commit 671adf3d

| File | Lines | Purpose |
|---|---|---|
| `holo_record.h` | 168 | .holo schema, structs, constants, API |
| `holo_record.c` | 165 | Init, orbit set, collapse validate, doctrine guard, JSON writer |
| `class_b_pdn_screen.c` | 68 | Scaffold entry: guard + dry-run .holo |

**Not committed (intentionally gitignored):**
- `results/l4a_class_b/scaffold_dry_run.holo` -- `**/*.holo` pattern in `CAT_CAS/.gitignore`

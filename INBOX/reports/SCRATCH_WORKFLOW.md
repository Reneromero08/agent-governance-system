# Scratch Layer Workflow (F2)

**Goal**: Guarantee safe, destructive experimentation with zero risk to the canonical repo state.

## The Principle
"Essence is preserved by isolating Entropy."

## The Workflow

1.  **Snapshot (Atomicity)**
    *   Compute the Merkle root (hash tree) of the target directory.
    *   This is the "Before" state ($S_0$).

2.  **Catalyze (Isolation)**
    *   Create a clean isolation layer (`_tmp/scratch_{id}`).
    *   **Method A (Fast)**: `git worktree add ...` (uses OS-level copy-on-write if available).
    *   **Method B (Simple)**: `shutil.copytree` (brute force, reliable).
    *   **Rule**: The Scratch Layer exists *outside* variables tracked by the main repo's `.git`.

3.  **Execute (Destruction)**
    *   The agent performs operations in the Scratch Layer:
        *   `rm -rf *`
        *   `sed -i ...`
        *   Build code, run tests, generate artifacts.
    *   **Constraint**: No symlinks pointing back to `$PROJECT_ROOT`.

4.  **Extract (Harvesting)**
    *   Identify the valuable artifacts (The "Gold").
    *   Copy them to `_tmp/staging/`.
    *   Hash the artifacts.

5.  **Revert (Cleanliness)**
    *   Destroy the Scratch Layer (`rm -rf _tmp/scratch_{id}`).
    *   Verify `$PROJECT_ROOT` hash == $S_0$.
    *   If mismatch: **PANIC** (hard fail, alert human).

6.  **Commit (Integration)**
    *   Move staged artifacts to their specific destination (only if validated).

## Invariants
*   **[INV-F2-1]**: Main repo hash must be identical before and after operation.
*   **[INV-F2-2]**: Scratch layers must be strictly ephemeral (deleted on exit).

# STYLE-005: Git Commit and Push Protocol

**Authority:** CONTEXT/preferences  
**Version:** 1.0.0  
**Status:** Active
**Category:** Governance
**Scope:** Repository
**Enforcement:** Strict

## Purpose

Prevent uncontrolled staging, committing, and pushing that spams CI and breaks governance.

---

## Hard Rules

### 1. Staging Rules

**NEVER use `git add .`**

Agents MUST:
- Stage files explicitly by path
- Group related changes into logical commits
- Verify what's staged before committing: `git status`

```bash
# ✅ CORRECT
git add CANON/SYSTEM_BUCKETS.md
git add AGS_ROADMAP_MASTER.md

# ❌ FORBIDDEN
git add .
git add -A
```

### 2. Branch Rules

**NO direct pushes to `main`**

All work MUST:
- Happen on feature branches
- Go through pull requests
- Pass CI checks before merge

```bash
# ✅ CORRECT workflow
git checkout -b feature/bucket-taxonomy
# ... make changes ...
git push origin feature/bucket-taxonomy
# ... create PR on GitHub ...

# ❌ FORBIDDEN
git push origin main
```

### 3. Required Local Checks

Before ANY commit, agents MUST run:

```bash
# 1. Critic (governance checks)
python TOOLS/ags.py preflight --allow-dirty-tracked

# 2. Contracts (fixtures)
python CONTRACTS/runner.py

# 3. Tests (if applicable)
python -m pytest TESTBENCH/ -v
```

**If any check fails, DO NOT COMMIT.**

### 4. One Approval = One Commit

From `AGENTS.md` Section 10:

- One user approval authorizes ONE commit only
- Completing additional work requires NEW approval
- Chaining commits under single approval is FORBIDDEN

### 5. No Push Without Green Checks

Pushes are blocked unless:
- `CONTRACTS/_runs/ALLOW_PUSH.token` exists (human-created)
- Local contracts pass
- Pre-push hook validates token

---

## Commit Ceremony

1. **Make changes** (code, docs, etc.)
2. **Run checks** (critic, contracts, tests)
3. **Stage explicitly** (by file path)
4. **Request approval** from user
5. **Commit once** with descriptive message
6. **DO NOT PUSH** - wait for explicit push approval

---

## Push Protocol

### Human-Only Push Approval

To allow a push, the human must create:

```bash
# Create one-time push token
echo "feature/bucket-taxonomy" > CONTRACTS/_runs/ALLOW_PUSH.token
```

The pre-push hook will:
1. Check token exists
2. Run minimal contracts validation
3. Allow push
4. **Delete token** (one-time use)

If token missing, push is **BLOCKED**.

---

## Enforcement

- **Pre-push hook**: Installed via `python TOOLS/setup_git_hooks.py`
- **CI triggers**: Only on PRs to `main`, not direct pushes
- **Contracts fixture**: Validates workflow trigger policy
- **GitHub branch protection**: Requires PR + status checks (human-configured)

---

## Violations

Agents that violate this protocol:
- Spam user's email with CI failures
- Break governance
- Violate commit ceremony (AGENTS.md Section 10)

**This is a critical governance failure.**

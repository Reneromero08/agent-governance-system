---
title: "GitHub Branch Protection Guide"
section: "guide"
author: "Antigravity"
priority: "High"
created: "2025-12-29 04:49"
modified: "2025-12-29 05:33"
status: "Active"
summary: "Step-by-step setup for GitHub branch protection rules"
tags: [github, security, guide]
---

<!-- CONTENT_HASH: 6453e06ed50370d90dc2b705127b22b9a0edf04ef302b5e891fc09630f0f016c -->

# GitHub Branch Protection Setup

**Purpose:** Protect `main` branch from direct pushes and require PR review.

**Status:** Manual configuration required (cannot be automated from repo)

---

## Steps to Configure

### 1. Navigate to Repository Settings

1. Go to: `https://github.com/[YOUR_ORG]/agent-governance-system/settings`
2. Click **Branches** in left sidebar
3. Click **Add rule** or edit existing rule for `main`

### 2. Branch Name Pattern

```
main
```

### 3. Required Settings

Enable the following protections:

#### ✅ Require a pull request before merging
- **Required approvals:** 1 (or more if desired)
- **Dismiss stale pull request approvals when new commits are pushed:** ✅
- **Require review from Code Owners:** (optional)

#### ✅ Require status checks to pass before merging
- **Require branches to be up to date before merging:** ✅
- **Status checks that are required:**
  - `Governance Checks` (from contracts.yml)
  - Add any other required checks

#### ✅ Require conversation resolution before merging
- Ensures all PR comments are addressed

#### ✅ Restrict who can push to matching branches
- **Restrict pushes that create matching branches:** ✅
- Add specific users/teams who can push directly (typically: repo admins only)

#### ⚠️ Do not allow bypassing the above settings
- Prevents admins from bypassing protections

### 4. Additional Recommended Settings

- **Require signed commits:** ✅ (if using GPG)
- **Require linear history:** ✅ (prevents merge commits)
- **Allow force pushes:** ❌ (disabled)
- **Allow deletions:** ❌ (disabled)

---

## Verification

After setup, verify:

1. Try to push directly to `main` - should be **blocked**
2. Create a PR - should trigger CI checks
3. Try to merge PR without approval - should be **blocked**
4. Try to merge PR with failing checks - should be **blocked**

---

## Enforcement

- **Local:** Pre-push hook blocks pushes without token
- **Remote:** GitHub branch protection blocks direct pushes
- **CI:** Only runs on PRs, not direct pushes

See: `CONTEXT/preferences/STYLE-005-git-commit-push-protocol.md`
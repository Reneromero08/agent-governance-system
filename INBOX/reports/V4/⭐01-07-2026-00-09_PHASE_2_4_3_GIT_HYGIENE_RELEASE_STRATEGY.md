---
uuid: 70a56db7-af16-417c-a14c-8788e7647573
title: "AGS Release Strategy - Separating Framework from Instance Data"
section: report
bucket: governance/release
author: Claude
priority: High
created: 2026-01-07 00:09
modified: 2026-01-07 00:30
status: Draft
summary: 'How to release AGS as a reusable framework/template while keeping your
  personal instance data (embeddings, proofs, experiments, reports) separate.'
tags:
- release-strategy
- template
- framework
- instance-data
hashtags:
- '#governance'
- '#releases'
- '#template'
---
<!-- CONTENT_HASH: 08806872c22ef9e678f5cb2ff04c179e271bfa031f617bfdb7c87735d21d8333 -->

# AGS Release Strategy: Framework vs Instance Data

**Goal**: Release AGS as a reusable template/framework that others can use, while keeping YOUR personal data separate.

---

## The Core Distinction

AGS has two layers:

| Layer | What It Is | Release? |
|-------|------------|----------|
| **Framework** | Code, structure, primitives, tools, UX | YES |
| **Instance Data** | Your embeddings, proofs, experiments, reports, blobs | NO |

**Every directory stays. Only YOUR DATA gets excluded.**

---

## Directory-by-Directory Breakdown

### NAVIGATION/CORTEX/
| Include | Exclude |
|---------|---------|
| `cortex.py`, `runner.py`, `indexer.py` | `_generated/*.db` (your embeddings) |
| `meta/` structure | Your indexed sections |
| Configuration files | `db/system1.db` (your vectors) |

**Why**: Cortex IS the semantic memory system. Users need the code to build their own indexes.

### NAVIGATION/PROOFS/
| Include | Exclude |
|---------|---------|
| Folder structure | Your specific proof JSONs |
| `COMPRESSION/`, `CATALYTIC/`, `CRYPTO_SAFE/` dirs | Your proof artifacts |
| README/templates if any | Your receipts |

**Why**: Proofs folder demonstrates WHERE users put their compression/catalytic proofs. Empty on release.

### MEMORY/LLM_PACKER/
| Include | Exclude |
|---------|---------|
| `Engine/` (all packer code) | `_packs/*` (your generated packs) |
| `README.md`, `CHANGELOG.md` | `_packs/_archive/*.zip` (your backups) |
| `.cmd` scripts | `_packs/_state/` (your state) |

**Why**: LLM Packer is essential for users working with other LLMs. They need the engine, not your packs.

### MEMORY/ARCHIVE/
| Include | Exclude |
|---------|---------|
| Folder structure | Your archived snapshots |
| `.gitkeep` files | `implemented/`, `historical/` contents |

**Why**: Archive is where users store their own historical snapshots. Empty on release.

### THOUGHT/LAB/
| Include | Exclude |
|---------|---------|
| Folder structure | Your experiments |
| `CAT_CHAT/` framework code | Your chat logs/archives |
| `TURBO_SWARM/` if framework | Your swarm runs |
| README files | `MCP_EXPERIMENTAL/` personal stuff |

**Why**: LAB is the experimentation sandbox. Users need the space, not your experiments.

### INBOX/
| Include | Exclude |
|---------|---------|
| Folder structure | Your reports |
| `reports/` dir (empty) | `reports/*.md` (your files) |
| `research/` dir (empty) | `research/*.db` (your DBs) |

**Why**: INBOX is critical UX for human-AI collaboration. Users need the structure.

### .ags-cas/ (CAS)
| Include | Exclude |
|---------|---------|
| Nothing (generated at runtime) | Everything (your blobs) |

**Why**: CAS is THE core primitive. But the `.ags-cas/` directory is generated on first use. Include the PRIMITIVES code, not the blob storage.

### LAW/CONTRACTS/_runs/
| Include | Exclude |
|---------|---------|
| Folder structure | Your run outputs |
| `_demos/` (example runs) | `RECEIPTS/`, `REPORTS/` contents |
| `.gitkeep` files | `pytest_tmp/`, `ALLOW_PUSH.token` |

**Why**: Runs infrastructure shows users how the system tracks work. They start fresh.

---

## Implementation Options

### Option 1: `.gitattributes` with `export-ignore`

Add to `.gitattributes`:
```gitattributes
# Instance data - exclude from git archive/export
NAVIGATION/CORTEX/_generated/** export-ignore
NAVIGATION/CORTEX/db/** export-ignore
NAVIGATION/PROOFS/**/*.json export-ignore
NAVIGATION/PROOFS/**/*.md export-ignore
MEMORY/LLM_PACKER/_packs/** export-ignore
MEMORY/ARCHIVE/** export-ignore
THOUGHT/LAB/**/archive/** export-ignore
THOUGHT/LAB/**/logs/** export-ignore
INBOX/reports/** export-ignore
INBOX/research/** export-ignore
.ags-cas/** export-ignore
LAW/CONTRACTS/_runs/RECEIPTS/** export-ignore
LAW/CONTRACTS/_runs/REPORTS/** export-ignore
```

Then release with:
```bash
git archive HEAD --prefix=ags-template/ -o ags-template.zip
```

### Option 2: Release Branch

Create a `template` branch that never has your instance data:
```bash
git checkout -b template
# Remove instance data
git rm -r --cached NAVIGATION/CORTEX/_generated/
git rm -r --cached .ags-cas/
# etc.
git commit -m "Template branch - framework only"
```

### Option 3: Export Script (Most Control)

Create `CAPABILITY/TOOLS/release/export_template.py`:
```python
"""
Export AGS as a clean template.
Copies framework, excludes instance data.
"""

INCLUDE_PATTERNS = [
    "LAW/**",
    "CAPABILITY/**",
    "NAVIGATION/**",
    "DIRECTION/**",
    "THOUGHT/**",
    "MEMORY/**",
    "INBOX/**",
    "AGENTS.md",
    "README.md",
    # etc.
]

EXCLUDE_PATTERNS = [
    # Your instance data
    "NAVIGATION/CORTEX/_generated/**",
    "NAVIGATION/CORTEX/db/*.db",
    "NAVIGATION/PROOFS/**/*.json",
    "NAVIGATION/PROOFS/**/*.md",
    "!NAVIGATION/PROOFS/**/README.md",  # Keep READMEs
    "MEMORY/LLM_PACKER/_packs/**",
    "MEMORY/ARCHIVE/**",
    "THOUGHT/LAB/**/archive/**",
    "THOUGHT/LAB/**/logs/**",
    "THOUGHT/LAB/**/receipts/**",
    "INBOX/reports/**",
    "INBOX/research/**",
    ".ags-cas/**",
    "LAW/CONTRACTS/_runs/RECEIPTS/**",
    "LAW/CONTRACTS/_runs/REPORTS/**",
    "LAW/CONTRACTS/_runs/*.token",
]
```

---

## What Users Get (Template Contents)

```
AGS-TEMPLATE/
├── LAW/
│   ├── CANON/           # Governance rules (full)
│   └── CONTRACTS/
│       └── _runs/       # Empty, with .gitkeep
├── CAPABILITY/
│   ├── PRIMITIVES/      # All code including CAS, write_firewall, etc.
│   ├── TOOLS/           # All tools
│   ├── SKILLS/          # All skill frameworks
│   ├── PIPELINES/       # Pipeline code
│   └── MCP/             # MCP server code
├── NAVIGATION/
│   ├── CORTEX/          # Indexer code, empty _generated/
│   ├── MAPS/            # Navigation structure
│   └── PROOFS/          # Empty proof directories
├── DIRECTION/           # Planning structure
├── THOUGHT/
│   └── LAB/             # Empty experimentation space
├── MEMORY/
│   ├── LLM_PACKER/
│   │   ├── Engine/      # Full packer code
│   │   └── _packs/      # Empty, with .gitkeep
│   └── ARCHIVE/         # Empty
├── INBOX/
│   ├── reports/         # Empty
│   └── research/        # Empty
├── AGENTS.md
├── README.md
└── .gitignore           # Pre-configured for instance data
```

---

## First-Run Experience for New Users

1. Clone template
2. Run `python CAPABILITY/TOOLS/ags.py init` (or similar)
3. CAS initializes `.ags-cas/`
4. Cortex builds their own indexes
5. They create their own experiments in LAB
6. Their proofs go in PROOFS
7. Their reports go in INBOX

**They get YOUR SYSTEM, not your data.**

---

## TODO: Implementation Tasks

1. [ ] Create `.gitattributes` with `export-ignore` patterns
2. [ ] Create `CAPABILITY/TOOLS/release/export_template.py` script
3. [ ] Add `.gitkeep` files to all empty directories
4. [ ] Test `git archive` produces clean template
5. [ ] Document first-run setup in README
6. [ ] Consider GitHub Release workflow automation

---

## Summary

| What | Included? | Notes |
|------|-----------|-------|
| All Python code | YES | Framework |
| All tool scripts | YES | Framework |
| Directory structure | YES | Framework |
| README/docs | YES | Framework |
| Your `.db` files | NO | Instance data |
| Your `.json` proofs | NO | Instance data |
| Your INBOX reports | NO | Instance data |
| Your CAS blobs | NO | Instance data |
| Your pack archives | NO | Instance data |
| Your LAB experiments | NO | Instance data |

**The framework is the product. Your data is yours.**

---

## CRYPTO_SAFE: What It's Actually For

CRYPTO_SAFE is NOT about hiding your data from releases. Your data is simply excluded - they never see it.

**CRYPTO_SAFE is about protecting the TEMPLATE you release.**

### The Purpose

| What | Why |
|------|-----|
| Seal the template artifacts | Tamper-evident proof |
| Cryptographic signatures | Provenance chain |
| Manifest of sealed files | "This is what I released" |

### The Use Case

1. You release AGS template with crypto seals
2. Someone downloads it, modifies it, redistributes it
3. They claim it's "their work" or violate your license
4. You say: **"You broke my seal."**
5. Cryptographic proof they tampered with your release

### What Gets Sealed (THE TEMPLATE)

| Seal | Why |
|------|-----|
| `LAW/CANON/**` | Your governance rules |
| `CAPABILITY/PRIMITIVES/**` | Your core code |
| `CAPABILITY/TOOLS/**` | Your tooling |
| Framework structure | Your architecture |

### What DOESN'T Get Sealed (excluded entirely)

| Excluded | Why |
|----------|-----|
| Your embeddings | Not in release at all |
| Your proofs | Not in release at all |
| Your experiments | Not in release at all |
| Your reports | Not in release at all |

**CRYPTO_SAFE = License enforcement. Provenance. Accountability.**

*"I can look you in the eye and say: you broke my seal."*

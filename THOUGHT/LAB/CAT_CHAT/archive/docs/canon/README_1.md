<!-- CONTENT_HASH: 946629d758fbdb6bf853726012403af5010139856a5066dd4563f8a9834cc941 -->

> **⚠️ DEPRECATED:** This document is archived. See `../../README_1.1.md` for the current version.

# Catalytic Chat System

**Roadmap:** [CAT_CHAT_ROADMAP.md](CAT_CHAT_ROADMAP.md)
**Status:** Phase 7 Complete (All Phases Complete)
**Date:** 2025-12-31

## Overview

Build a chat substrate where models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** as needed.

### Current Status

- **Phase 0:** ✅ COMPLETE (Contract frozen)
- **Phase 1:** ✅ COMPLETE (Substrate + deterministic indexing)
- **Phase 2:** ✅ COMPLETE (Symbol registry + bounded resolver)
- **Phase 3:** ✅ COMPLETE (Message cassette)
- **Phase 4:** ⏳ NOT STARTED (Discovery: FTS + vectors)
- **Phase 5:** ⏳ NOT STARTED (Translation protocol)
- **Phase 6:** ✅ COMPLETE (Measurement and regression harness)

## Documentation

- **Roadmap:** [docs/catalytic-chat/ROADMAP.md](docs/catalytic-chat/ROADMAP.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Contract:** [CAT_CHAT_CONTRACT.md](CAT_CHAT_CONTRACT.md)
- **Phase Reports:** [docs/catalytic-chat/phases/](docs/catalytic-chat/phases/)
- **Notes:** [docs/catalytic-chat/notes/](docs/catalytic-chat/notes/)

## Canonical Package

The active implementation is in the `catalytic_chat/` package:

```bash
# CLI usage
python -m catalytic_chat.cli --help

# Build section index
python -m catalytic_chat.cli build

# Resolve a symbol
python -m catalytic_chat.cli resolve @CANON/AGREEMENT --slice "lines[0:50]" --run-id test-001
```

## Running Tests

```bash
# Run canonical tests (excludes legacy)
python -m pytest -q
```

## Legacy Files

Deprecated scripts and data are quarantined in [legacy/](legacy/) for historical reference. See [legacy/README.md](legacy/README.md) for details.

## Directory Layout

```
CAT_CHAT/
├── README.md                  # This file
├── ROADMAP.md                 # Roadmap
├── CHANGELOG.md               # Canonical changelog (639 lines, all Phase 1-7)
├── pytest.ini                 # Test config (excludes legacy/)
├── catalytic_chat/            # Canonical package
│   ├── CONTRACT.md            # Immutable contract
│   ├── ROADMAP.md            # Canonical roadmap
│   ├── CHANGELOG.md          # Canonical changelog
│   └── [all CAT_CHAT files]
├── archive/                   # Historical documentation (40 files, organized)
└── tests/                     # Canonical tests (excludes legacy/)
│   ├── cli.py                 # CLI entry point
│   ├── section_extractor.py     # Section extraction
│   ├── section_indexer.py      # Indexing and storage
│   ├── symbol_registry.py       # Symbol management
│   ├── symbol_resolver.py       # Symbol resolution with cache
│   └── slice_resolver.py       # Slice parsing
├── tests/                     # Canonical tests (excludes legacy/)
├── legacy/                    # Deprecated scripts (NOT CANONICAL)
│   ├── README.md              # Legacy explanation
│   ├── tests/                 # Legacy test files
│   ├── chats/                 # Legacy chat data
│   └── symbols/               # Legacy symbol dictionary
└── archive/                   # Historical documentation (organized)
```
CAT_CHAT/
├── README.md                  # This file
├── ROADMAP.md                 # Stub -> docs/catalytic-chat/ROADMAP.md
├── CHANGELOG.md               # Stub -> docs/catalytic-chat/CHANGELOG.md
├── pytest.ini                 # Test config (excludes legacy/)
├── catalytic_chat/            # Canonical package
│   ├── cli.py                 # CLI entry point
│   ├── section_extractor.py   # Section extraction
│   ├── section_indexer.py     # Indexing and storage
│   ├── symbol_registry.py     # Symbol management
│   ├── symbol_resolver.py     # Symbol resolution with cache
│   └── slice_resolver.py      # Slice parsing
├── archive/                   # Historical documentation (organized)
├── legacy/                    # Deprecated scripts (NOT CANONICAL)
│   ├── README.md              # Legacy explanation
│   ├── tests/                 # Legacy test files
│   ├── chats/                 # Legacy chat data
│   └── symbols/               # Legacy symbol dictionary
├── tests/                     # Canonical tests only
└── archive/                   # Older roadmap/research
```

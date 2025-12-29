# Catalytic Chat System

**Roadmap:** [docs/catalytic-chat/ROADMAP.md](docs/catalytic-chat/ROADMAP.md)
**Status:** Phase 2 Complete (Substrate + Symbol Registry + Bounded Resolver)
**Date:** 2025-12-29

## Overview

Build a chat substrate where models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** as needed.

### Current Status

- **Phase 0:** ✅ COMPLETE (Contract frozen)
- **Phase 1:** ✅ COMPLETE (Substrate + deterministic indexing)
- **Phase 2:** ✅ COMPLETE (Symbol registry + bounded resolver)
- **Phase 3:** ⏳ NOT STARTED (Message cassette)
- **Phase 4:** ⏳ NOT STARTED (Discovery: FTS + vectors)
- **Phase 5:** ⏳ NOT STARTED (Translation protocol)
- **Phase 6:** ⏳ NOT STARTED (Measurement and regression harness)

## Documentation

- **Roadmap:** [docs/catalytic-chat/ROADMAP.md](docs/catalytic-chat/ROADMAP.md)
- **Changelog:** [docs/catalytic-chat/CHANGELOG.md](docs/catalytic-chat/CHANGELOG.md)
- **Contract:** [docs/catalytic-chat/CONTRACT.md](docs/catalytic-chat/CONTRACT.md)
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
├── docs/
│   └── catalytic-chat/
│       ├── CONTRACT.md        # Immutable contract
│       ├── ROADMAP.md         # Canonical roadmap
│       ├── CHANGELOG.md       # Canonical changelog
│       ├── phases/            # Phase completion reports
│       └── notes/             # Historical notes
├── legacy/                    # Deprecated scripts (NOT CANONICAL)
│   ├── README.md              # Legacy explanation
│   ├── tests/                 # Legacy test files
│   ├── chats/                 # Legacy chat data
│   └── symbols/               # Legacy symbol dictionary
├── tests/                     # Canonical tests only
└── archive/                   # Older roadmap/research
```

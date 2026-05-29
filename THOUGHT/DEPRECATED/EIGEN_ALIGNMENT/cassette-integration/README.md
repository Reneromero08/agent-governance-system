# Cassette Integration for ESAP

**Status:** EXPERIMENTAL - Not production ready

## Overview

This directory contains experimental code for integrating ESAP (Eigen-Spectrum Alignment Protocol) with the Cassette Network. These files were prematurely moved to production (`NAVIGATION/CORTEX/network/`) before ESAP was complete.

**Moved back to LAB on 2026-01-25** because:
1. ESAP roadmap items (ESAP.1-5) are all incomplete
2. This is a simplified/incomplete port of the real protocol in `../lib/handshake.py`
3. Missing features: message types (ESAPHello/Ack/Reject), replay protection, proper tests

## Files

- `esap_cassette.py` - Simplified ESAP mixin for cassettes (183 lines)
- `esap_hub.py` - ESAP-enabled network hub (204 lines)

## Relationship to Main ESAP Implementation

The **canonical ESAP implementation** is in:
- `../lib/handshake.py` - Full protocol with ESAPHello, ESAPAck, ESAPReject
- `../lib/protocol.py` - Message types and schemas
- `../tests/test_handshake.py` - 16 tests

This cassette integration is a **consumer** of that protocol, not a replacement.

## When to Graduate

These files should move back to `NAVIGATION/CORTEX/network/` when:
1. [ ] ESAP.1 - Full protocol implemented per spec
2. [ ] ESAP.2 - Benchmarked with anchor sets
3. [ ] ESAP.3 - Neighborhood overlap tested
4. [ ] ESAP.4 - Compared with vec2vec approach
5. [ ] ESAP.5 - Integrated as cassette handshake artifact

See: `AGS_ROADMAP_MASTER.md` Phase 6 Future Work (ESAP section)

## Usage (Experimental)

```python
from esap_cassette import ESAPCassetteMixin
from esap_hub import ESAPNetworkHub

# These are NOT imported by production code
# They require the full ESAP protocol to be complete first
```

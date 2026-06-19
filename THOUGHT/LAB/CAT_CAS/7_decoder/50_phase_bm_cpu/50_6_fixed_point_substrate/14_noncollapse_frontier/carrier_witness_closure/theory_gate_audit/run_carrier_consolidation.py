#!/usr/bin/env python3
"""Generate a deterministic Phase 6B.5D packet.

The consolidation is derived entirely from the bound Phase 6B.5C packet, so its
provenance timestamp is inherited from that packet rather than wall-clock time.
Repeated executions over identical inputs therefore produce identical bytes and
SHA-256 values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_carrier_consolidation as consolidation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    source_manifest = consolidation.load_json(result_dir / "analysis_manifest.json")
    bound_time = source_manifest.get("generated_utc")
    if not isinstance(bound_time, str) or not bound_time:
        raise ValueError("Phase 6B.5C manifest has no generated_utc binding")

    consolidation.utc_now = lambda: bound_time
    manifest = consolidation.build(result_dir, args.output.resolve())
    print(json.dumps(manifest["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

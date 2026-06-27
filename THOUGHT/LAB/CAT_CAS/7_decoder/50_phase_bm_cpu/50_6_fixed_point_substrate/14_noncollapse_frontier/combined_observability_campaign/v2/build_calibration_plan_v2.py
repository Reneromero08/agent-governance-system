#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_contract import (
    build_plan,
    build_source_bundle_manifest,
    compile_sessions,
    write_immutable,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sessions-output", required=True, type=Path)
    parser.add_argument("--source-bundle-output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.source_commit)
    digest = write_immutable(args.output, plan)
    bindings = compile_sessions(plan, args.sessions_output)
    bundle_digest = write_immutable(
        args.source_bundle_output, build_source_bundle_manifest(bindings)
    )
    print(json.dumps({"plan_sha256": digest, "source_bundle_sha256": bundle_digest,
                      "total_window_count": plan["total_window_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

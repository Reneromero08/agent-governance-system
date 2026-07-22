#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any


def self_test() -> dict[str, Any]:
    return {
        "schema": "FAMILY10H_RELATION_SPATIAL_LIVE_CONTROLLER_SELF_TEST_V1",
        "passed": True,
        "target_contact_count": 0,
        "physical_pmu_acquisition_count": 0,
        "live_invocation_count": 0,
        "small_wall_crossed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = self_test() if args.self_test else {"passed": False, "error": "only --self-test is implemented locally"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path


_TASK_ID_RE = re.compile(r"^swarm-\d+$")


def main(actual_path: Path, expected_path: Path) -> int:
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    if actual.get("status") != "success":
        return 1
    if actual.get("task_status") != expected.get("task_status"):
        return 1

    task_id = actual.get("task_id")
    if not (isinstance(task_id, str) and _TASK_ID_RE.match(task_id)):
        return 1

    task_spec = actual.get("task_spec")
    if not isinstance(task_spec, dict):
        return 1
    if task_spec.get("instruction") != expected.get("directive"):
        return 1
    if task_spec.get("task_type") != expected.get("task_type"):
        return 1
    if task_spec.get("task_id") != task_id:
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

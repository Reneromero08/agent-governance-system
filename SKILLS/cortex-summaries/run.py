#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_build_module() -> object:
    build_path = PROJECT_ROOT / "CORTEX" / "cortex.build.py"
    cortex_dir = str(PROJECT_ROOT / "CORTEX")
    if cortex_dir not in sys.path:
        sys.path.insert(0, cortex_dir)
    spec = importlib.util.spec_from_file_location("cortex_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main(input_path: Path, output_path: Path) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    record = payload.get("record") or {}
    slice_text = str(payload.get("slice_text") or "")

    build = load_build_module()
    safe_filename = build._safe_section_id_filename(str(record.get("section_id") or ""))  # type: ignore[attr-defined]
    summary_md = build._summarize_section(record, slice_text)  # type: ignore[attr-defined]
    summary_sha256 = hashlib.sha256(summary_md.encode("utf-8")).hexdigest()

    output = {
        "safe_filename": safe_filename,
        "summary_md": summary_md,
        "summary_sha256": summary_sha256,
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <actual.json>")
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

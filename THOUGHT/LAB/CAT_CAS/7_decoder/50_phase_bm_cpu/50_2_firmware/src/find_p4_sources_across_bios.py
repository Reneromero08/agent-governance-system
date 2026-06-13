#!/usr/bin/env python3
"""Search extracted Exp50 firmware artifacts for separate P4/P-state sources."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


PSTATE_VALUES = [0xC0010064 + i for i in range(5)]
KNOWN_CLOSED_NAMES = {
    "AmdProcessorInitPeim",
}


def find_binary_hits(path: Path, value: int) -> list[int]:
    data = path.read_bytes()
    needle = struct.pack("<I", value)
    out = []
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx < 0:
            return out
        out.append(idx)
        start = idx + 1


def module_name(path: Path) -> str:
    parts = path.parts
    for part in reversed(parts):
        if " " in part and not part.endswith(".bin"):
            return part
    return path.parent.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="cpu_sing_3/PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_SEARCH.md")
    args = parser.parse_args()
    root = Path(args.root)
    dump_root = root / "cpu_hack" / "bios_dump.bin.dump"
    rows = []
    for path in dump_root.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size > 32 * 1024 * 1024:
            continue
        suffix = path.suffix.lower()
        if suffix not in {".bin", ".efi", ".pe32", ""} and "body" not in path.name.lower():
            continue
        try:
            per_value = {value: find_binary_hits(path, value) for value in PSTATE_VALUES}
        except OSError:
            continue
        hit_count = sum(len(v) for v in per_value.values())
        if hit_count:
            rows.append({
                "path": path,
                "module": module_name(path),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                "per_value": per_value,
                "hit_count": hit_count,
            })

    text_hits = []
    for path in (root / "cpu_hack").rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        count = sum(text.count(f"0x{value:08X}") + text.count(f"{value:08X}") for value in PSTATE_VALUES)
        if count:
            text_hits.append((path, count))

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# PHASE2_FIRMWARE_P4_SEPARATE_SOURCE_SEARCH\n\n")
        handle.write("## Verdict\n\n")
        outside_closed = [row for row in rows if not any(name in row["module"] for name in KNOWN_CLOSED_NAMES)]
        handle.write("`FIRMWARE_P4_SEPARATE_SOURCE_CANDIDATE_FOUND`\n\n" if outside_closed else "`FIRMWARE_P4_SEPARATE_SOURCE_NOT_FOUND_CURRENT_DUMP`\n\n")
        handle.write("Search for P-state MSR constants across the extracted firmware tree, looking for a P4-affecting source outside the already closed AmdProcessorInitPeim runtime-MSR helper chain.\n\n")
        handle.write("## Binary Hits\n\n")
        handle.write("| Module | File | Hits | Values | SHA-256 |\n")
        handle.write("|---|---|---:|---|---|\n")
        for row in rows:
            values = []
            for value, hits in row["per_value"].items():
                if hits:
                    values.append(f"0x{value:08X}:{len(hits)}")
            handle.write(f"| `{row['module']}` | `{row['path'].as_posix()}` | {row['hit_count']} | {', '.join(values)} | `{row['sha256']}` |\n")
        if not rows:
            handle.write("| none | none | 0 | none | none |\n")
        handle.write("\n## Text Trace Hits\n\n")
        handle.write("| File | Hit count |\n")
        handle.write("|---|---:|\n")
        for path, count in text_hits:
            handle.write(f"| `{path.as_posix()}` | {count} |\n")
        if not text_hits:
            handle.write("| none | 0 |\n")
        handle.write("\n## Interpretation\n\n")
        if outside_closed:
            handle.write("At least one binary hit exists outside the closed AmdProcessorInitPeim chain. The next step is module-level disassembly and P0-P4 sibling proof before any candidate construction.\n")
        else:
            handle.write("No separate P4/P-state MSR constant source was found outside the already decoded AmdProcessorInitPeim runtime-MSR chain in the current extracted dump.\n")
        handle.write("\n## Boundary\n\n")
        handle.write("- Search only; no image modification.\n")
        handle.write("- No candidate construction.\n")
        handle.write("- No platform setting changes.\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

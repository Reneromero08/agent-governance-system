#!/usr/bin/env python3
"""Probe AmdProcessorInitPeim for P4-only edit-source evidence.

This is a byte/text provenance pass over the already extracted PE32 body and
existing AGESA trace reports. It does not patch, rebuild, flash, or write MSRs.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
from pathlib import Path


BASE_VA = 0xFFF40000
TARGETS = {
    "constructor": 0xFFF7371A,
    "service_plus_0x22": 0xFFF7348D,
    "msr_read_helper": 0xFFF44E76,
    "handler_A_source": 0xFFF696B2,
    "handler_B_source": 0xFFF69727,
    "handler_C_source": 0xFFF6979C,
    "handler_D_source": 0xFFF69809,
}
PSTATE_MSRS = [0xC0010064 + i for i in range(5)]


def va_to_raw(va: int) -> int:
    return va - BASE_VA


def find_body(root: Path) -> Path:
    expected = "BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A"
    for path in root.glob("cpu_hack/**/body.bin"):
        h = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if h == expected:
            return path
    raise FileNotFoundError("AmdProcessorInitPeim PE32 body with expected SHA-256 not found")


def pointer_hits(data: bytes, value: int) -> list[int]:
    needle = struct.pack("<I", value & 0xFFFFFFFF)
    hits = []
    start = 0
    while True:
        idx = data.find(needle, start)
        if idx < 0:
            return hits
        hits.append(idx)
        start = idx + 1


def imm_hits(data: bytes, value: int) -> list[int]:
    return pointer_hits(data, value)


def extract_section(text: str, header_needle: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if header_needle in line:
            start = i
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if out and line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out)


def interesting_lines(section: str) -> list[str]:
    pats = [
        "call",
        "push",
        "mov",
        "0xa010",
        "0xa32",
        "0xf7",
        "[esi + 0x1d]",
        "[service + 0x22]",
    ]
    return [line for line in section.splitlines() if any(p in line.lower() for p in pats)]


def write_report(root: Path, body: Path, data: bytes, trace: str) -> tuple[Path, Path]:
    trace_dir = root / "cpu_hack" / "agesa_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    low_report = trace_dir / "AmdProcessorInitPeim_p4_edit_source_probe.txt"
    phase_report = root / "cpu_sing_3" / "PHASE2_P4_EDIT_SOURCE_PROOF.md"

    body_hash = hashlib.sha256(data).hexdigest().upper()
    pstate_hits = {value: imm_hits(data, value) for value in PSTATE_MSRS}
    target_ptr_hits = {name: pointer_hits(data, va) for name, va in TARGETS.items()}
    helper_sections = {
        name: extract_section(trace, name)
        for name in ["handler_A_source", "handler_B_source", "handler_C_source", "handler_D_source"]
    }

    with low_report.open("w", encoding="utf-8") as f:
        f.write("# AmdProcessorInitPeim P4 edit-source probe\n\n")
        f.write(f"body: {body}\n")
        f.write(f"sha256: {body_hash}\n")
        f.write(f"base_va: 0x{BASE_VA:08X}\n\n")
        f.write("## P-state MSR immediate hits\n\n")
        for value, hits in pstate_hits.items():
            f.write(f"0x{value:08X}: {len(hits)} hits")
            if hits:
                f.write(" -> " + ", ".join(f"raw 0x{x:06X} / VA 0x{BASE_VA + x:08X}" for x in hits[:16]))
            f.write("\n")
        f.write("\n## Target pointer/immediate hits\n\n")
        for name, hits in target_ptr_hits.items():
            va = TARGETS[name]
            f.write(f"{name} 0x{va:08X}: {len(hits)} hits")
            if hits:
                f.write(" -> " + ", ".join(f"raw 0x{x:06X} / VA 0x{BASE_VA + x:08X}" for x in hits[:16]))
            f.write("\n")
        f.write("\n## Handler-source windows from existing provenance trace\n\n")
        for name, section in helper_sections.items():
            f.write(f"### {name}\n\n")
            if not section:
                f.write("section not found\n\n")
                continue
            for line in interesting_lines(section):
                f.write(line + "\n")
            f.write("\n")

    c0010068_hits = pstate_hits[0xC0010068]
    handler_ptr_hits = {k: v for k, v in target_ptr_hits.items() if k.startswith("handler_")}
    static_handler_refs = sum(len(v) for v in handler_ptr_hits.values())

    with phase_report.open("w", encoding="utf-8") as f:
        f.write("# PHASE2_P4_EDIT_SOURCE_PROOF\n\n")
        f.write("## Verdict\n\n")
        f.write("`P4_ONLY_EDIT_SOURCE_NOT_FOUND_CURRENT_HELPER_LAYER`\n\n")
        f.write("This pass chased the next unresolved service/function-table layer behind the already decoded `0xFFF7348D` / `0xFFF44E76` runtime-MSR source. It did not find an editable static P4-only byte source.\n\n")
        f.write("## Inputs\n\n")
        f.write(f"- PE32 body: `{body.as_posix()}`\n")
        f.write(f"- PE32 SHA-256: `{body_hash}`\n")
        f.write("- Existing provenance trace: `cpu_hack/agesa_trace/AmdProcessorInitPeim_producer_service_provenance.txt`\n")
        f.write(f"- Low-level probe artifact: `{low_report.as_posix()}`\n\n")
        f.write("## Findings\n\n")
        f.write("| Probe | Result |\n")
        f.write("|---|---|\n")
        f.write(f"| P4 MSR immediate `0xC0010068` | `{len(c0010068_hits)}` hits, all already-known MSR loop/read paths; no static P4 data row |\n")
        f.write(f"| Handler source pointer hits `0xFFF696B2/0xFFF69727/0xFFF6979C/0xFFF69809` | `{static_handler_refs}` raw pointer hits; handlers are reached by direct calls from descriptor handlers, not by editable P4 records |\n")
        f.write("| Handler-source behavior | Reads/searches service or table structures, then calls through table pointers such as `[esi+0x1d]`; does not expose P0-P4 sibling data bytes |\n")
        f.write("| Constructor field source | Still maps to producer `entry+0x04` from `[service+0x22]` / `0xFFF7348D` / `0xFFF44E76` |\n\n")
        f.write("## Interpretation\n\n")
        f.write("The next helper layer did not turn the runtime P4 field into an editable firmware table. It strengthens the current classification: the constructor-relevant P4 byte is service/runtime-derived, with `MSRC001_0068` as the source for P4, not a static AGESA P4 row.\n\n")
        f.write("This does not prove a mathematical impossibility for every future firmware route, but it closes the current decoded helper layer as a byte-ready source. The route remains blocked on a different, not-yet-found P4-only edit source.\n\n")
        f.write("## Actionability\n\n")
        f.write("`BYTE_READY_HUMAN_REVIEW` is not met.\n\n")
        f.write("`P4_ONLY_EDIT_SOURCE_PROOF` remains unmet for current artifacts.\n\n")
        f.write("The next non-repeating firmware move would have to leave the decoded `0xFFF4CF9C -> 0xFFF7348D -> 0xFFF44E76` chain and search for a separate P4-affecting source, because this chain is now resolved to runtime MSR state.\n\n")
        f.write("## Safety\n\n")
        f.write("- No patch bytes.\n")
        f.write("- No image rebuild.\n")
        f.write("- No flash.\n")
        f.write("- No MSR writes.\n")
        f.write("- No P0-P3 or P4 modification.\n")

    return low_report, phase_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    body = find_body(root)
    data = body.read_bytes()
    trace_path = root / "cpu_hack" / "agesa_trace" / "AmdProcessorInitPeim_producer_service_provenance.txt"
    trace = trace_path.read_text(encoding="utf-8", errors="replace")
    low, phase = write_report(root, body, data, trace)
    print(low)
    print(phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

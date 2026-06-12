#!/usr/bin/env python
"""lint_packet.py - blind-packet hygiene linter for v2_3.

CLI:
    python lint_packet.py <packet-file>

Greps the packet for forbidden substrings (case-insensitive). Any hit:
print the offending line(s) to stdout and exit 1. Clean: print the sha256
of the file and exit 0.

ASCII only. Pure stdlib. Deterministic.
"""

import argparse
import hashlib
import sys
from pathlib import Path

FORBIDDEN = (
    "/v4/",
    "INDEX.md",
    "AGENT_HANDOFF",
    "VERDICT",
    "DISCOVERY_REPORT",
    "RECONCILIATION",
    "VERIFIED",
    "FALSIFIED",
    "CONFIRMED",
    "UNSUPPORTED",
    "Tier ",
)


def lint(packet_path):
    """Return (offending_lines, sha256_hex) for the packet file."""
    data = Path(packet_path).read_bytes()
    text = data.decode("utf-8", errors="replace")
    needles = [f.lower() for f in FORBIDDEN]
    offending = []
    for line in text.split("\n"):
        low = line.lower()
        if any(needle in low for needle in needles):
            offending.append(line)
    return offending, hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lint a blind packet for status/verdict leakage.")
    parser.add_argument("packet", help="path to the packet file")
    args = parser.parse_args(argv)
    path = Path(args.packet)
    if not path.is_file():
        sys.stderr.write("ERROR: packet file not found: %s\n" % args.packet)
        return 1
    offending, digest = lint(path)
    if offending:
        for line in offending:
            print(line)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

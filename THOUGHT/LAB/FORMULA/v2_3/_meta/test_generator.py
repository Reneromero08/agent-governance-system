#!/usr/bin/env python
"""test_generator.py - fixture harness for generate_index.py.

Iterates ALL fixture dirs under _meta/fixtures/ in sorted order. For each:
- runs: python generate_index.py --root <fixture>/tree
- asserts the exit code matches expect.json
- asserts the expected error code appears in stderr (when expected)
- byte-compares tree/INDEX.md against expected_INDEX.md (when exit 0)
- asserts INDEX.md was NOT written (when exit 1)

Prints PASS/FAIL per fixture; exits non-zero if any FAIL.

Fixture format (_meta/fixtures/<name>/):
    tree/         miniature v2_3 root
    expect.json   {"exit_code": 0, "expected_index": "expected_INDEX.md"}
                  or {"exit_code": 1, "error_code": "E_..."}

ASCII only. Pure stdlib.
"""

import json
import subprocess
import sys
from pathlib import Path

META = Path(__file__).resolve().parent
GENERATOR = META / "generate_index.py"
FIXTURES = META / "fixtures"


def run_fixture(fixture):
    """Return a list of failure reasons (empty = pass)."""
    reasons = []
    expect_path = fixture / "expect.json"
    if not expect_path.is_file():
        return ["expect.json missing"]
    try:
        expect = json.loads(expect_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return ["expect.json unparseable: %s" % exc]

    tree = fixture / "tree"
    if not tree.is_dir():
        return ["tree/ missing"]

    index_path = tree / "INDEX.md"
    if index_path.exists():
        index_path.unlink()

    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--root", str(tree)],
        capture_output=True, text=True)

    want_exit = expect.get("exit_code")
    if proc.returncode != want_exit:
        reasons.append("exit code %d != expected %d; stderr: %s"
                       % (proc.returncode, want_exit,
                          proc.stderr.strip()[:300]))

    if want_exit == 1:
        code = expect.get("error_code")
        if code and code not in proc.stderr:
            reasons.append("error code %s not found in stderr: %s"
                           % (code, proc.stderr.strip()[:300]))
        if index_path.exists():
            reasons.append("INDEX.md was written despite errors")

    if want_exit == 0:
        expected_name = expect.get("expected_index")
        if expected_name:
            expected_path = fixture / expected_name
            if not expected_path.is_file():
                reasons.append("%s missing" % expected_name)
            elif not index_path.is_file():
                reasons.append("INDEX.md was not written")
            elif not bytes_match(expected_path.read_bytes(),
                                 index_path.read_bytes(),
                                 bool(expect.get("digest_wildcard"))):
                reasons.append("INDEX.md bytes differ from %s"
                               % expected_name)
    return reasons


DIGEST_SKIP = b"<!-- INPUTS_DIGEST: SKIP -->"
DIGEST_PREFIX = b"<!-- INPUTS_DIGEST: "
DIGEST_SUFFIX = b" -->"


def bytes_match(expected, got, digest_wildcard):
    """Byte-compare; with digest_wildcard a SKIP digest line matches any
    generated INPUTS_DIGEST line."""
    if not digest_wildcard:
        return expected == got
    exp_lines = expected.split(b"\n")
    got_lines = got.split(b"\n")
    if len(exp_lines) != len(got_lines):
        return False
    for exp, actual in zip(exp_lines, got_lines):
        if exp == DIGEST_SKIP:
            if actual.startswith(DIGEST_PREFIX) \
                    and actual.endswith(DIGEST_SUFFIX):
                continue
            return False
        if exp != actual:
            return False
    return True


def main():
    if not FIXTURES.is_dir():
        sys.stderr.write("ERROR: fixtures dir not found: %s\n" % FIXTURES)
        return 1
    fixtures = sorted((d for d in FIXTURES.iterdir() if d.is_dir()),
                      key=lambda d: d.name)
    if not fixtures:
        sys.stderr.write("ERROR: no fixture dirs under %s\n" % FIXTURES)
        return 1
    failed = 0
    for fixture in fixtures:
        reasons = run_fixture(fixture)
        if reasons:
            failed += 1
            print("FAIL %s: %s" % (fixture.name, "; ".join(reasons)))
        else:
            print("PASS %s" % fixture.name)
    print("RESULT: %d passed, %d failed, %d total"
          % (len(fixtures) - failed, failed, len(fixtures)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

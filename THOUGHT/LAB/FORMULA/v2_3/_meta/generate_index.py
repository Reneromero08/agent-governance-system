#!/usr/bin/env python
"""generate_index.py - deterministic INDEX.md generator for v2_3.

CLI:
    python generate_index.py [--root <v2_3 dir>] [--check]

Importable:
    from generate_index import build_index
    content = build_index(root)    # str; raises ValidationFailure on errors

Behavior:
- Validates EVERY verdict first (via validate_verdict). Any error: print
  all errors to stderr, exit 1, INDEX.md is never written.
- A question dir with no VERDICT.md renders as OPEN.
- Output is byte-deterministic (INV-005): no wall-clock, no randomness,
  sorted iteration, LF line endings, UTF-8 without BOM.

ASCII only. Pure stdlib + PyYAML.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import validate_verdict as vv  # noqa: E402

QDIR_RE = re.compile(r"^q[0-9]{2}_[a-z0-9_]+$")
STATUS_KEYS = ("OPEN", "FALSIFIED", "UNSUPPORTED",
               "PARTIALLY_VERIFIED", "VERIFIED")
CLAIM_ABBREV = (("VERIFIED", "V"), ("PARTIALLY_VERIFIED", "PV"),
                ("UNSUPPORTED", "U"), ("FALSIFIED", "F"))

HEADER_LINE = ("<!-- GENERATED FILE. DO NOT EDIT. "
               "Regenerate: python _meta/generate_index.py -->")

LEGEND_LINES = (
    "- Enum (ascending): FALSIFIED < UNSUPPORTED < PARTIALLY_VERIFIED "
    "< VERIFIED",
    "- MIN rule: verdict status equals MIN over claim statuses",
    "- Floor rule: empty evidence manifest caps claims at UNSUPPORTED; "
    "VERIFIED or PARTIALLY_VERIFIED claims need evidence",
    "- OPEN: no VERDICT.md exists for the question",
    "- Verification: blind = packet-isolated; primed = verifier saw "
    "repo context",
)


class ValidationFailure(Exception):
    """Raised by build_index when any verdict or catalog error exists."""

    def __init__(self, errors):
        super().__init__("validation failed with %d error(s)" % len(errors))
        self.errors = errors


def scan_question_dirs(root):
    """Sorted list of q<NN>_<name> directories directly under root."""
    return sorted((d for d in Path(root).iterdir()
                   if d.is_dir() and QDIR_RE.match(d.name)),
                  key=lambda d: d.name)


def validate_all(root):
    """Validate catalog + every verdict. Returns [(code, file, detail)]."""
    root = Path(root)
    errors = []
    catalog_path = root / "_meta" / "questions.yaml"
    entries, cat_errors = vv.load_catalog(root)
    for code, detail in cat_errors:
        errors.append((code, str(catalog_path), detail))
    slugs = set()
    if entries is not None:
        slugs = {str(e["slug"]) for e in entries}
    for qdir in scan_question_dirs(root):
        if entries is not None and qdir.name not in slugs:
            errors.append(("E_CATALOG", str(qdir),
                           "question dir slug not present in "
                           "_meta/questions.yaml"))
        vpath = qdir / "VERDICT.md"
        if vpath.is_file():
            for code, detail in vv.validate(vpath, root):
                errors.append((code, str(vpath), detail))
    return errors


def inputs_digest(root):
    """sha256 over sorted (relpath, sha256) pairs of all input files."""
    root = Path(root)
    files = []
    for rel in ("VARIABLES.md", "PREDICTIONS.md", "_meta/questions.yaml"):
        p = root / rel
        if p.is_file():
            files.append(p)
    for qdir in scan_question_dirs(root):
        vpath = qdir / "VERDICT.md"
        if vpath.is_file():
            files.append(vpath)
    pairs = sorted((p.relative_to(root).as_posix(), vv.sha256_file(p))
                   for p in files)
    h = hashlib.sha256()
    for rel, digest in pairs:
        h.update(("%s %s\n" % (rel, digest)).encode("utf-8"))
    return h.hexdigest()


def _load_frontmatter(vpath):
    split = vv.split_frontmatter(vpath.read_text(encoding="utf-8"))
    return yaml.safe_load(split[0])


def _claims_summary(claims):
    counts = {}
    for claim in claims:
        if isinstance(claim, dict):
            counts[claim.get("status")] = counts.get(claim.get("status"),
                                                     0) + 1
    parts = ["%d%s" % (counts[s], abbrev)
             for s, abbrev in CLAIM_ABBREV if s in counts]
    return "/".join(parts) if parts else "-"


def _escape_cell(text):
    return str(text).replace("|", "\\|")


def _render(root):
    """Render INDEX.md content for an already-validated root."""
    root = Path(root)
    entries, _ = vv.load_catalog(root)
    entries = sorted(entries, key=lambda e: int(str(e["id"])[1:]))

    status_counts = {s: 0 for s in STATUS_KEYS}
    n_blind = 0
    n_primed = 0
    rows_by_tier = {t: [] for t in range(6)}

    for entry in entries:
        slug = str(entry["slug"])
        tier = int(entry["tier"])
        vpath = root / slug / "VERDICT.md"
        if vpath.is_file():
            fm = _load_frontmatter(vpath)
            status = fm["status"]
            ver = fm["verification"]
            if ver == "blind":
                n_blind += 1
            else:
                n_primed += 1
            verifs = fm.get("verifications") or []
            df_ver = sum(1 for v in verifs if isinstance(v, dict)
                         and v.get("mode") in ("blind", "refute"))
            claims_cell = _claims_summary(fm.get("claims") or [])
            evidence_count = len(fm.get("evidence_manifest") or [])
        else:
            status = "OPEN"
            ver = "-"
            df_ver = 0
            claims_cell = "-"
            evidence_count = 0
        status_counts[status] += 1
        rows_by_tier[tier].append(
            "| %s | %s | %s | %s | %d | %s | %d | %s |"
            % (entry["id"], _escape_cell(entry["hypothesis"]), status, ver,
               df_ver, claims_cell, evidence_count, slug))

    lines = [HEADER_LINE,
             "<!-- INPUTS_DIGEST: %s -->" % inputs_digest(root),
             "# Living Formula v2.3 - Index",
             "",
             "## Status Legend",
             ""]
    lines.extend(LEGEND_LINES)
    lines.extend([
        "",
        "## Summary",
        "",
        "- FALSIFIED: %d" % status_counts["FALSIFIED"],
        "- UNSUPPORTED: %d" % status_counts["UNSUPPORTED"],
        "- PARTIALLY_VERIFIED: %d" % status_counts["PARTIALLY_VERIFIED"],
        "- VERIFIED: %d" % status_counts["VERIFIED"],
        "- OPEN: %d" % status_counts["OPEN"],
        "- Blind verdicts: %d" % n_blind,
        "- Primed verdicts: %d" % n_primed,
    ])
    for tier in range(6):
        lines.extend([
            "",
            "## Tier %d" % tier,
            "",
            "| Q | Hypothesis | Status | Ver | Df(ver) | Claims | "
            "Evidence | Dir |",
            "|---|---|---|---|---|---|---|---|",
        ])
        lines.extend(rows_by_tier[tier])
    return "\n".join(lines) + "\n"


def build_index(root):
    """Validate everything under root, then return INDEX.md content."""
    errors = validate_all(root)
    if errors:
        raise ValidationFailure(errors)
    return _render(root)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate (or check) the deterministic v2_3 INDEX.md.")
    parser.add_argument("--root", default=None,
                        help="v2_3 root dir (default: parent of _meta)")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and byte-compare with "
                             "the existing INDEX.md")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else _HERE.parent

    errors = validate_all(root)
    if errors:
        for code, fpath, detail in errors:
            sys.stderr.write("ERROR %s %s: %s\n" % (code, fpath, detail))
        return 1

    content = _render(root)
    index_path = root / "INDEX.md"
    if args.check:
        if not index_path.is_file():
            sys.stderr.write("ERROR CHECK %s: INDEX.md does not exist\n"
                             % index_path)
            return 1
        if index_path.read_bytes() != content.encode("utf-8"):
            sys.stderr.write("ERROR CHECK %s: INDEX.md differs from "
                             "regenerated content\n" % index_path)
            return 1
        return 0
    with open(str(index_path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())

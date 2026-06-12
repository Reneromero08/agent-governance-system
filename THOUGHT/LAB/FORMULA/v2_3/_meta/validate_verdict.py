#!/usr/bin/env python
"""validate_verdict.py - validate one v2_3 VERDICT.md against verdict/v2.

CLI:
    python validate_verdict.py <path-to-VERDICT.md> [--root <v2_3 root>]

Importable:
    from validate_verdict import validate
    errors = validate(verdict_path, root)   # list of (code, detail) tuples

Exit 0 when clean, 1 on any error. Errors are printed to stderr as:
    ERROR <CODE> <file>: <detail>

Pure stdlib + PyYAML. ASCII only. Deterministic (no wall-clock, no
randomness, sorted iteration).
"""

import argparse
import datetime
import hashlib
import re
import sys
from pathlib import Path

import yaml

STATUS_ORDER = {
    "FALSIFIED": 0,
    "UNSUPPORTED": 1,
    "PARTIALLY_VERIFIED": 2,
    "VERIFIED": 3,
}
STATUSES = ("FALSIFIED", "UNSUPPORTED", "PARTIALLY_VERIFIED", "VERIFIED")
VERIFICATION_MODES = ("blind", "primed")
VERIF_ENTRY_MODES = ("blind", "primed", "refute")
VERIF_RESULTS = STATUSES + ("UNREFUTED",)

TOP_KEYS = (
    "schema", "question", "slug", "date", "status", "verification",
    "packet_sha256", "predecessor", "method_summary", "registry_ids",
    "prediction_ids", "claims", "evidence_manifest", "verifications",
)
CLAIM_KEYS = ("id", "text", "status", "falsifier", "key_results", "evidence")
VERIF_KEYS = ("date", "mode", "result")

REQUIRED_SECTIONS = (
    "## Hypothesis", "## Claims", "## Method",
    "## Results", "## Status", "## Provenance",
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
QUESTION_RE = re.compile(r"^Q([1-9][0-9]?)$")
SLUG_RE = re.compile(r"^q([0-9]{2})_[a-z0-9_]+$")
CLAIM_ID_RE = re.compile(r"^C[0-9]+$")
REGISTRY_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]*-[A-Z0-9]+-[0-9]+$")
PREDICTION_ID_RE = re.compile(r"^P-[0-9]{3}$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")


def sha256_file(path):
    """Return the sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_repo_root(start):
    """Walk up from start to the first dir containing .git; else start."""
    p = Path(start).resolve()
    for cand in [p] + list(p.parents):
        if (cand / ".git").exists():
            return cand
    return p


def split_frontmatter(text):
    """Split text into (frontmatter_yaml, body) or None if no --- block."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    return None


def load_registry_ids(root):
    """Set of registry IDs found as table rows in VARIABLES.md, or None."""
    path = Path(root) / "VARIABLES.md"
    if not path.is_file():
        return None
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and REGISTRY_ID_RE.match(cells[0]):
            ids.add(cells[0])
    return ids


def load_prediction_ids(root):
    """Set of P-NNN IDs found as table rows in PREDICTIONS.md, or None."""
    path = Path(root) / "PREDICTIONS.md"
    if not path.is_file():
        return None
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and PREDICTION_ID_RE.match(cells[0]):
            ids.add(cells[0])
    return ids


def load_catalog(root):
    """Load _meta/questions.yaml.

    Returns (entries_or_None, errors) where errors is a list of
    (code, detail) tuples (all E_CATALOG).
    """
    path = Path(root) / "_meta" / "questions.yaml"
    if not path.is_file():
        return None, [("E_CATALOG", "questions.yaml not found at %s" % path)]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [("E_CATALOG", "questions.yaml unparseable: %s" % exc)]
    if not isinstance(data, list):
        return None, [("E_CATALOG", "questions.yaml is not a YAML list")]
    errors = []
    entries = []
    seen_ids = set()
    seen_slugs = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not all(
                k in entry for k in ("id", "slug", "tier", "hypothesis",
                                     "predecessor")):
            errors.append(("E_CATALOG",
                           "catalog entry %d malformed (need id, slug, tier,"
                           " hypothesis, predecessor)" % i))
            continue
        qid = str(entry["id"])
        slug = str(entry["slug"])
        if qid in seen_ids:
            errors.append(("E_CATALOG", "duplicate question id %s" % qid))
        if slug in seen_slugs:
            errors.append(("E_CATALOG", "duplicate slug %s" % slug))
        seen_ids.add(qid)
        seen_slugs.add(slug)
        entries.append(entry)
    return entries, errors


def _valid_date(value):
    if isinstance(value, datetime.datetime):
        return False
    if isinstance(value, datetime.date):
        return True
    if isinstance(value, str) and DATE_RE.match(value):
        try:
            datetime.date.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False


def _manifest_path_problem(qdir, rel):
    """Return a problem string for a manifest path, or None when fine."""
    if not isinstance(rel, str) or not rel:
        return "path must be a non-empty string"
    if rel.startswith(("/", "\\")) or DRIVE_RE.match(rel) \
            or Path(rel).is_absolute():
        return "absolute path not allowed"
    if ".." in Path(rel).parts:
        return "path escapes the question dir"
    target = Path(qdir) / rel
    try:
        target.resolve().relative_to(Path(qdir).resolve())
    except ValueError:
        return "path escapes the question dir"
    if not target.is_file():
        return "file not found on disk"
    return None


def validate(verdict_path, root=None):
    """Validate one VERDICT.md. Returns a list of (code, detail) tuples."""
    errors = []
    vpath = Path(verdict_path).resolve()
    qdir = vpath.parent
    root_path = Path(root).resolve() if root is not None else qdir.parent

    try:
        raw = vpath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [("E_YAML", "cannot read file: %s" % exc)]

    split = split_frontmatter(raw)
    if split is None:
        return [("E_YAML", "frontmatter missing (expected --- delimited "
                           "YAML block at top of file)")]
    fm_text, body = split

    try:
        fm = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        return [("E_YAML", "frontmatter unparseable: %s" % exc)]
    if not isinstance(fm, dict):
        return [("E_YAML", "frontmatter is not a YAML mapping")]

    # ---- field presence / unknown keys -------------------------------
    for key in TOP_KEYS:
        if key not in fm:
            errors.append(("E_SCHEMA", "missing field: %s" % key))
    for key in sorted(set(fm) - set(TOP_KEYS)):
        errors.append(("E_SCHEMA", "unknown field: %s" % key))

    # ---- scalar fields ----------------------------------------------
    if "schema" in fm and fm.get("schema") != "verdict/v2":
        errors.append(("E_SCHEMA",
                       "schema must be verdict/v2, got %s"
                       % ascii(fm.get("schema"))))

    qnum = None
    question = fm.get("question")
    if "question" in fm:
        m = QUESTION_RE.match(question) if isinstance(question, str) else None
        if not m or not 1 <= int(m.group(1)) <= 57:
            errors.append(("E_SCHEMA",
                           "question must be Q<N>, N in 1..57, no zero "
                           "padding, got %s" % ascii(question)))
        else:
            qnum = int(m.group(1))

    slug = fm.get("slug")
    if "slug" in fm:
        m = SLUG_RE.match(slug) if isinstance(slug, str) else None
        if not m:
            errors.append(("E_SCHEMA",
                           "slug must be q<NN>_<name>, got %s" % ascii(slug)))
        else:
            if slug != qdir.name:
                errors.append(("E_SCHEMA",
                               "slug %s does not match directory name %s"
                               % (slug, qdir.name)))
            if qnum is not None and int(m.group(1)) != qnum:
                errors.append(("E_SCHEMA",
                               "slug number %s does not match question Q%d"
                               % (m.group(1), qnum)))

    if "date" in fm and not _valid_date(fm.get("date")):
        errors.append(("E_SCHEMA",
                       "date must be YYYY-MM-DD, got %s"
                       % ascii(fm.get("date"))))

    status = fm.get("status")
    status_ok = isinstance(status, str) and status in STATUS_ORDER
    if "status" in fm and not status_ok:
        errors.append(("E_SCHEMA",
                       "status must be one of %s, got %s"
                       % ("|".join(STATUSES), ascii(status))))

    verification = fm.get("verification")
    if "verification" in fm and verification not in VERIFICATION_MODES:
        errors.append(("E_SCHEMA",
                       "verification must be blind or primed, got %s"
                       % ascii(verification)))

    packet = fm.get("packet_sha256")
    if "packet_sha256" in fm and packet is not None and not (
            isinstance(packet, str) and HEX64_RE.match(packet)):
        errors.append(("E_SCHEMA",
                       "packet_sha256 must be null or 64 lowercase hex "
                       "chars, got %s" % ascii(packet)))

    pred = fm.get("predecessor")
    if "predecessor" in fm and pred is not None:
        if not isinstance(pred, str) or not pred:
            errors.append(("E_SCHEMA",
                           "predecessor must be null or a repo-relative "
                           "path string"))
        else:
            repo_root = find_repo_root(root_path)
            candidates = [repo_root / pred, root_path / pred]
            if not any(c.exists() for c in candidates):
                errors.append(("E_PREDECESSOR",
                               "predecessor path does not exist: %s" % pred))

    ms = fm.get("method_summary")
    if "method_summary" in fm and (not isinstance(ms, str)
                                   or not ms.strip() or "\n" in ms):
        errors.append(("E_SCHEMA",
                       "method_summary must be a non-empty one-line string"))

    # ---- registry_ids -------------------------------------------------
    reg = fm.get("registry_ids")
    if "registry_ids" in fm:
        if not isinstance(reg, list) or any(
                not isinstance(r, str) for r in reg):
            errors.append(("E_SCHEMA",
                           "registry_ids must be a list of strings"))
        elif reg:
            known = load_registry_ids(root_path)
            if known is None:
                errors.append(("E_REGISTRY",
                               "VARIABLES.md not found under %s" % root_path))
            else:
                for rid in reg:
                    if rid not in known:
                        errors.append(("E_REGISTRY",
                                       "registry id not found in "
                                       "VARIABLES.md: %s" % rid))

    # ---- prediction_ids ------------------------------------------------
    pids = fm.get("prediction_ids")
    if "prediction_ids" in fm:
        if not isinstance(pids, list) or any(
                not isinstance(p, str) for p in pids):
            errors.append(("E_SCHEMA",
                           "prediction_ids must be a list of strings"))
        else:
            if verification == "blind" and not pids:
                errors.append(("E_PREDICTION",
                               "blind verdict must have non-empty "
                               "prediction_ids"))
            if pids:
                known = load_prediction_ids(root_path)
                if known is None:
                    errors.append(("E_PREDICTION",
                                   "PREDICTIONS.md not found under %s"
                                   % root_path))
                else:
                    for pid in pids:
                        if pid not in known:
                            errors.append(("E_PREDICTION",
                                           "prediction id not found in "
                                           "PREDICTIONS.md: %s" % pid))

    # ---- evidence_manifest ----------------------------------------------
    manifest = fm.get("evidence_manifest")
    manifest_ok = isinstance(manifest, list)
    manifest_paths = set()
    if "evidence_manifest" in fm and not manifest_ok:
        errors.append(("E_SCHEMA", "evidence_manifest must be a list"))
    if manifest_ok:
        for i, entry in enumerate(manifest):
            label = "evidence_manifest[%d]" % i
            if not isinstance(entry, dict):
                errors.append(("E_SCHEMA",
                               "%s must be a map with keys path, sha256"
                               % label))
                continue
            if set(entry) != {"path", "sha256"}:
                errors.append(("E_SCHEMA",
                               "%s keys must be exactly path, sha256"
                               % label))
                continue
            rel = entry["path"]
            digest = entry["sha256"]
            if isinstance(rel, str):
                manifest_paths.add(rel)
            problem = _manifest_path_problem(qdir, rel)
            if problem:
                errors.append(("E_MANIFEST_MISSING",
                               "%s: %s (%s)" % (label, problem, ascii(rel))))
            digest_ok = isinstance(digest, str) and HEX64_RE.match(digest)
            if not digest_ok:
                errors.append(("E_SCHEMA",
                               "%s sha256 must be 64 lowercase hex chars"
                               % label))
            if not problem and digest_ok:
                actual = sha256_file(Path(qdir) / rel)
                if actual != digest:
                    errors.append(("E_HASH",
                                   "%s: sha256 mismatch for %s (manifest "
                                   "%s..., actual %s...)"
                                   % (label, rel, digest[:12], actual[:12])))

    # ---- claims ----------------------------------------------------------
    claims = fm.get("claims")
    claims_ok = isinstance(claims, list) and len(claims) > 0
    if "claims" in fm and not claims_ok:
        errors.append(("E_SCHEMA", "claims must be a non-empty list"))
    claim_statuses = []
    all_claim_statuses_valid = True
    parsed_claims = []
    if claims_ok:
        for i, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(("E_SCHEMA", "claims[%d] must be a map" % i))
                all_claim_statuses_valid = False
                continue
            cid_val = claim.get("id")
            label = cid_val if isinstance(cid_val, str) and cid_val \
                else "claims[%d]" % i
            for k in sorted(set(claim) - set(CLAIM_KEYS)):
                errors.append(("E_SCHEMA",
                               "%s: unknown claim key: %s" % (label, k)))
            for k in ("id", "text", "status", "key_results", "evidence"):
                if k not in claim:
                    errors.append(("E_SCHEMA",
                                   "%s: missing claim key: %s" % (label, k)))
            fals = claim.get("falsifier")
            if not isinstance(fals, str) or not fals.strip():
                errors.append(("E_FALSIFIER",
                               "%s: falsifier missing or empty" % label))
            if "id" in claim and (not isinstance(cid_val, str)
                                  or not CLAIM_ID_RE.match(cid_val)):
                errors.append(("E_SCHEMA",
                               "%s: claim id must be C<N>" % label))
            text_val = claim.get("text")
            if "text" in claim and (not isinstance(text_val, str)
                                    or not text_val.strip()):
                errors.append(("E_SCHEMA",
                               "%s: claim text must be a non-empty string"
                               % label))
            cstat = claim.get("status")
            if "status" in claim and isinstance(cstat, str) \
                    and cstat in STATUS_ORDER:
                claim_statuses.append(cstat)
            else:
                if "status" in claim:
                    errors.append(("E_SCHEMA",
                                   "%s: bad claim status %s"
                                   % (label, ascii(cstat))))
                all_claim_statuses_valid = False
            kr_list = []
            kr = claim.get("key_results")
            if "key_results" in claim:
                if not isinstance(kr, list) or any(
                        not isinstance(s, str) for s in kr):
                    errors.append(("E_SCHEMA",
                                   "%s: key_results must be a list of "
                                   "strings" % label))
                else:
                    kr_list = kr
            ev_list = None
            ev = claim.get("evidence")
            if "evidence" in claim:
                if not isinstance(ev, list) or any(
                        not isinstance(s, str) for s in ev):
                    errors.append(("E_SCHEMA",
                                   "%s: evidence must be a list of strings"
                                   % label))
                else:
                    ev_list = ev
                    for pth in ev:
                        if pth not in manifest_paths:
                            errors.append(("E_SCHEMA",
                                           "%s: evidence path not in "
                                           "evidence_manifest: %s"
                                           % (label, pth)))
            # Floor rules (E_FLOOR)
            if isinstance(cstat, str) and cstat in ("VERIFIED",
                                                    "PARTIALLY_VERIFIED"):
                if not ev_list:
                    errors.append(("E_FLOOR",
                                   "%s: status %s requires at least 1 "
                                   "evidence path" % (label, cstat)))
                if manifest_ok and len(manifest) == 0:
                    errors.append(("E_FLOOR",
                                   "%s: empty evidence_manifest caps claim "
                                   "status at UNSUPPORTED, got %s"
                                   % (label, cstat)))
            parsed_claims.append((label, kr_list))

    # ---- status MIN rule -------------------------------------------------
    if status_ok and claims_ok and all_claim_statuses_valid \
            and claim_statuses:
        min_status = min(claim_statuses, key=lambda s: STATUS_ORDER[s])
        if status != min_status:
            errors.append(("E_STATUS_NOT_MIN",
                           "frontmatter status %s != MIN(claim statuses) %s"
                           % (status, min_status)))

    # ---- verifications ----------------------------------------------------
    verifs = fm.get("verifications")
    if "verifications" in fm:
        if not isinstance(verifs, list):
            errors.append(("E_SCHEMA", "verifications must be a list"))
        else:
            for i, v in enumerate(verifs):
                label = "verifications[%d]" % i
                if not isinstance(v, dict):
                    errors.append(("E_SCHEMA", "%s must be a map" % label))
                    continue
                if set(v) != set(VERIF_KEYS):
                    errors.append(("E_SCHEMA",
                                   "%s keys must be exactly date, mode, "
                                   "result" % label))
                    continue
                if not _valid_date(v["date"]):
                    errors.append(("E_SCHEMA",
                                   "%s: date must be YYYY-MM-DD" % label))
                if v["mode"] not in VERIF_ENTRY_MODES:
                    errors.append(("E_SCHEMA",
                                   "%s: mode must be blind, primed, or "
                                   "refute" % label))
                if v["result"] not in VERIF_RESULTS:
                    errors.append(("E_SCHEMA",
                                   "%s: result must be a status or "
                                   "UNREFUTED" % label))

    # ---- body sections -----------------------------------------------------
    body_lines = body.split("\n")
    h2_positions = [i for i, ln in enumerate(body_lines)
                    if ln.startswith("## ")]
    first_idx = {}
    for i in h2_positions:
        heading = body_lines[i].rstrip()
        if heading in REQUIRED_SECTIONS and heading not in first_idx:
            first_idx[heading] = i
    missing_sections = [h for h in REQUIRED_SECTIONS if h not in first_idx]
    if missing_sections:
        errors.append(("E_SCHEMA",
                       "missing required body sections: %s"
                       % ", ".join(missing_sections)))
    else:
        idxs = [first_idx[h] for h in REQUIRED_SECTIONS]
        if idxs != sorted(idxs):
            errors.append(("E_SCHEMA",
                           "body sections out of required order "
                           "(Hypothesis, Claims, Method, Results, Status, "
                           "Provenance)"))

    def section_bounds(heading):
        if heading not in first_idx:
            return None, None
        start = first_idx[heading]
        later = [j for j in h2_positions if j > start]
        end = later[0] if later else len(body_lines)
        return start, end

    # ---- body Status line ----------------------------------------------
    status_line_idxs = [i for i, ln in enumerate(body_lines)
                        if ln.startswith("**Status:**")]
    if len(status_line_idxs) != 1:
        errors.append(("E_BODY_MISMATCH",
                       "body must contain exactly one line beginning with "
                       "**Status:**, found %d" % len(status_line_idxs)))
    elif status_ok:
        line = body_lines[status_line_idxs[0]].rstrip()
        if line != "**Status:** %s" % status:
            errors.append(("E_BODY_MISMATCH",
                           "body status line %s != frontmatter status %s"
                           % (ascii(line), status)))
        else:
            s_start, s_end = section_bounds("## Status")
            if s_start is not None and not (
                    s_start < status_line_idxs[0] < s_end):
                errors.append(("E_BODY_MISMATCH",
                               "**Status:** line is outside the ## Status "
                               "section"))

    # ---- key_results verbatim in Results ----------------------------------
    r_start, r_end = section_bounds("## Results")
    if r_start is not None:
        results_text = "\n".join(body_lines[r_start + 1:r_end])
        for label, kr_list in parsed_claims:
            for s in kr_list:
                if s not in results_text:
                    errors.append(("E_KEYNUM",
                                   "%s: key result not found verbatim in "
                                   "## Results: %s" % (label, ascii(s))))

    # ---- blind packet ------------------------------------------------------
    if verification == "blind":
        if not (isinstance(packet, str) and HEX64_RE.match(packet)):
            errors.append(("E_BLIND",
                           "blind verdict requires a non-null 64-hex "
                           "packet_sha256"))
        else:
            pdir = qdir / "_packets"
            matched = False
            if pdir.is_dir():
                for f in sorted(pdir.iterdir(), key=lambda p: p.name):
                    if f.is_file() and sha256_file(f) == packet:
                        matched = True
                        break
            if not matched:
                errors.append(("E_BLIND",
                               "no file in %s/_packets matches "
                               "packet_sha256" % qdir.name))

    # ---- catalog ----------------------------------------------------------
    entries, cat_errors = load_catalog(root_path)
    errors.extend(cat_errors)
    if entries is not None:
        by_slug = {}
        for entry in entries:
            if isinstance(entry.get("slug"), str):
                by_slug[entry["slug"]] = entry
        if qdir.name not in by_slug:
            errors.append(("E_CATALOG",
                           "question dir slug not present in "
                           "_meta/questions.yaml: %s" % qdir.name))
        elif isinstance(question, str) \
                and by_slug[qdir.name].get("id") != question:
            errors.append(("E_CATALOG",
                           "frontmatter question %s != catalog id %s for "
                           "slug %s" % (question,
                                        by_slug[qdir.name].get("id"),
                                        qdir.name)))

    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one v2_3 VERDICT.md (verdict/v2 schema).")
    parser.add_argument("verdict", help="path to VERDICT.md")
    parser.add_argument("--root", default=None,
                        help="v2_3 root (default: parent of question dir)")
    args = parser.parse_args(argv)
    errors = validate(args.verdict, args.root)
    for code, detail in errors:
        sys.stderr.write("ERROR %s %s: %s\n" % (code, args.verdict, detail))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

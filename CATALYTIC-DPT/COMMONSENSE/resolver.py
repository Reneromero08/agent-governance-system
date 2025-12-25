"""Deterministic resolver with symbolic fact expansion (Phase 2).

Adds:
  - optional CODEBOOK.json for symbolic handles
  - expands facts before resolution (translate.expand_facts)

Determinism:
  - stable candidate sort: priority desc, confidence desc, id asc
  - fact expansion de-dupes in-order
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

RESOLVER_DIR = Path(__file__).resolve().parent
if str(RESOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(RESOLVER_DIR))

from translate import expand_facts


RESOLVER_VERSION = "0.2.0"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_facts(path: Path) -> List[str]:
    if path.suffix.lower() in {".json"}:
        obj = load_json(path)
        if isinstance(obj, dict) and "facts" in obj and isinstance(obj["facts"], list):
            return [str(x) for x in obj["facts"]]
        raise ValueError("facts json must be an object: {\"facts\": [...]}")
    # text
    lines = path.read_text(encoding="utf-8").splitlines()
    facts = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        facts.append(s)
    return facts


def applies(entry: Dict[str, Any], facts: Set[str]) -> Tuple[bool, str]:
    scope = entry.get("scope") or {}
    applies_when = scope.get("applies_when") or []
    not_when = scope.get("not_when") or []
    # ALL applies_when required
    for p in applies_when:
        if p not in facts:
            return False, "missing_required_predicate"
    # ANY veto predicate disables
    for p in not_when:
        if p in facts:
            return False, "veto_predicate"
    return True, "ok"


def candidate_key(entry: Dict[str, Any]) -> Tuple[int, float, str]:
    pr = int(entry.get("priority", 0))
    conf = float(entry.get("confidence", 0.0))
    return (-pr, -conf, str(entry["id"]))


def apply_rule_effects(rule: Dict[str, Any], derived: Set[str], emits: List[Any], trace: List[Dict[str, Any]], step: int) -> int:
    for eff in rule.get("then", []):
        op = eff.get("op")
        path = eff.get("path", "")
        value = eff.get("value")
        note = eff.get("note")
        if op == "emit":
            emits.append(value)
            trace.append({"step": step, "event": "emit", "detail": {"value": value, "note": note}})
            step += 1
            continue

        # derived facts operations
        if path.startswith("facts") or path == "facts":
            if op in ("set", "add"):
                if isinstance(value, list):
                    for v in value:
                        derived.add(str(v))
                elif value is not None:
                    derived.add(str(value))
                trace.append({"step": step, "event": op, "detail": {"fact": value, "note": note}})
                step += 1
            elif op in ("unset", "remove"):
                if isinstance(value, list):
                    for v in value:
                        derived.discard(str(v))
                elif value is not None:
                    derived.discard(str(value))
                trace.append({"step": step, "event": op, "detail": {"fact": value, "note": note}})
                step += 1
            else:
                trace.append({"step": step, "event": "skip_unknown_op", "detail": {"op": op, "path": path}})
                step += 1
        else:
            trace.append({"step": step, "event": "skip_unknown_path", "detail": {"op": op, "path": path}})
            step += 1
    return step


def rule_matches(rule: Dict[str, Any], facts: Set[str]) -> bool:
    if_all = rule.get("if_all") or []
    if_any = rule.get("if_any") or []
    unless = rule.get("unless") or []
    for p in if_all:
        if p not in facts:
            return False
    if if_any:
        ok = any(p in facts for p in if_any)
        if not ok:
            return False
    for p in unless:
        if p in facts:
            return False
    return True


def resolve(db: List[Dict[str, Any]], input_facts: List[str], codebook: Dict[str, Any] | None = None, with_trace: bool = False) -> Dict[str, Any]:
    db_ids = [str(e.get("id")) for e in db if isinstance(e, dict) and e.get("id")]
    expanded_facts, unresolved = expand_facts(input_facts, codebook or {}, db_ids=db_ids)

    facts = set(expanded_facts)
    derived: Set[str] = set()
    emits: List[Any] = []
    selected: List[str] = []
    skipped: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    step = 0

    candidates: List[Dict[str, Any]] = []
    for e in db:
        ok, reason = applies(e, facts)
        if ok:
            candidates.append(e)
        else:
            skipped.append({"id": e.get("id", "UNKNOWN"), "reason": reason})
    candidates.sort(key=candidate_key)

    if with_trace:
        trace.append({"step": step, "event": "facts_expanded", "detail": {"input": input_facts, "expanded": expanded_facts, "unresolved": unresolved}})
        step += 1
        trace.append({"step": step, "event": "candidates_sorted", "detail": {"count": len(candidates)}})
        step += 1

    for e in candidates:
        selected.append(e["id"])
        if e.get("kind") == "logic_rule":
            rule = e.get("rule") or {}
            if rule_matches(rule, facts | derived):
                if with_trace:
                    trace.append({"step": step, "event": "rule_match", "id": e["id"]})
                    step += 1
                step = apply_rule_effects(rule, derived, emits, trace, step)
            else:
                if with_trace:
                    trace.append({"step": step, "event": "rule_no_match", "id": e["id"]})
                    step += 1

    out: Dict[str, Any] = {
        "resolver_version": RESOLVER_VERSION,
        "input_facts": list(input_facts),
        "selected_ids": selected,
        "skipped": skipped,
        "derived_facts": sorted(derived),
        "emits": emits,
        "tiebreak": ["priority_desc", "confidence_desc", "id_asc"],
        "unresolved_symbols": unresolved,
        "expanded_facts": expanded_facts,
    }
    if with_trace:
        out["trace"] = trace
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to db.json (list of entries)")
    ap.add_argument("--facts", required=True, help="Path to facts.json or facts.txt")
    ap.add_argument("--out", required=True, help="Output path for result.json")
    ap.add_argument("--codebook", help="Optional CODEBOOK.json for symbolic fact expansion")
    ap.add_argument("--trace", action="store_true", help="Include trace events (debug)")
    args = ap.parse_args()

    db_path = Path(args.db)
    facts_path = Path(args.facts)
    out_path = Path(args.out)
    codebook_path = Path(args.codebook) if args.codebook else None

    db = load_json(db_path)
    if not isinstance(db, list):
        raise SystemExit("DB must be a JSON list of entries")

    facts = load_facts(facts_path)
    codebook = load_json(codebook_path) if codebook_path else None

    result = resolve(db, facts, codebook=codebook, with_trace=bool(args.trace))
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

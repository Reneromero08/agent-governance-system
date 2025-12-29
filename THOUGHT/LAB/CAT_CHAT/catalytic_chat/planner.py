#!/usr/bin/env python3
"""
Deterministic Planner (Phase 4)

Compiles high-level request messages into deterministic step sequences.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from catalytic_chat.section_indexer import SectionIndexer
from catalytic_chat.symbol_registry import SymbolRegistry
from catalytic_chat.symbol_resolver import SymbolResolver
from catalytic_chat.slice_resolver import SliceResolver
from catalytic_chat.message_cassette import MessageCassette, MessageCassetteError, _generate_id


class PlannerError(Exception):
    pass


class Planner:
    
    VERSION = "4.0.0"
    
    def __init__(self, repo_root: Optional[Path] = None, substrate_mode: str = "sqlite"):
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = repo_root
        self.substrate_mode = substrate_mode
        
        self._section_indexer = SectionIndexer(repo_root=repo_root, substrate_mode=substrate_mode)
        self._symbol_registry = SymbolRegistry(repo_root=repo_root, substrate_mode=substrate_mode)
        self._slice_resolver = SliceResolver()
        
        symbol_registry = SymbolRegistry(repo_root=repo_root, substrate_mode=substrate_mode)
        self._symbol_resolver = SymbolResolver(repo_root=repo_root, substrate_mode=substrate_mode, symbol_registry=symbol_registry)
    
    def _validate_request(self, request: Dict[str, Any]) -> None:
        if "run_id" not in request:
            raise PlannerError("Missing required field: run_id")
        if "request_id" not in request:
            raise PlannerError("Missing required field: request_id")
        if "intent" not in request:
            raise PlannerError("Missing required field: intent")
        if "budgets" not in request:
            raise PlannerError("Missing required field: budgets")
        if "max_steps" not in request["budgets"]:
            raise PlannerError("Missing required field: budgets.max_steps")
        
        inputs = request.get("inputs", {})
        budgets = request["budgets"]
        
        if "symbols" in inputs:
            for symbol_id in inputs["symbols"]:
                if not symbol_id.startswith("@"):
                    raise PlannerError(f"Invalid symbol_id (must start with @): {symbol_id}")
        
        if budgets.get("max_steps", 100) < 1:
            raise PlannerError("max_steps must be at least 1")
        if budgets.get("max_bytes", 10000000) < 0:
            raise PlannerError("max_bytes must be non-negative")
        if budgets.get("max_symbols", 100) < 0:
            raise PlannerError("max_symbols must be non-negative")
    
    def _canonicalize_step(self, step: Dict[str, Any]) -> str:
        required = ["step_id", "ordinal", "op"]
        for field in required:
            if field not in step:
                raise PlannerError(f"Step missing required field: {field}")
        
        if step["op"] == "READ_SYMBOL" and "symbol_id" not in step.get("refs", {}):
            raise PlannerError(f"READ_SYMBOL step missing refs.symbol_id")
        if step["op"] == "READ_SECTION" and "section_id" not in step.get("refs", {}):
            raise PlannerError(f"READ_SECTION step missing refs.section_id")
        
        step_for_hash = {
            "step_id": step["step_id"],
            "ordinal": step["ordinal"],
            "op": step["op"],
            "refs": step.get("refs", {}),
            "constraints": step.get("constraints", {}),
            "expected_outputs": step.get("expected_outputs", {})
        }
        
        return json.dumps(step_for_hash, sort_keys=True)
    
    def _compute_step_id(self, canonical_json: str) -> str:
        return f"step_{hashlib.sha256(canonical_json.encode()).hexdigest()[:16]}"
    
    def _compute_plan_hash(self, run_id: str, request_id: str, steps: List[Dict[str, Any]]) -> str:
        canonical_steps = [self._canonicalize_step(step) for step in steps]
        
        canonical_plan = json.dumps({
            "run_id": run_id,
            "request_id": request_id,
            "steps": canonical_steps
        }, sort_keys=True)
        
        return hashlib.sha256(canonical_plan.encode()).hexdigest()
    
    def _estimate_bytes(self, inputs: Dict[str, Any], dry_run: bool = False) -> int:
        total_bytes = 0
        
        if "files" in inputs:
            for file_path in inputs["files"]:
                full_path = self.repo_root / file_path
                if full_path.exists() and full_path.is_file():
                    total_bytes += full_path.stat().st_size
        
        if not dry_run and "symbols" in inputs:
            for symbol_id in inputs["symbols"]:
                try:
                    entry = self._symbol_registry.get_symbol(symbol_id)
                    if entry is None:
                        continue
                    
                    section = self._section_indexer.get_section_by_id(entry.target_ref)
                    if section is None:
                        continue
                    
                    section_content, _, _, _, _ = self._section_indexer.get_section_content(
                        entry.target_ref, 
                        entry.default_slice if entry.default_slice else None
                    )
                    
                    if section_content:
                        total_bytes += len(section_content.encode('utf-8'))
                except Exception:
                    pass
        
        return total_bytes
    
    def _estimate_symbols(self, inputs: Dict[str, Any], dry_run: bool = False) -> int:
        if "symbols" not in inputs:
            return 0
        return len(set(inputs["symbols"]))
    
    def plan_request(self, request: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        self._validate_request(request)
        
        run_id = request["run_id"]
        request_id = request["request_id"]
        intent = request["intent"]
        inputs = request.get("inputs", {})
        budgets = request["budgets"]
        
        max_steps = budgets.get("max_steps", 100)
        max_bytes = budgets.get("max_bytes", 10000000)
        max_symbols = budgets.get("max_symbols", 100)
        
        estimated_bytes = self._estimate_bytes(inputs, dry_run=dry_run)
        estimated_symbols = self._estimate_symbols(inputs, dry_run=dry_run)
        
        if estimated_bytes > max_bytes:
            raise PlannerError(f"Budget exceeded: estimated_bytes={estimated_bytes} > max_bytes={max_bytes}")
        if estimated_symbols > max_symbols:
            raise PlannerError(f"Budget exceeded: estimated_symbols={estimated_symbols} > max_symbols={max_symbols}")
        
        symbols_to_resolve = inputs.get("symbols", [])
        
        resolved_symbols = []
        for symbol_id in symbols_to_resolve:
            try:
                if dry_run:
                    entry = self._symbol_registry.get_symbol(symbol_id)
                    
                    if entry is not None:
                        if entry.default_slice and entry.default_slice.lower() == "all":
                            raise PlannerError(f"slice=ALL is forbidden for symbol {symbol_id}")
                        slice_expr = entry.default_slice if entry.default_slice else None
                        resolved_symbols.append({
                            "symbol_id": symbol_id,
                            "section_id": entry.target_ref,
                            "slice_expr": slice_expr
                        })
                    else:
                        resolved_symbols.append({
                            "symbol_id": symbol_id,
                            "section_id": None,
                            "slice_expr": None
                        })
                else:
                    entry = self._symbol_registry.get_symbol(symbol_id)
                    if entry is None:
                        raise PlannerError(f"Symbol not found: {symbol_id}")
                    
                    section = self._section_indexer.get_section_by_id(entry.target_ref)
                    if section is None:
                        raise PlannerError(f"Section not found for symbol: {symbol_id} (section_id={entry.target_ref})")
                    
                    if entry.default_slice and entry.default_slice.lower() == "all":
                        raise PlannerError(f"slice=ALL is forbidden for symbol {symbol_id}")
                    
                    slice_expr = entry.default_slice if entry.default_slice else None
                    
                    payload, cache_hit = self._symbol_resolver.resolve(
                        symbol_id=symbol_id,
                        slice_expr=slice_expr,
                        run_id=run_id
                    )
                    
                    resolved_symbols.append({
                        "symbol_id": symbol_id,
                        "section_id": entry.target_ref,
                        "slice_expr": slice_expr,
                        "payload": payload
                    })
            except Exception as e:
                if not dry_run:
                    raise PlannerError(f"Failed to resolve symbol {symbol_id}: {e}")
        
        steps: List[Dict[str, Any]] = []
        
        for i, symbol_ref in enumerate(resolved_symbols):
            step_ordinal = len(steps)
            if step_ordinal + 1 > max_steps:
                raise PlannerError(f"Budget exceeded: step_ordinal={step_ordinal} would exceed max_steps={max_steps}")
            
            canonical_json = json.dumps({
                "ordinal": step_ordinal,
                "op": "READ_SYMBOL",
                "refs": {
                    "symbol_id": symbol_ref["symbol_id"],
                    "section_id": symbol_ref["section_id"],
                    "slice_expr": symbol_ref["slice_expr"]
                },
                "expected_outputs": {
                    "symbols_referenced": [symbol_ref["symbol_id"]]
                }
            }, sort_keys=True)
            
            step_id = self._compute_step_id(canonical_json)
            steps.append({
                "step_id": step_id,
                "ordinal": step_ordinal,
                "op": "READ_SYMBOL",
                "refs": {
                    "symbol_id": symbol_ref["symbol_id"]
                },
                "expected_outputs": {
                    "symbols_referenced": [symbol_ref["symbol_id"]]
                }
            })
        
        plan_output = {
            "run_id": run_id,
            "request_id": request_id,
            "planner_version": self.VERSION,
            "steps": steps,
            "plan_hash": self._compute_plan_hash(run_id, request_id, steps)
        }
        
        return plan_output


def post_request_and_plan(
    run_id: str,
    request_payload: Dict[str, Any],
    idempotency_key: Optional[str] = None,
    repo_root: Optional[Path] = None
) -> Tuple[str, str, List[str]]:
    cassette = MessageCassette(repo_root=repo_root)
    planner = Planner(repo_root=repo_root)
    
    try:
        conn = cassette._get_conn()
        
        intent = request_payload.get("intent", "")
        
        cursor = conn.execute("""
            SELECT m.message_id, j.job_id
            FROM cassette_messages m
            JOIN cassette_jobs j ON m.message_id = j.message_id
            WHERE m.run_id = ? AND m.idempotency_key = ? AND m.source = 'PLANNER'
        """, (run_id, idempotency_key))
        
        existing_row = cursor.fetchone()
        
        if existing_row:
            message_id = existing_row["message_id"]
            job_id = existing_row["job_id"]
            
            cursor = conn.execute("""
                SELECT step_id FROM cassette_steps
                WHERE job_id = ?
                ORDER BY ordinal
            """, (job_id,))
            
            steps_ids = [row["step_id"] for row in cursor.fetchall()]
            
            return (message_id, job_id, steps_ids)
        
        message_id = _generate_id("msg", run_id, idempotency_key or "")
        job_id = _generate_id("job", message_id)
        
        conn.execute("""
            INSERT INTO cassette_messages 
            (message_id, run_id, source, idempotency_key, payload_json)
            VALUES (?, ?, ?, ?, ?)
        """, (message_id, run_id, "PLANNER", idempotency_key, json.dumps(request_payload)))
        
        conn.execute("""
            INSERT INTO cassette_jobs 
            (job_id, message_id, intent, ordinal)
            VALUES (?, ?, ?, 1)
        """, (job_id, message_id, intent))
        
        plan_output = planner.plan_request(request_payload)
        
        steps_ids = []
        for step in plan_output["steps"]:
            step_id = step["step_id"]
            steps_ids.append(step_id)
            
            conn.execute("""
                INSERT INTO cassette_steps
                (step_id, job_id, ordinal, status, payload_json)
                VALUES (?, ?, ?, 'PENDING', ?)
            """, (step_id, job_id, step["ordinal"], json.dumps(step)))
        
        conn.commit()
        
        return (message_id, job_id, steps_ids)
        
    except (MessageCassetteError, PlannerError) as e:
        raise PlannerError(f"Failed to post request and plan: {e}")
    finally:
        cassette.close()


def verify_plan_stored(
    run_id: str,
    request_id: str,
    repo_root: Optional[Path] = None
) -> bool:
    cassette = MessageCassette(repo_root=repo_root)
    planner = Planner(repo_root=repo_root)
    
    try:
        conn = cassette._get_conn()
        
        cursor = conn.execute("""
            SELECT m.payload_json 
            FROM cassette_messages m
            WHERE m.run_id = ? AND m.idempotency_key = ? AND m.source = 'PLANNER'
        """, (run_id, request_id))
        
        row = cursor.fetchone()
        if row is None:
            raise PlannerError(f"No plan request found for run_id={run_id}, request_id={request_id}")
        
        stored_payload = json.loads(row["payload_json"])
        recomputed_plan = planner.plan_request(stored_payload)
        
        stored_plan_hash = recomputed_plan["plan_hash"]
        
        cursor = conn.execute("""
            SELECT j.ordinal, s.payload_json
            FROM cassette_jobs j
            JOIN cassette_steps s ON j.job_id = s.job_id
            JOIN cassette_messages m ON j.message_id = m.message_id
            WHERE m.run_id = ? AND m.idempotency_key = ? AND m.source = 'PLANNER'
            ORDER BY s.ordinal
        """, (run_id, request_id))
        
        stored_steps = []
        for job_row in cursor.fetchall():
            step_payload = json.loads(job_row["payload_json"])
            stored_steps.append(step_payload)
        
        recomputed_hash = planner._compute_plan_hash(
            run_id, request_id, stored_steps
        )
        
        return stored_plan_hash == recomputed_hash
        
    finally:
        cassette.close()

#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from campaign_orders import FAMILIES, MODES, PHASES, ROUTES, SEEDS, frozen_orders, gauge_symbols, keyed_rng, partition, symbol

SCHEMA = "CAT_CAS_PHASE6_COMBINED_OBSERVABILITY_PLAN_V1"


def persistence_events(session_id: str, sequence: list[str], orders: dict[str, list[int]]) -> list[dict[str, Any]]:
    structured = "FWD" if sequence.index("FWD") < sequence.index("REV") else "REV"
    randomized = "RND1" if sequence.index("RND1") < sequence.index("RND2") else "RND2"
    events = []
    for order_name in (structured, randomized):
        for input_type in ("impulse", "step"):
            for mode in MODES:
                for theta in PHASES:
                    events.append({
                        "event_id": f"{session_id}|persist|{order_name}|{input_type}|{mode}|{theta}",
                        "input_type": input_type,
                        "actual_mode": mode,
                        "theta_idx": theta,
                        "executed_tone_order": order_name,
                        "tone_execution_indices": orders[order_name],
                        "prepare_windows": 12 if input_type == "impulse" else 24,
                        "sender_off_windows": 8,
                        "sender_off_required": True,
                        "periodic_refresh_allowed": False,
                        "hidden_replay_allowed": False,
                    })
    return events


def trajectory(session_id: str, order_name: str, orders: dict[str, list[int]], length: int = 64) -> dict[str, Any]:
    rng = keyed_rng(session_id, order_name, "trajectory")
    steps = []
    for index in range(length):
        off = index % 16 in (12, 13, 14, 15)
        steps.append({
            "step": index,
            "drive_on": not off,
            "actual_mode": None if off else rng.choice(MODES),
            "theta_idx": None if off else rng.randrange(8),
            "amplitude_level": 0 if off else rng.choice((1, 2, 3)),
            "executed_tone_order": order_name,
            "tone_execution_indices": orders[order_name],
        })
    return {"order": order_name, "steps": steps}


def make_plan(source_commit: str, ratification_sha256: str) -> dict[str, Any]:
    orders = frozen_orders()
    names = tuple(orders)
    sessions = []
    session_index = 0
    for route in ROUTES:
        for seed in SEEDS:
            session_id = f"{route}_seed{seed}"
            sequence = [names[(session_index + offset) % 4] for offset in range(4)]
            tone_blocks = []
            for order_name in sequence:
                rows = [
                    symbol(session_id, order_name, orders, family, trial)
                    for trial in range(12)
                    for family in FAMILIES
                ]
                tone_blocks.append({"order": order_name, "symbols": rows})
            sessions.append({
                "session_id": session_id,
                "session_index": session_index,
                "route": route,
                "seed": seed,
                "partition": partition(seed),
                "order_sequence": sequence,
                "blocks": {
                    "gauge_preamble": gauge_symbols(orders),
                    "tone_order": tone_blocks,
                    "persistence": persistence_events(session_id, sequence, orders),
                    "trajectories": [trajectory(session_id, name, orders) for name in sequence],
                },
            })
            session_index += 1
    return {
        "schema_id": SCHEMA,
        "schema_version": "1.0.0",
        "source_commit": source_commit,
        "owner_decision": "RATIFY_AND_AUTHORIZE_COMBINED_TONE_ORDER_OBSERVABILITY_CAMPAIGN",
        "ratification_sha256": ratification_sha256,
        "orders": orders,
        "partitions": {"train": [0, 1, 2], "validation": [3], "stress": [4], "test": [5]},
        "state_partition": {"response": "r_t", "executed_control": "u_t", "context": "c_t", "session_gauge": "g_s"},
        "gauge_rule": "preamble_only_frozen_before_evaluated_rows",
        "persistence_outcomes": ["PERSISTENT_STATE_CANDIDATE", "DRIVEN_RELATIONAL_TRANSPORT_ONLY"],
        "operator_outcomes": ["S0_SUFFICIENT", "S1_SUFFICIENT", "S2_HISTORY_REQUIRED", "NO_STABLE_PREDICTIVE_OPERATOR"],
        "physical_acquisition_authorized_after_preflight": True,
        "restoration_authorized": False,
        "sessions": sessions,
    }


def validate(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    orders = plan.get("orders", {})
    expected = set(range(12))
    if set(orders) != {"FWD", "REV", "RND1", "RND2"}:
        errors.append("order names invalid")
    for name, values in orders.items():
        if len(values) != 12 or set(values) != expected:
            errors.append(f"invalid order {name}")
    sessions = plan.get("sessions", [])
    if len(sessions) != 12:
        errors.append("expected 12 sessions")
    counts = {(name, pos): 0 for name in orders for pos in range(4)}
    for session in sessions:
        sequence = session.get("order_sequence", [])
        if set(sequence) != set(orders):
            errors.append(f"bad order sequence {session.get('session_id')}")
            continue
        for pos, name in enumerate(sequence):
            counts[(name, pos)] += 1
        if session.get("seed") == 4 and session.get("partition") != "stress":
            errors.append("seed4 not retained as stress")
        gauge = session.get("blocks", {}).get("gauge_preamble", [])
        if len(gauge) != 8:
            errors.append("bad gauge count")
        for block in session.get("blocks", {}).get("tone_order", []):
            rows = block.get("symbols", [])
            if len(rows) != 72:
                errors.append("bad tone block count")
            for row in rows:
                if row["family"] == "order_sham" and row["declared_tone_order"] == row["executed_tone_order"]:
                    errors.append("order sham equals execution")
                if row["family"] == "pseudo" and row["codeword_bin_permutation"] == list(range(12)):
                    errors.append("pseudo identity permutation")
                if row["family"] == "silent" and row["drive_on"]:
                    errors.append("silent drive enabled")
        for event in session.get("blocks", {}).get("persistence", []):
            if not event.get("sender_off_required") or event.get("periodic_refresh_allowed"):
                errors.append("sender-off contract violated")
    if any(value != 3 for value in counts.values()):
        errors.append("order chronology not Latin-balanced")
    if plan.get("restoration_authorized") is not False:
        errors.append("restoration improperly authorized")
    return errors

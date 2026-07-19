#!/usr/bin/env python3
"""AST/control-flow/def-use proof for the pre-K3 P0 signal-path witness."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "p0_scientific_analyzer.py"
OUTPUT = ROOT / "P0_SIGNAL_PATH_ORDERING_PROOF.json"


class ProofFailure(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ProofFailure(f"FUNCTION_IDENTITY:{name}")
    return matches[0]


def named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [item for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == name]


def expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, annotate_fields=True, include_attributes=False) == ast.dump(right, annotate_fields=True, include_attributes=False)


def name_assignment(node: ast.AST, name: str) -> ast.Assign:
    matches = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == name
    ]
    if len(matches) != 1:
        raise ProofFailure(f"ASSIGNMENT_CARDINALITY:{name}")
    return matches[0]


def tuple_assignment(node: ast.AST, names: tuple[str, ...]) -> ast.Assign:
    matches: list[ast.Assign] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Assign) or len(item.targets) != 1 or not isinstance(item.targets[0], ast.Tuple):
            continue
        identifiers = tuple(element.id for element in item.targets[0].elts if isinstance(element, ast.Name))
        if identifiers == names and len(identifiers) == len(item.targets[0].elts):
            matches.append(item)
    if len(matches) != 1:
        raise ProofFailure(f"TUPLE_ASSIGNMENT_CARDINALITY:{','.join(names)}")
    return matches[0]


def call_assignment(node: ast.AST, target: str, call_name: str) -> tuple[ast.Assign, ast.Call]:
    matches = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and len(item.targets) == 1
        and isinstance(item.targets[0], ast.Name)
        and item.targets[0].id == target
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == call_name
    ]
    if len(matches) != 1:
        raise ProofFailure(f"CALL_ASSIGNMENT:{target}:{call_name}")
    assignment = matches[0]
    return assignment, assignment.value


def statement_list(root: ast.AST, statement: ast.stmt) -> tuple[ast.AST, str, list[ast.stmt]]:
    matches: list[tuple[ast.AST, str, list[ast.stmt]]] = []
    for parent in ast.walk(root):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list) and statement in value:
                matches.append((parent, field, value))
    if len(matches) != 1:
        raise ProofFailure("STATEMENT_CONTAINER")
    return matches[0]


def require_call_args(call: ast.Call, args: tuple[str, ...], label: str) -> None:
    if call.keywords or len(call.args) != len(args):
        raise ProofFailure(f"CALL_ARGUMENTS:{label}")
    for actual, expected in zip(call.args, args, strict=True):
        if not same_ast(actual, expression(expected)):
            raise ProofFailure(f"CALL_ARGUMENTS:{label}:{expected}")


def require_assignment(node: ast.AST, name: str, expected: str) -> None:
    assignment = name_assignment(node, name)
    if not same_ast(assignment.value, expression(expected)):
        raise ProofFailure(f"DEF_USE:{name}")


def reject_code(node: ast.If) -> str | None:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
        return None
    value = node.body[0].exc
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "Reject" or not value.args:
        return None
    first = value.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def require_gate(function_node: ast.FunctionDef, condition: str, code: str) -> ast.If:
    wanted = expression(condition)
    matches = [
        item
        for item in function_node.body
        if isinstance(item, ast.If) and same_ast(item.test, wanted) and reject_code(item) == code
    ]
    if len(matches) != 1:
        raise ProofFailure(f"TRANSFER_GATE:{code}:{condition}")
    return matches[0]


def require_result_binding(analyze: ast.FunctionDef, statement_body: list[ast.stmt], after_index: int) -> None:
    matches: list[ast.Expr] = []
    for statement in statement_body[after_index + 1 :]:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "update" or len(call.args) != 1 or not isinstance(call.args[0], ast.Dict):
            continue
        pairs = {
            key.value: value
            for key, value in zip(call.args[0].keys, call.args[0].values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "path_metrics" in pairs and isinstance(pairs["path_metrics"], ast.Name) and pairs["path_metrics"].id == "path_metrics":
            matches.append(statement)
    if len(matches) != 1:
        raise ProofFailure("PATH_RESULT_NOT_CARRIED")
    if not any(item is matches[0] for item in ast.walk(analyze)):
        raise ProofFailure("PATH_RESULT_SCOPE")


def require_return_contract(transfer: ast.FunctionDef) -> None:
    returns = [item for item in transfer.body if isinstance(item, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise ProofFailure("TRANSFER_RETURN")
    pairs = {
        key.value: value
        for key, value in zip(returns[0].value.keys, returns[0].value.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    claim = pairs.get("actual_path_claim")
    if not isinstance(claim, ast.Constant) or claim.value != "ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT":
        raise ProofFailure("TRANSFER_CLAIM")
    exact_names = {
        "common_mode_peak_v": "common_mode_peak",
        "pre_open_complex_separation": "separation",
        "pre_pilot_snr": "pre_snr",
        "r_drop": "r_drop",
        "source_pilot_snr": "source_snr",
    }
    for key, name in exact_names.items():
        if key not in pairs or not isinstance(pairs[key], ast.Name) or pairs[key].id != name:
            raise ProofFailure(f"TRANSFER_RESULT_DEF_USE:{key}")


def prove_bytes(data: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProofFailure("PARSE") from exc
    assembly_custody_fragments = {
        "role_mapping": b'expected_assembly = ASSEMBLY_FOR_ROLE[meta["role"]]',
        "role_gate": b'if assembly["assembly_id"] != expected_assembly or assembly["carrier_population"] != expected_population:',
        "manifest_binding": b'or assembly["assembly_manifest_sha256"] != receipts["assembly_manifest"]["sha256"]',
        "event_binding": b'or topology_receipt["qualified_native_file_sha256"] != payload["sha256"]',
        "canonical_utc_parser": b'def canonical_utc(value: Any, rejection: str) -> datetime:',
        "canonical_utc_grammar_gate": b'if not isinstance(value, str) or CANONICAL_UTC_RE.fullmatch(value) is None:',
        "canonical_utc_parse_call": b'parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")',
        "canonical_utc_roundtrip_gate": b'if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:',
        "acquisition_utc_parse": b'acquisition_started = canonical_utc(chronology_receipt["acquisition_started_utc"], "CHRONOLOGY_CUSTODY")',
        "scan_utc_parse": b'scan_started = canonical_utc(topology_receipt["scan_started_utc"], "SIGNAL_PATH_SCAN_CHRONOLOGY")',
        "chronology_gate": b'if not scan_started < scan_completed <= acquisition_started < acquisition_completed:',
        "invariant_comparison": b'if canonical_bytes({key: meta["signal_path"][key] for key in SIGNAL_PATH_INVARIANT_FIELDS}) != common["signal_path"]:',
        "scan_replay_gate": b'if topology_scan_hash in topology_scan_hashes:',
        "nonlinear_control_replay_gate": b'if nonlinear_control_hash in nonlinear_control_hashes:',
    }
    for label, fragment in assembly_custody_fragments.items():
        if data.count(fragment) != 1:
            raise ProofFailure(f"ASSEMBLY_CUSTODY:{label}")
    analyze = function(tree, "analyze_bundle")
    series = function(tree, "decode_series_witness")
    transfer = function(tree, "signal_path_transfer")
    guard = function(tree, "decode_guard_witness")

    series_calls = named_calls(analyze, "decode_series_witness")
    project_calls = named_calls(analyze, "project")
    if len(series_calls) != 1 or len(project_calls) != 1:
        raise ProofFailure("CALL_CARDINALITY")
    drive_statement, drive_call = call_assignment(analyze, "drive", "drive_fit")
    path_statement, path_call = call_assignment(analyze, "path_metrics", "signal_path_transfer")
    guard_statement = tuple_assignment(analyze, ("n_contact", "n_iso", "n_admit"))
    if not isinstance(guard_statement.value, ast.Call) or not isinstance(guard_statement.value.func, ast.Name) or guard_statement.value.func.id != "decode_guard_witness":
        raise ProofFailure("CALL_ASSIGNMENT:guard")

    require_call_args(path_call, ("ch0", "ch1", "n_gate", "records[role]['series_start']", "records[role]['n_series_open']", "meta['_signal_model']"), "signal_path_transfer")
    require_call_args(drive_call, ("ch0", "n_gate", "decimal(meta['source']['phase_command_rad'], 'phase')", "common['f_ref']", "nonnegative_decimal(meta['source']['phase_skew_standard_uncertainty_rad'], 'source.phase_skew_standard_uncertainty_rad')", "nonnegative_decimal(meta['source']['phase_drive_cal_standard_uncertainty_rad'], 'source.phase_drive_cal_standard_uncertainty_rad')"), "drive_fit")
    require_call_args(guard_statement.value, ("records[role]['decoded']", "n_gate", "records[role]['n_series_open']", "meta['witness']"), "decode_guard_witness")

    drive_parent, drive_field, drive_body = statement_list(analyze, drive_statement)
    path_parent, path_field, path_body = statement_list(analyze, path_statement)
    guard_parent, guard_field, guard_body = statement_list(analyze, guard_statement)
    if drive_parent is not path_parent or path_parent is not guard_parent or drive_field != "body" or path_field != "body" or guard_field != "body" or drive_body is not path_body or path_body is not guard_body:
        raise ProofFailure("PRE_K3_CONTROL_FLOW")
    drive_index = path_body.index(drive_statement)
    path_index = path_body.index(path_statement)
    guard_index = path_body.index(guard_statement)
    if not drive_index < path_index < guard_index:
        raise ProofFailure("PRE_K3_DOMINANCE")
    require_result_binding(analyze, path_body, guard_index)

    ordered_lines = {
        "decode_series_witness": series_calls[0].lineno,
        "drive_fit": drive_call.lineno,
        "signal_path_transfer": path_call.lineno,
        "decode_guard_witness": guard_statement.value.lineno,
        "project": project_calls[0].lineno,
    }
    if list(ordered_lines.values()) != sorted(ordered_lines.values()):
        raise ProofFailure("PRE_K3_ORDER")
    if named_calls(series, "decode_guard_witness") or named_calls(transfer, "decode_guard_witness"):
        raise ProofFailure("HIDDEN_GUARD_ACCEPTANCE")

    assignments = {
        "pre0": "signal_path_tone_fit(ch0, pre_start, pre_stop, thresholds)",
        "pre1": "signal_path_tone_fit(ch1, pre_start, pre_stop, thresholds)",
        "open0": "signal_path_tone_fit(ch0, open_start, open_stop, thresholds)",
        "open1": "signal_path_tone_fit(ch1, open_start, open_stop, thresholds)",
        "h_pre": "pre1['c2'] / pre0['c2']",
        "h_open": "open1['c2'] / open0['c2']",
        "pre_snr": "abs(pre1['c2']) / math.sqrt(max(pre1['c2_variance'], 1e-30))",
        "separation": "abs(h_pre - h_open)",
        "r_drop": "abs(open1['c2']) / abs(pre1['c2'])",
        "phase_pre": "math.atan2(h_pre.imag, h_pre.real)",
        "phase_open": "math.atan2(h_open.imag, h_open.real)",
        "common_mode_peak": "0.5 * raw_peak",
    }
    for name, expected in assignments.items():
        require_assignment(transfer, name, expected)

    gates = (
        ("SIGNAL_PATH_WINDOW_ORDER", "pre_stop - pre_start != int(pre_spec['samples']) or open_stop - open_start != int(open_spec['samples']) or pre_stop > n_gate + 250 or n_gate + 250 > series_start or open_stop > n_series_open + 1"),
        ("SIGNAL_PATH_WINDOW_CYCLES", "(pre_stop - pre_start) * F_WITNESS / FS < model_number(pre_spec['valid_cycles'], 'model.pre_valid_cycles') - 1e-12"),
        ("SIGNAL_PATH_WINDOW_CYCLES", "(open_stop - open_start) * F_WITNESS / FS < model_number(open_spec['valid_cycles'], 'model.open_valid_cycles') - 1e-12"),
        ("SIGNAL_PATH_CLIPPING", "raw_peak > clip"),
        ("SIGNAL_PATH_COMMON_MODE", "common_mode_peak > model_number(thresholds['common_mode_abs_v_max'], 'model.common_mode_abs_v_max')"),
        ("SIGNAL_PATH_C2_MISSING", "abs(pre0['c2']) <= 0.0 or abs(open0['c2']) <= 0.0"),
        ("SIGNAL_PATH_C2_MISSING", "source_snr < model_number(thresholds['minimum_pre_pilot_snr'], 'model.minimum_pre_pilot_snr')"),
        ("SIGNAL_PATH_PRE_SNR", "pre_snr < model_number(thresholds['minimum_pre_pilot_snr'], 'model.minimum_pre_pilot_snr')"),
        ("SIGNAL_PATH_PRE_TRANSFER", "abs(h_pre) < model_number(thresholds['minimum_pre_abs_h2'], 'model.minimum_pre_abs_h2')"),
        ("SIGNAL_PATH_PRE_TRANSFER", "abs(h_pre) > model_number(thresholds['maximum_pre_abs_h2'], 'model.maximum_pre_abs_h2')"),
        ("SIGNAL_PATH_PHASE", "not model_number(phase_limits['minimum'], 'model.pre_phase.minimum') <= phase_pre <= model_number(phase_limits['maximum'], 'model.pre_phase.maximum')"),
        ("SIGNAL_PATH_NOT_ISOLATED", "abs(h_open) > model_number(thresholds['isolated_abs_h2_max'], 'model.isolated_abs_h2_max')"),
        ("SIGNAL_PATH_PHASE", "not model_number(isolated_phase_limits['minimum'], 'model.isolated_phase.minimum') <= phase_open <= model_number(isolated_phase_limits['maximum'], 'model.isolated_phase.maximum')"),
        ("SIGNAL_PATH_UNCERTAINTY", "open_u95 > model_number(thresholds['isolated_u95_h2_max'], 'model.isolated_u95_h2_max')"),
        ("SIGNAL_PATH_SEPARATION", "separation < model_number(thresholds['minimum_pre_open_complex_separation'], 'model.minimum_pre_open_complex_separation')"),
        ("SIGNAL_PATH_R_DROP", "r_drop > model_number(thresholds['r_drop_max'], 'model.r_drop_max')"),
    )
    gate_nodes = [require_gate(transfer, condition, code) for code, condition in gates]
    if [transfer.body.index(node) for node in gate_nodes] != sorted(transfer.body.index(node) for node in gate_nodes):
        raise ProofFailure("TRANSFER_GATE_ORDER")
    require_return_contract(transfer)

    function_hashes = {
        item.name: sha256(ast.dump(item, annotate_fields=True, include_attributes=False).encode("utf-8"))
        for item in (series, transfer, guard, analyze)
    }
    return {
        "analyzer_sha256": sha256(data),
        "claim_ceiling": "NON_EXECUTING_P0_SIGNAL_PATH_WITNESS_REPAIR_ONLY",
        "assembly_custody_fragment_count": len(assembly_custody_fragments),
        "def_use_assignment_count": len(assignments),
        "function_ast_sha256": function_hashes,
        "ordered_call_lines": ordered_lines,
        "physical_claim_authorized": False,
        "properties": {
            "actual_path_gate_before_k3_guard_acceptance": True,
            "assembly_role_event_and_chronology_custody_bound": True,
            "canonical_utc_chronology_enforced": True,
            "all_transfer_gates_are_direct_and_dominating": True,
            "exact_call_arguments_bound": True,
            "guard_cannot_define_h2_window": True,
            "h2_window_ends_inside_code0_run": True,
            "no_n_contact_or_n_admit_dependency": True,
            "path_result_carried_to_final_result": True,
            "raw_channel_to_metric_def_use_bound": True,
            "topology_scan_replay_rejected": True,
            "nonlinear_control_replay_rejected": True,
            "source_continuity_precedes_path_gate": True,
        },
        "result": "PASS",
        "schema": "p0.signal-path-ordering-proof.v4",
        "transfer_gate_count": len(gates),
    }


def build() -> bytes:
    return canonical(prove_bytes(ANALYZER.read_bytes()))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["build"], ["verify"]):
        print("usage: p0_signal_path_ordering_proof.py {build|verify}", file=sys.stderr)
        return 2
    data = build()
    if args == ["build"]:
        temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(OUTPUT)
    elif not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
        print("FAIL: P0_SIGNAL_PATH_ORDERING_PROOF.json is stale", file=sys.stderr)
        return 1
    print(json.dumps({"bytes": len(data), "result": "PASS", "sha256": sha256(data)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

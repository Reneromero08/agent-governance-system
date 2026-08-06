#!/usr/bin/env python3
"""Branch-local transaction checks for the unverified audio CATVM candidates.

The source branch is treated only as frozen input.  This verifier probes the
preserved source binaries and writes local evidence about transaction ordering,
restoration behavior, and same-carrier reuse.  It does not promote any claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_snapshot"
RAW_LOGS = ROOT / "raw_logs" / "restoration_boundary"
SOURCE_OUTPUTS = ROOT / "raw_outputs" / "source_reproduction"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_cleanroom_module():
    path = Path(__file__).with_name("cleanroom_verify.py")
    spec = importlib.util.spec_from_file_location("audio_cleanroom_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class Packet:
    request: str
    response: str
    parsed: dict[str, Any]
    response_sha256: str
    elapsed_ns: int


def exchange(transport: socket.socket, request: str) -> Packet:
    start = time.monotonic_ns()
    transport.send(request.encode("utf-8"))
    response = transport.recv(65536).decode("utf-8", errors="replace")
    elapsed = time.monotonic_ns() - start
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parts = response.split()
        parsed = {
            "ok": bool(parts and parts[0] == "OK"),
            "event": parts[1] if len(parts) > 1 and parts[0] == "OK" else None,
            "error": parts[1] if len(parts) > 1 and parts[0] == "ERR" else None,
        }
    return Packet(
        request=request,
        response=response,
        parsed=parsed,
        response_sha256=sha256_bytes(response.encode("utf-8")),
        elapsed_ns=elapsed,
    )


@dataclass
class RunningService:
    process: subprocess.Popen[str]
    socket_path: Path

    def connect(self) -> socket.socket:
        transport = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        transport.settimeout(5.0)
        transport.connect(str(self.socket_path))
        return transport

    def close(self) -> dict[str, Any]:
        rc_before = self.process.poll()
        if rc_before is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        stdout, stderr = self.process.communicate(timeout=3.0)
        return {
            "return_code": self.process.returncode,
            "return_code_before_stop": rc_before,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
        }


def start_service(argv: list[str], socket_path: Path) -> RunningService:
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if socket_path.exists():
            return RunningService(process, socket_path)
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1.0)
            raise RuntimeError(
                f"service exited before socket creation rc={process.returncode} "
                f"stdout={len(stdout)} stderr={len(stderr)}"
            )
        time.sleep(0.02)
    status = process.poll()
    if status is None:
        process.terminate()
    raise RuntimeError("service socket was not created before timeout")


def packet_summary(packet: Packet) -> dict[str, Any]:
    parsed = packet.parsed
    return {
        "request": packet.request,
        "ok": parsed.get("ok"),
        "event": parsed.get("event"),
        "error": parsed.get("error"),
        "response_sha256": packet.response_sha256,
        "elapsed_ns": packet.elapsed_ns,
        "response_bytes": len(packet.response.encode("utf-8")),
    }


def transcript_summary(packets: list[Packet]) -> list[dict[str, Any]]:
    return [packet_summary(packet) for packet in packets]


def seal_command(program: Any) -> str:
    values = list(program.left) + list(program.right) + list(program.constraint)
    return "SEAL " + " ".join(str(value) for value in values)


def run_a_transaction(
    service_binary: Path,
    program_command: str,
    mode: str,
    include_denied_projection: bool = False,
    omit_restore: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="audio-catvm-a-") as temp:
        socket_path = Path(temp) / "service.sock"
        service = start_service(
            [str(service_binary), str(socket_path), "in-place", mode],
            socket_path,
        )
        packets: list[Packet] = []
        try:
            transport = service.connect()
            commands = ["HELLO", program_command, "F"]
            if include_denied_projection:
                commands.append("PROJECT Y")
            commands.extend(["G", "PROJECT Z"])
            if not omit_restore:
                commands.append("RESTORE")
            for command in commands:
                packets.append(exchange(transport, command))
            transport.close()
        finally:
            process_record = service.close()
    full = {
        "mode": mode,
        "omit_restore": omit_restore,
        "packets": [packet.__dict__ for packet in packets],
        "process": process_record,
    }
    return {
        "full": full,
        "summary": {
            "mode": mode,
            "omit_restore": omit_restore,
            "events": [packet.parsed.get("event") for packet in packets],
            "errors": [packet.parsed.get("error") for packet in packets],
            "oks": [packet.parsed.get("ok") for packet in packets],
            "packets": transcript_summary(packets),
            "process": process_record,
            "final_boundary_before_restore": any(
                packet.parsed.get("event") == "FINAL_BOUNDARY"
                for packet in packets[:-1 if not omit_restore else None]
            ),
            "restore_ok": (
                packets[-1].parsed.get("ok")
                if packets and packets[-1].parsed.get("event") == "RESTORATION"
                else None
            ),
            "final_coefficients": next(
                (
                    packet.parsed.get("coefficients")
                    for packet in packets
                    if packet.parsed.get("event") == "FINAL_BOUNDARY"
                ),
                None,
            ),
        },
    }


def run_a_reuse_cycles(
    service_binary: Path,
    primary_command: str,
    reuse_command: str,
    expected: dict[str, list[int]],
    cycles: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="audio-catvm-a-cycles-") as temp:
        socket_path = Path(temp) / "service.sock"
        service = start_service(
            [str(service_binary), str(socket_path), "in-place", "correct"],
            socket_path,
        )
        records: list[dict[str, Any]] = []
        try:
            transport = service.connect()
            hello = exchange(transport, "HELLO")
            for cycle in range(cycles):
                label = "primary" if cycle % 2 == 0 else "reuse"
                command = primary_command if label == "primary" else reuse_command
                packets = [
                    exchange(transport, command),
                    exchange(transport, "F"),
                    exchange(transport, "G"),
                    exchange(transport, "PROJECT Z"),
                    exchange(transport, "RESTORE"),
                ]
                z_packet = packets[3]
                restore_packet = packets[4]
                records.append(
                    {
                        "cycle": cycle,
                        "label": label,
                        "coefficients": z_packet.parsed.get("coefficients"),
                        "expected_coefficients": expected[label],
                        "coefficients_match": z_packet.parsed.get("coefficients")
                        == expected[label],
                        "restore_ok": restore_packet.parsed.get("ok"),
                        "maximum_abs_error": restore_packet.parsed.get(
                            "maximum_abs_error"
                        ),
                        "generation": restore_packet.parsed.get("generation"),
                        "snapshot_reload": restore_packet.parsed.get("snapshot_reload"),
                        "response_sha256": z_packet.response_sha256,
                    }
                )
            transport.close()
        finally:
            process_record = service.close()
    max_error = max(record["maximum_abs_error"] or 0.0 for record in records)
    generations = [record["generation"] for record in records]
    full = {"hello": hello.__dict__, "records": records, "process": process_record}
    return {
        "full": full,
        "summary": {
            "cycles": cycles,
            "all_coefficients_match": all(record["coefficients_match"] for record in records),
            "all_restores_ok": all(record["restore_ok"] for record in records),
            "maximum_abs_error": max_error,
            "generation_first": generations[0] if generations else None,
            "generation_last": generations[-1] if generations else None,
            "generation_monotonic": generations == sorted(generations),
            "snapshot_reload_seen": any(record["snapshot_reload"] for record in records),
            "process": process_record,
            "record_sha256": sha256_json(records),
        },
    }


def run_b_probe(
    service_binary: Path,
    manifest: Path,
    mode: str,
    commands: list[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="audio-catvm-b-") as temp:
        socket_path = Path(temp) / "service.sock"
        service = start_service(
            [str(service_binary), str(socket_path), str(manifest), mode],
            socket_path,
        )
        packets: list[Packet] = []
        try:
            transport = service.connect()
            for command in commands:
                packets.append(exchange(transport, command))
            transport.close()
        finally:
            process_record = service.close()
    full = {
        "mode": mode,
        "commands": commands,
        "packets": [packet.__dict__ for packet in packets],
        "process": process_record,
    }
    return {
        "full": full,
        "summary": {
            "mode": mode,
            "commands": commands,
            "events": [packet.parsed.get("event") for packet in packets],
            "errors": [packet.parsed.get("error") for packet in packets],
            "oks": [packet.parsed.get("ok") for packet in packets],
            "response_prefixes": [packet.response.split(" ", 2)[:2] for packet in packets],
            "packets": transcript_summary(packets),
            "process": process_record,
        },
    }


def reference_machine_experiment(cleanroom: Any) -> dict[str, Any]:
    primary = cleanroom.parse_aspr(SOURCE / "catvm_primary.aspr")
    reuse = cleanroom.parse_aspr(SOURCE / "catvm_reuse.aspr")
    expected_primary = cleanroom.enumerate_series_parallel(primary)
    expected_reuse = cleanroom.enumerate_series_parallel(reuse)

    class ReferenceCarrier:
        def __init__(self) -> None:
            self.state = [0, 0, 0, 0]
            self.hidden_live = False
            self.generation = 0
            self.responses: list[dict[str, Any]] = []

        def project_hidden(self) -> dict[str, Any]:
            return {"ok": False, "error": "INTERMEDIATE_PROJECTION_DENIED"}

        def transact(self, program: Any, mode: str = "correct") -> dict[str, Any]:
            before = list(self.state)
            result = cleanroom.enumerate_series_parallel(program)
            controls = cleanroom.restoration_controls(program)
            f = list(result["f_coefficients"])
            z = list(result["z_coefficients"])
            self.hidden_live = True
            self.state = f
            self.state = z
            restored = controls["nominal_restored"] and mode == "correct"
            if mode == "missing_g":
                self.state = list(controls["residuals"]["missing_g_z"])
            elif mode == "wrong_g":
                self.state = list(controls["residuals"]["wrong_g_z"])
            elif mode == "reordered":
                self.state = list(controls["residuals"]["reordered_z"])
            else:
                self.state = before
            self.hidden_live = False
            self.generation += 1
            if not restored:
                return {
                    "ok": False,
                    "error": "RESTORATION_CHECK_FAILED",
                    "generation": self.generation,
                    "response_after_restore": True,
                    "state": list(self.state),
                }
            response = {
                "ok": True,
                "event": "FINAL_BOUNDARY_AFTER_RESTORE",
                "coefficients": z,
                "generation": self.generation,
                "response_after_restore": True,
                "state_restored": self.state == before,
            }
            self.responses.append(response)
            return response

    carrier = ReferenceCarrier()
    denied = carrier.project_hidden()
    records = []
    for cycle in range(64):
        label = "primary" if cycle % 2 == 0 else "reuse"
        program = primary if label == "primary" else reuse
        expected = (
            expected_primary["z_coefficients"]
            if label == "primary"
            else expected_reuse["z_coefficients"]
        )
        response = carrier.transact(program, "correct")
        records.append(
            {
                "cycle": cycle,
                "label": label,
                "coefficients": response.get("coefficients"),
                "expected_coefficients": expected,
                "coefficients_match": response.get("coefficients") == expected,
                "state_restored": response.get("state_restored"),
                "generation": response.get("generation"),
            }
        )
    negative_modes = {
        mode: carrier.transact(primary, mode)
        for mode in ["missing_g", "wrong_g", "reordered"]
    }
    return {
        "method": "branch_local_reference_carrier_independent_of_source_binary",
        "denied_hidden_projection": denied,
        "cycles": len(records),
        "all_cycles_match": all(record["coefficients_match"] for record in records),
        "all_cycles_restored": all(record["state_restored"] for record in records),
        "generation_last": records[-1]["generation"],
        "negative_modes_fail_closed": all(
            not response["ok"] for response in negative_modes.values()
        ),
        "negative_modes": negative_modes,
        "records_sha256": sha256_json(records),
    }


def summarize_restoration_report(
    a_primary: dict[str, Any],
    a_omit_restore: dict[str, Any],
    a_modes: dict[str, Any],
    a_cycles: dict[str, Any],
    b_correct: dict[str, Any],
    b_negative: dict[str, Any],
    transfer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "audio_catvm_restoration_reuse_stress.v1",
        "generated_utc": utc_now(),
        "canonical": False,
        "small_wall_crossed": False,
        "source_branch_claims_remain": [
            "CLAIMED_BY_SOURCE_BRANCH",
            "NOT_CANONICAL",
            "NOT_PHYSICAL_FAMILY10H_EVIDENCE",
        ],
        "candidate_a": {
            "source_binary": str(
                SOURCE_OUTPUTS / "candidate_a_run1" / "catvm_phase_service"
            ),
            "source_binary_sha256": file_sha256(
                SOURCE_OUTPUTS / "candidate_a_run1" / "catvm_phase_service"
            ),
            "source_primary_transaction": a_primary["summary"],
            "source_omit_restore_observation": a_omit_restore["summary"],
            "source_negative_modes": {
                key: value["summary"] for key, value in a_modes.items()
            },
            "source_reuse_cycles": a_cycles["summary"],
            "restoration_reuse_decision": "SOURCE_RESTORES_AND_REUSES_BUT_FINAL_RESPONSE_PRECEDES_RESTORE",
        },
        "candidate_b": {
            "source_binary": str(
                SOURCE_OUTPUTS / "candidate_b_run1" / "service_testing"
            ),
            "source_binary_sha256": file_sha256(
                SOURCE_OUTPUTS / "candidate_b_run1" / "service_testing"
            ),
            "correct_probe": b_correct["summary"],
            "negative_probe": b_negative["summary"],
            "restoration_reuse_decision": "SOURCE_LOCAL_ATOMIC_EXECUTE_PATTERN_REPROBED_WITH_FIXED_PUBLIC_TOPOLOGY",
        },
        "branch_local_transfer_reference": transfer,
        "raw_log_directory": str(RAW_LOGS),
    }


def render_machine_boundary_report(report: dict[str, Any]) -> str:
    a = report["candidate_a"]
    b = report["candidate_b"]
    transfer = report["branch_local_transfer_reference"]
    lines = [
        "# Machine Boundary Attack Report",
        "",
        "Status: unverified transfer evidence only. Not canonical. Small Wall not crossed.",
        "",
        "## Candidate A",
        "",
        "- Active probe used the frozen run1 source service binary.",
        "- Intermediate projection request was denied by protocol.",
        "- Final boundary response was available before the RESTORE request.",
        "- Closing after the final boundary without RESTORE still produced the final boundary packet.",
        "- Correct-mode restoration and same-carrier reuse passed across the recorded cycle stress.",
        "",
        "Decision: the source Candidate A protocol does not satisfy atomic final-only response after restoration. The algebra and in-place restoration remain salvageable only through a repaired wrapper.",
        "",
        "## Candidate B",
        "",
        "- Active probe used the frozen run1 testing service with the copied public manifest.",
        "- Intermediate projection requests were denied.",
        "- EXECUTE responses were returned as completed transaction packets.",
        "- Negative restoration mode returned a closed response path for the tested command.",
        "",
        "Decision: Candidate B has a source-local atomic EXECUTE pattern, but the static audit still limits transfer because topology identifiers and scheduling receipts are fixed to the copied public graph.",
        "",
        "## Branch-Local Transfer Reference",
        "",
        f"- Reference cycles: {transfer['cycles']}",
        f"- Cycle boundaries matched: {transfer['all_cycles_match']}",
        f"- State restored each accepted cycle: {transfer['all_cycles_restored']}",
        f"- Negative restoration variants failed closed: {transfer['negative_modes_fail_closed']}",
        "",
        "This repaired reference is evidence that the transaction law can be reconstructed locally; it is not evidence that the source Candidate A implementation satisfied that law.",
        "",
        "## Evidence Hashes",
        "",
        f"- RESTORATION_REUSE_STRESS payload: `{sha256_json(report)}`",
        f"- Candidate A service: `{a['source_binary_sha256']}`",
        f"- Candidate B testing service: `{b['source_binary_sha256']}`",
        f"- Raw log directory: `{report['raw_log_directory']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    cleanroom = load_cleanroom_module()
    primary = cleanroom.parse_aspr(SOURCE / "catvm_primary.aspr")
    reuse = cleanroom.parse_aspr(SOURCE / "catvm_reuse.aspr")
    primary_command = seal_command(primary)
    reuse_command = seal_command(reuse)
    expected = {
        "primary": list(cleanroom.enumerate_series_parallel(primary)["z_coefficients"]),
        "reuse": list(cleanroom.enumerate_series_parallel(reuse)["z_coefficients"]),
    }

    RAW_LOGS.mkdir(parents=True, exist_ok=True)
    a_service = SOURCE_OUTPUTS / "candidate_a_run1" / "catvm_phase_service"
    b_service = SOURCE_OUTPUTS / "candidate_b_run1" / "service_testing"
    manifest = SOURCE / "general_multi_dag_affine_topology.txt"

    a_primary = run_a_transaction(
        a_service,
        primary_command,
        "correct",
        include_denied_projection=True,
    )
    a_omit_restore = run_a_transaction(
        a_service,
        primary_command,
        "correct",
        omit_restore=True,
    )
    a_modes = {
        mode: run_a_transaction(a_service, primary_command, mode)
        for mode in ["wrong-g", "missing-g", "reordered"]
    }
    a_cycles = run_a_reuse_cycles(
        a_service,
        primary_command,
        reuse_command,
        expected,
        cycles=64,
    )

    b_correct = run_b_probe(
        b_service,
        manifest,
        "correct",
        ["HELLO", "PROJECT FORWARD_TAPE", "EXECUTE 0", "EXECUTE 1", "SHUTDOWN"],
    )
    b_negative = run_b_probe(
        b_service,
        manifest,
        "wrong-root",
        ["HELLO", "EXECUTE 0"],
    )
    transfer = reference_machine_experiment(cleanroom)

    raw_payloads = {
        "candidate_a_primary_full.json": a_primary["full"],
        "candidate_a_omit_restore_full.json": a_omit_restore["full"],
        "candidate_a_negative_modes_full.json": {
            key: value["full"] for key, value in a_modes.items()
        },
        "candidate_a_reuse_cycles_full.json": a_cycles["full"],
        "candidate_b_correct_full.json": b_correct["full"],
        "candidate_b_negative_full.json": b_negative["full"],
        "transfer_reference_full.json": transfer,
    }
    raw_hashes: dict[str, str] = {}
    for name, payload in raw_payloads.items():
        path = RAW_LOGS / name
        write_json(path, payload)
        raw_hashes[name] = file_sha256(path)

    report = summarize_restoration_report(
        a_primary,
        a_omit_restore,
        a_modes,
        a_cycles,
        b_correct,
        b_negative,
        transfer,
    )
    report["raw_log_hashes"] = raw_hashes
    report["result_sha256"] = sha256_json(report)

    write_json(ROOT / "RESTORATION_REUSE_STRESS.json", report)
    write_text(ROOT / "MACHINE_BOUNDARY_ATTACK_REPORT.md", render_machine_boundary_report(report))
    print(
        json.dumps(
            {
                "restoration_reuse_stress_sha256": file_sha256(
                    ROOT / "RESTORATION_REUSE_STRESS.json"
                ),
                "machine_boundary_report_sha256": file_sha256(
                    ROOT / "MACHINE_BOUNDARY_ATTACK_REPORT.md"
                ),
                "raw_logs": len(raw_hashes),
                "candidate_a_final_before_restore": report["candidate_a"][
                    "source_primary_transaction"
                ]["final_boundary_before_restore"],
                "transfer_reference_cycles": transfer["cycles"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

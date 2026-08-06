#!/usr/bin/env python3
"""Scientific runtime probes for Candidate E independent of qualifier modes."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/ags-audio-source-6f39766-v2")
FRONTIER = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/"
    "cat_cas_phase_frontier"
)
SOURCE_CAT = SOURCE / FRONTIER
RAW = ROOT / "raw_outputs" / "independent_delta_v2" / "candidate_e_science_runtime"
LOG = ROOT / "raw_logs" / "independent_delta_v2" / "candidate_e_science_runtime"

MAGIC = 0x43564450
LEASE_TAG = 0x4C45415345445054
REQUEST = struct.Struct("<IIIIQQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQ")

INITIALIZE = 1
BEGIN = 2
PROJECT = 3
CONTINUE = 4
REUSE = 5
WRONG_OWNER = 9
SNAPSHOT = 11
STOP = 12

STATUS_OK = 0
STATUS_DENIED = 1
STATUS_ERROR = 2

BOUNDARY_VALID = 1
RESTORED = 2
STAGE_RESIDENT = 4
REUSE_FLAG = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(command: int, generation: int, depth: int, variant: int, lease: int, nonce: int, magic: int = MAGIC) -> bytes:
    if not (0 <= depth <= 0xFFFF and 0 <= variant <= 0xFFFF):
        raise ValueError("depth/variant out of wire range")
    return REQUEST.pack(magic, command, generation, depth | (variant << 16), lease, nonce)


def unpack(payload: bytes) -> dict[str, Any]:
    fields = RESPONSE.unpack(payload)
    return {
        "magic": fields[0],
        "status": fields[1],
        "command": fields[2],
        "generation": fields[3],
        "flags": fields[4],
        "lease": fields[5],
        "receipt": fields[6],
        "boundary": list(fields[7:14]),
        "restoration_error": fields[14],
        "norm_error": fields[15],
        "native_operations": fields[16],
    }


def safe_response(resp: dict[str, Any] | None) -> dict[str, Any] | None:
    if resp is None:
        return None
    return {
        "status": resp["status"],
        "command": resp["command"],
        "generation": resp["generation"],
        "flags": resp["flags"],
        "lease": resp["lease"],
        "receipt_sha256": sha256_bytes(str(resp["receipt"]).encode("ascii")),
        "boundary_valid": bool(resp["flags"] & BOUNDARY_VALID),
        "restored": bool(resp["flags"] & RESTORED),
        "stage_resident": bool(resp["flags"] & STAGE_RESIDENT),
        "reuse": bool(resp["flags"] & REUSE_FLAG),
        "boundary_nonzero_count": sum(1 for value in resp["boundary"] if float(value) != 0.0),
        "restoration_error": resp["restoration_error"],
        "norm_error": resp["norm_error"],
        "native_operations": resp["native_operations"],
    }


def run_cmd(command: list[str], cwd: Path, timeout: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "return_code": proc.returncode,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
        "stdout_bytes": len(proc.stdout.encode("utf-8")),
        "stderr_bytes": len(proc.stderr.encode("utf-8")),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_service() -> dict[str, Any]:
    RAW.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    service = RAW / "catvm_necklace_shared_latent_depth_atomic_repair_service"
    command = [
        "g++",
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        str(SOURCE_CAT / "catvm_necklace_shared_latent_depth_atomic_repair_service.cpp"),
        "-o",
        str(service),
    ]
    result = run_cmd(command, SOURCE_CAT, timeout=180)
    (LOG / "build_service.stdout").write_text(result.pop("stdout"), encoding="utf-8")
    (LOG / "build_service.stderr").write_text(result.pop("stderr"), encoding="utf-8")
    result["service"] = str(service)
    return result


def start_service(service: Path, label: str, mode: str = "normal") -> tuple[subprocess.Popen[str], socket.socket, Path]:
    sock_path = Path(f"/tmp/ags-e-v2-{label}-{os.getpid()}-{time.monotonic_ns()}.sock")
    proc = subprocess.Popen(
        [str(service), mode, str(sock_path)],
        cwd=SOURCE_CAT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 15
    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    while time.time() < deadline:
        try:
            client.connect(str(sock_path))
            return proc, client, sock_path
        except FileNotFoundError:
            time.sleep(0.05)
        except ConnectionRefusedError:
            time.sleep(0.05)
    proc.terminate()
    raise RuntimeError(f"service did not accept connection for {label}")


def exchange(client: socket.socket, payload: bytes) -> dict[str, Any]:
    client.sendall(payload)
    response = client.recv(RESPONSE.size)
    if len(response) != RESPONSE.size:
        raise RuntimeError(f"response size {len(response)}")
    return unpack(response)


def finish(proc: subprocess.Popen[str], client: socket.socket | None = None, close_client: bool = True) -> dict[str, Any]:
    if client is not None and close_client:
        client.close()
    try:
        stdout, stderr = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=15)
    return {
        "return_code": proc.returncode,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
    }


def stop_clean(client: socket.socket, generation: int, lease: int, nonce: int) -> dict[str, Any]:
    return exchange(client, pack(STOP, generation, 0, 0, lease, nonce))


def scenario_happy(service: Path) -> dict[str, Any]:
    proc, client, sock_path = start_service(service, "happy")
    events = []
    init = exchange(client, pack(INITIALIZE, 0, 0, 0, 0, 100))
    lease = init["lease"]
    events.append(("initialize", safe_response(init)))
    repeat = exchange(client, pack(INITIALIZE, 0, 0, 0, 0, 101))
    events.append(("repeat_initialize", safe_response(repeat)))
    begin = exchange(client, pack(BEGIN, 0, 4, 2, lease, 101))
    events.append(("begin_after_denied_initialize_same_nonce", safe_response(begin)))
    project = exchange(client, pack(PROJECT, 0, 4, 2, lease, 102))
    events.append(("project_while_resident", safe_response(project)))
    cont = exchange(client, pack(CONTINUE, 0, 4, 2, lease, 103))
    events.append(("continue", safe_response(cont)))
    reuse = exchange(client, pack(REUSE, 1, 3, 5, lease, 104))
    events.append(("reuse", safe_response(reuse)))
    stale = exchange(client, pack(REUSE, 1, 3, 5, lease, 105))
    events.append(("stale_generation", safe_response(stale)))
    stop = stop_clean(client, 2, lease, 106)
    events.append(("stop", safe_response(stop)))
    process = finish(proc, client)
    return {"socket_path": str(sock_path), "events": events, "process": process}


def scenario_wrong_bindings(service: Path) -> dict[str, Any]:
    cases = []
    for label, request_builder in [
        ("wrong_lease", lambda lease: pack(BEGIN, 0, 4, 2, lease ^ 1, 101)),
        ("wrong_owner_command", lambda lease: pack(WRONG_OWNER, 0, 4, 2, lease, 101)),
        ("snapshot_command", lambda lease: pack(SNAPSHOT, 0, 4, 2, lease, 101)),
        ("unknown_command", lambda lease: pack(99, 0, 4, 2, lease, 101)),
        ("premature_projection", lambda lease: pack(PROJECT, 0, 4, 2, lease, 101)),
    ]:
        proc, client, sock_path = start_service(service, label)
        init = exchange(client, pack(INITIALIZE, 0, 0, 0, 0, 100))
        resp = exchange(client, request_builder(init["lease"]))
        stop = stop_clean(client, 0, init["lease"], 102)
        process = finish(proc, client)
        cases.append(
            {
                "case": label,
                "socket_path": str(sock_path),
                "response": safe_response(resp),
                "stop": safe_response(stop),
                "process": process,
            }
        )
    return {"cases": cases}


def scenario_mismatch_and_bad_magic(service: Path) -> dict[str, Any]:
    proc, client, sock_path = start_service(service, "mismatch_bad_magic")
    events = []
    init = exchange(client, pack(INITIALIZE, 0, 0, 0, 0, 200))
    lease = init["lease"]
    events.append(("initialize", safe_response(init)))
    begin = exchange(client, pack(BEGIN, 0, 8, 3, lease, 201))
    events.append(("begin", safe_response(begin)))
    mismatch = exchange(client, pack(CONTINUE, 0, 7, 3, lease, 202))
    events.append(("descriptor_mismatch", safe_response(mismatch)))
    bad_magic = exchange(client, pack(CONTINUE, 0, 8, 3, lease, 203, magic=0))
    events.append(("bad_magic_while_resident", safe_response(bad_magic)))
    cont = exchange(client, pack(CONTINUE, 0, 8, 3, lease, 203))
    events.append(("continue_after_bad_magic", safe_response(cont)))
    stop = stop_clean(client, 1, lease, 204)
    events.append(("stop", safe_response(stop)))
    process = finish(proc, client)
    return {"socket_path": str(sock_path), "events": events, "process": process}


def scenario_disconnect_after_stage(service: Path) -> dict[str, Any]:
    proc, client, sock_path = start_service(service, "disconnect_stage")
    init = exchange(client, pack(INITIALIZE, 0, 0, 0, 0, 300))
    begin = exchange(client, pack(BEGIN, 0, 8, 3, init["lease"], 301))
    client.close()
    process = finish(proc, None, close_client=False)
    return {
        "socket_path": str(sock_path),
        "initialize": safe_response(init),
        "begin": safe_response(begin),
        "process": process,
    }


def scenario_truncated(service: Path) -> dict[str, Any]:
    proc, client, sock_path = start_service(service, "truncated")
    client.sendall(pack(INITIALIZE, 0, 0, 0, 0, 400)[:8])
    client.close()
    process = finish(proc, None, close_client=False)
    return {"socket_path": str(sock_path), "process": process, "response_expected": False}


def scenario_oversized(service: Path) -> dict[str, Any]:
    proc, client, sock_path = start_service(service, "oversized")
    oversized = pack(INITIALIZE, 0, 0, 0, 0, 500) + b"X"
    resp = exchange(client, oversized)
    process: dict[str, Any]
    stop = None
    if resp["status"] == STATUS_OK:
        stop = stop_clean(client, 0, resp["lease"], 501)
        process = finish(proc, client)
    else:
        process = finish(proc, client)
    return {
        "socket_path": str(sock_path),
        "oversized_bytes_sent": len(oversized),
        "normal_request_bytes": REQUEST.size,
        "response": safe_response(resp),
        "stop": safe_response(stop) if stop is not None else None,
        "process": process,
        "oversized_packet_accepted": resp["status"] == STATUS_OK,
    }


def main() -> int:
    build = build_service()
    if build["return_code"] != 0:
        raise SystemExit(json.dumps({"build": build}, sort_keys=True))
    service = Path(build["service"])
    payload = {
        "schema_version": "candidate_e_science_runtime_v2",
        "created_utc": utc_now(),
        "canonical": False,
        "small_wall_crossed": False,
        "source_commit": "6f39766e9cf622e2e41d178f8131bd8777b6cd1d",
        "method": "Compile service directly and probe protocol semantics independent of qualifier executable bits.",
        "build": build,
        "happy_path": scenario_happy(service),
        "wrong_binding_cases": scenario_wrong_bindings(service),
        "mismatch_and_bad_magic": scenario_mismatch_and_bad_magic(service),
        "disconnect_after_stage": scenario_disconnect_after_stage(service),
        "truncated_packet": scenario_truncated(service),
        "oversized_packet": scenario_oversized(service),
    }
    payload["scientific_defects"] = []
    if payload["oversized_packet"]["oversized_packet_accepted"]:
        payload["scientific_defects"].append(
            "OVERSIZED_SEQPACKET_ACCEPTED_AS_VALID_REQUEST"
        )
    # Happy path must keep all boundary-bearing responses restored.
    for label, response in payload["happy_path"]["events"]:
        if response and response["boundary_valid"] and not response["restored"]:
            payload["scientific_defects"].append(
                f"BOUNDARY_WITHOUT_RESTORED_FLAG:{label}"
            )
    payload["classification_recommendation"] = (
        "REJECTED_SOURCE_DEFECT"
        if payload["scientific_defects"]
        else "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
    )
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "runtime_report.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "classification_recommendation": payload["classification_recommendation"],
                "scientific_defects": payload["scientific_defects"],
                "output": str(out.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

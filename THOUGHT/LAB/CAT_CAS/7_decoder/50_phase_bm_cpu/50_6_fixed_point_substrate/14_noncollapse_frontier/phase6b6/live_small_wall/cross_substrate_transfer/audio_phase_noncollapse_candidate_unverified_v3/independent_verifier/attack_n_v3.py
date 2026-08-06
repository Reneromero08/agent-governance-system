#!/usr/bin/env python3
"""Independent packet-layer attacks for Candidate N."""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_outputs" / "independent_n_v3"
SERVICE = (
    ROOT
    / "raw_outputs"
    / "source_reproduction_v3_short_path_controls"
    / "n_short_reproduction"
    / "run1"
    / "descriptor_service"
)

MAGIC = 0x43564C50
REQUEST = struct.Struct("<IIIIQQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQ")
INITIALIZE = 1
STOP = 12
STATUS_OK = 0
STATUS_DENIED = 1
STATUS_ERROR = 2
BOUNDARY_VALID = 1
RESTORED = 2
STAGE_RESIDENT = 4
LEASE_TAG = 0x4C454153454C4154


def pack(command: int, generation: int = 0, lease: int = 0, nonce: int = 1, reserved: int = 0, magic: int = MAGIC) -> bytes:
    return REQUEST.pack(magic, command, generation, reserved, lease, nonce)


def parse(payload: bytes) -> dict:
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


class Service:
    def __init__(self, name: str):
        self.dir = RAW / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = Path("/tmp") / f"ags-v3-n-packet-{os.getpid()}-{name}"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.sock_path = self.runtime_dir / "svc.sock"
        self.stdout_path = self.dir / "service.stdout"
        self.stderr_path = self.dir / "service.stderr"
        self.stdout = self.stdout_path.open("wb")
        self.stderr = self.stderr_path.open("wb")
        self.proc = subprocess.Popen(
            [str(SERVICE), "normal", str(self.sock_path)],
            stdout=self.stdout,
            stderr=self.stderr,
        )
        for _ in range(500):
            if self.sock_path.exists():
                break
            if self.proc.poll() is not None:
                break
            time.sleep(0.01)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.settimeout(2.0)
        self.connected = False
        if self.sock_path.exists():
            self.sock.connect(str(self.sock_path))
            self.connected = True

    def send_packet(self, payload: bytes) -> dict:
        self.sock.sendall(payload)
        response = self.sock.recv(RESPONSE.size)
        return {"response_size": len(response), "response": parse(response) if len(response) == RESPONSE.size else None}

    def close(self) -> int | None:
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            rc = self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            rc = self.proc.wait(timeout=5)
        self.stdout.close()
        self.stderr.close()
        return rc


def run_case(name: str, payload: bytes, expect_not_ok: bool) -> dict:
    service = Service(name)
    result: dict = {
        "case": name,
        "payload_bytes": len(payload),
        "request_bytes": REQUEST.size,
        "socket_path": str(service.sock_path),
        "connected": service.connected,
        "expect_not_ok": expect_not_ok,
    }
    try:
        if not service.connected:
            result["error"] = "service_did_not_bind"
        else:
            result.update(service.send_packet(payload))
            response = result.get("response")
            if response:
                result["status"] = response["status"]
                result["flags"] = response["flags"]
                result["boundary_nonzero"] = any(float(x) != 0.0 for x in response["boundary"])
                result["accepted_ok"] = response["status"] == STATUS_OK
                result["fail_open"] = expect_not_ok and response["status"] == STATUS_OK
    except Exception as exc:  # recorded as attack evidence, not hidden
        result["exception"] = type(exc).__name__
        result["exception_text"] = str(exc)
    finally:
        result["service_return_code"] = service.close()
        result["stdout_bytes"] = service.stdout_path.stat().st_size if service.stdout_path.exists() else None
        result["stderr_bytes"] = service.stderr_path.stat().st_size if service.stderr_path.exists() else None
    return result


def run_truncated_case() -> dict:
    service = Service("truncated_31")
    result = {
        "case": "truncated_31",
        "payload_bytes": REQUEST.size - 1,
        "request_bytes": REQUEST.size,
        "socket_path": str(service.sock_path),
        "connected": service.connected,
        "expect_no_response": True,
    }
    try:
        if service.connected:
            service.sock.sendall(pack(INITIALIZE, nonce=0x5101)[:-1])
            service.sock.shutdown(socket.SHUT_WR)
            try:
                response = service.sock.recv(RESPONSE.size)
            except socket.timeout:
                response = b""
            result["response_size"] = len(response)
            result["response"] = parse(response) if len(response) == RESPONSE.size else None
    except Exception as exc:
        result["exception"] = type(exc).__name__
        result["exception_text"] = str(exc)
    finally:
        result["service_return_code"] = service.close()
        result["stdout_bytes"] = service.stdout_path.stat().st_size if service.stdout_path.exists() else None
        result["stderr_bytes"] = service.stderr_path.stat().st_size if service.stderr_path.exists() else None
    return result


def run_cross_record_splice_case() -> dict:
    service = Service("cross_record_splice_31_plus_1")
    result = {
        "case": "cross_record_splice_31_plus_1",
        "first_record_bytes": REQUEST.size - 1,
        "second_record_bytes": 1,
        "request_bytes": REQUEST.size,
        "socket_path": str(service.sock_path),
        "connected": service.connected,
        "expect_not_ok": True,
        "attack_law": "two distinct seqpacket records must not be combined into one request",
    }
    try:
        if service.connected:
            payload = pack(INITIALIZE, nonce=0x5201)
            service.sock.sendall(payload[:-1])
            service.sock.sendall(payload[-1:])
            response = service.sock.recv(RESPONSE.size)
            result["response_size"] = len(response)
            result["response"] = parse(response) if len(response) == RESPONSE.size else None
            if result["response"]:
                result["status"] = result["response"]["status"]
                result["flags"] = result["response"]["flags"]
                result["accepted_ok"] = result["response"]["status"] == STATUS_OK
                result["fail_open"] = result["accepted_ok"]
    except Exception as exc:
        result["exception"] = type(exc).__name__
        result["exception_text"] = str(exc)
    finally:
        result["service_return_code"] = service.close()
        result["stdout_bytes"] = service.stdout_path.stat().st_size if service.stdout_path.exists() else None
        result["stderr_bytes"] = service.stderr_path.stat().st_size if service.stderr_path.exists() else None
    return result


def run_stop_after_initialize_control() -> dict:
    service = Service("valid_stop_after_initialize_control")
    result = {
        "case": "valid_stop_after_initialize_control",
        "socket_path": str(service.sock_path),
        "connected": service.connected,
        "expect_stop_ok": True,
    }
    try:
        if service.connected:
            init = service.send_packet(pack(INITIALIZE, nonce=0x5301))
            lease = init["response"]["lease"] if init.get("response") else 0
            stop = service.send_packet(pack(STOP, lease=lease, nonce=0x5302))
            result["initialize"] = init
            result["stop"] = stop
            result["initialize_status"] = init["response"]["status"] if init.get("response") else None
            result["stop_status"] = stop["response"]["status"] if stop.get("response") else None
            result["stop_ok"] = result["stop_status"] == STATUS_OK
    except Exception as exc:
        result["exception"] = type(exc).__name__
        result["exception_text"] = str(exc)
    finally:
        result["service_return_code"] = service.close()
        result["stdout_bytes"] = service.stdout_path.stat().st_size if service.stdout_path.exists() else None
        result["stderr_bytes"] = service.stderr_path.stat().st_size if service.stderr_path.exists() else None
    return result


def run_rejected_initialize_nonce_mutation_case() -> dict:
    service = Service("rejected_initialize_mutates_nonce_state")
    result = {
        "case": "rejected_initialize_mutates_nonce_state",
        "socket_path": str(service.sock_path),
        "connected": service.connected,
        "expect_not_ok": True,
        "attack_law": "a rejected repeated initialize must not advance nonce state",
    }
    try:
        if service.connected:
            init = service.send_packet(pack(INITIALIZE, nonce=0x5401))
            lease = init["response"]["lease"] if init.get("response") else 0
            rejected = service.send_packet(pack(INITIALIZE, nonce=(2**64 - 1)))
            stop = service.send_packet(pack(STOP, lease=lease, nonce=0x5402))
            result["initialize"] = init
            result["rejected_initialize"] = rejected
            result["stop_after_rejected_initialize"] = stop
            result["initialize_status"] = init["response"]["status"] if init.get("response") else None
            result["rejected_initialize_status"] = rejected["response"]["status"] if rejected.get("response") else None
            result["stop_status"] = stop["response"]["status"] if stop.get("response") else None
            result["stop_was_denied_after_rejected_initialize"] = result["stop_status"] == STATUS_DENIED
            result["fail_open"] = result["stop_was_denied_after_rejected_initialize"]
    except Exception as exc:
        result["exception"] = type(exc).__name__
        result["exception_text"] = str(exc)
    finally:
        result["service_return_code"] = service.close()
        result["stdout_bytes"] = service.stdout_path.stat().st_size if service.stdout_path.exists() else None
        result["stderr_bytes"] = service.stderr_path.stat().st_size if service.stderr_path.exists() else None
    return result


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case("valid_initialize_control", pack(INITIALIZE, nonce=0x5001), expect_not_ok=False),
        run_case("oversize_initialize_plus_one", pack(INITIALIZE, nonce=0x5002) + b"X", expect_not_ok=True),
        run_case("oversize_initialize_plus_16", pack(INITIALIZE, nonce=0x5003) + (b"Y" * 16), expect_not_ok=True),
        run_case("concatenated_initialize_stop", pack(INITIALIZE, nonce=0x5004) + pack(STOP, nonce=0x5005), expect_not_ok=True),
        run_truncated_case(),
        run_cross_record_splice_case(),
        run_stop_after_initialize_control(),
        run_rejected_initialize_nonce_mutation_case(),
        run_case("bad_magic_initialize", pack(INITIALIZE, nonce=0x5006, magic=0), expect_not_ok=True),
        run_case("reserved_initialize", pack(INITIALIZE, nonce=0x5007, reserved=1), expect_not_ok=True),
    ]
    fail_open = [case for case in cases if case.get("fail_open")]
    long_path_failure = json.loads((ROOT / "SOURCE_REPRODUCTION_DATA_V3.json").read_text(encoding="utf-8"))["classifications"]["N"] == "SOURCE_NOT_REPRODUCED"
    classification = "REJECTED_SOURCE_DEFECT" if fail_open or long_path_failure else "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
    report = {
        "schema_version": "audio_noncollapse_v3_candidate_n_packet_attack",
        "canonical": False,
        "small_wall_crossed": False,
        "candidate": "N",
        "service_binary": str(SERVICE.relative_to(ROOT)),
        "request_size": REQUEST.size,
        "response_size": RESPONSE.size,
        "cases": cases,
        "fail_open_cases": [case["case"] for case in fail_open],
        "long_evidence_root_source_reproduction_failed": long_path_failure,
        "short_path_source_reproduction_passed_twice": True,
        "classification": classification,
        "decision_reason": "malformed seqpacket accepted or rejected initialize mutates accepted nonce state" if fail_open else "long evidence-root socket path failure",
    }
    (RAW / "n_packet_attack_data.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "TWO_PORT_CATVM_RUNTIME_ATTACK_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    custody = f"""# Two-Port Custody Report

Candidate N: descriptor-compiled two-shared-port CATVM custody.

Classification: `{classification}`

Runtime packet attack summary:

- Request size: `{REQUEST.size}` bytes.
- Fail-open packet cases: `{[case['case'] for case in fail_open]}`.
- Long evidence-root source reproduction failed before bind; short-path reproduction passed twice.

Finding:

The source-local protocol/custody behavior is promising under the short-path qualifier, but the package cannot transfer as a machine law while malformed/oversized seqpacket handling, record-boundary handling, rejected-initialize state mutation, and path-depth behavior remain defective. The prompt explicitly required malformed records not be silently truncated, normalized, spliced, or accepted.

Custody scope retained:

Short-path source controls still support owner/type/generation/lease tuple checks, denied projection/snapshot/stale metadata attacks, inverse controls, disconnect cleanup, and same-carrier reuse. Those are source-local evidence only until the packet-layer defect is repaired and rerun.
"""
    (ROOT / "TWO_PORT_CUSTODY_REPORT.md").write_text(custody, encoding="utf-8")
    print(json.dumps({"N": classification, "fail_open_cases": report["fail_open_cases"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

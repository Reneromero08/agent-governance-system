#!/usr/bin/env python3
"""Durable, one-shot SSH/SCP state machine for a future exact Gate A authority.

Nothing runs at import time.  Qualification injects a command runner, clocks,
and failure checkpoints, so no test opens a network connection or invokes the
physical runtime.  The first target operation is a read-only namespace scan;
the first target mutation is an atomic authority-bound claim outside every
cleanup root.

The injected fake-command surface is the zero-contact null baseline; stage and
receipt mutations must fail against it without opening a network connection.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import build_gate_a_execution_bundle as bundle
import gate_a_process_custody as process_custody


class TransportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportError(message)


class TransportStage(enum.Enum):
    AUTHORITY_VALIDATED = "authority_validated"
    REMOTE_NAMESPACE_INSPECTED = "remote_namespace_inspected"
    AUTHORITY_CLAIMED = "authority_claimed"
    BUNDLE_STAGED = "bundle_staged"
    AUTHORITY_STAGED = "authority_staged"
    TARGET_RUNNER_STARTED = "target_runner_started"
    POST_RUNTIME_PROCESS_SCANNED = "post_runtime_process_scanned"
    EVIDENCE_ARCHIVED = "evidence_archived"
    EVIDENCE_COPIED_BACK = "evidence_copied_back"
    COPY_BACK_VERIFIED = "copy_back_verified"
    COPY_BACK_RECEIPT_UPLOADED = "copy_back_receipt_uploaded"
    REMOTE_CLEANUP_ATTEMPTED = "remote_cleanup_attempted"
    POST_CLEANUP_PROCESS_SCANNED = "post_cleanup_process_scanned"
    FINAL_LOCAL_SEALED = "final_local_sealed"


@dataclass(frozen=True)
class HostExecutionRequest:
    target: str
    authority_path: Path
    authority_sha256: str
    reviewed_adapter_head: str
    independent_review_id: int
    execution_bundle_sha256: str
    schedule_sha256: str
    namespace_sha256: str
    remote_execution_root: str
    remote_output_root: str
    local_evidence_root: Path
    authority_bytes: bytes = b""
    schedule_bytes: bytes = b""
    manifest_bytes: bytes = b""
    source_review_binding: dict[str, Any] | None = None
    authority_bearing_execution_commit: str = ""
    reviewed_source_tree: str = ""
    authority_bearing_execution_tree: str = ""
    authority_git_blob_sha1: str = ""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
FailureInjector = Callable[[str], None]


def _run_command(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def _stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_line_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value) + b"\n")


def _expected_claim(request: HostExecutionRequest) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_TRANSPORT_CLAIM_V1",
        "authority_sha256": request.authority_sha256,
        "execution_bundle_sha256": request.execution_bundle_sha256,
        "maximum_execution_count": 1,
        "automatic_retry": False,
    }


def _expected_execution_started(request: HostExecutionRequest) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EXECUTION_STARTED_V1",
        "authority_sha256": request.authority_sha256,
        "execution_bundle_sha256": request.execution_bundle_sha256,
        "runtime_execution_count": 1,
        "automatic_retry": False,
    }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short durable write: {path.name}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> tuple[dict[str, Any], str]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid evidence path: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        data = path.read_bytes()
        files.append({"path": relative, "size": len(data), "sha256": _sha256_bytes(data)})
    require(files, "evidence inventory is empty")
    inventory = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1",
        "files": files,
    }
    return inventory, _sha256_bytes(_canonical_bytes(inventory))


def _validate_inventory_shape(value: dict[str, Any]) -> str:
    require(set(value) == {"schema_id", "files"}, "evidence inventory key set mismatch")
    require(value["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1", "evidence inventory schema mismatch")
    require(isinstance(value["files"], list) and value["files"], "evidence inventory files missing")
    previous = ""
    for item in value["files"]:
        require(isinstance(item, dict) and set(item) == {"path", "size", "sha256"}, "evidence inventory entry mismatch")
        path = item["path"]
        require(isinstance(path, str) and path > previous and not path.startswith("/") and ".." not in Path(path).parts, "evidence inventory path ordering or safety mismatch")
        require(isinstance(item["size"], int) and item["size"] >= 0, "evidence inventory size malformed")
        digest = item["sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "evidence inventory digest malformed")
        previous = path
    return _sha256_bytes(_canonical_bytes(value))


def validate_source_review_binding(
    value: dict[str, Any],
    *,
    request: HostExecutionRequest,
    manifest: dict[str, Any],
) -> None:
    require(set(value) == {
        "schema_id", "reviewed_source_commit", "reviewed_source_tree",
        "independent_review_id", "authority_bearing_execution_commit",
        "authority_bearing_execution_tree", "authority_sha256",
        "authority_git_blob_sha1", "source_identities", "schedule_sha256",
        "target_namespace_sha256", "execution_bundle_sha256",
        "deterministic_archive_sha256", "target_identity_sha256", "target",
        "remote_execution_root", "remote_output_root",
    }, "source-review binding key set mismatch")
    require(value["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_SOURCE_REVIEW_BINDING_V1", "source-review binding schema mismatch")
    require(value["reviewed_source_commit"] == request.reviewed_adapter_head, "source-review commit mismatch")
    require(value["independent_review_id"] == request.independent_review_id, "source-review ID mismatch")
    require(value["authority_bearing_execution_commit"] == request.authority_bearing_execution_commit, "authority-bearing commit mismatch")
    expected_git_objects = {
        "reviewed_source_tree": request.reviewed_source_tree,
        "authority_bearing_execution_tree": request.authority_bearing_execution_tree,
        "authority_git_blob_sha1": request.authority_git_blob_sha1,
    }
    for key, expected in expected_git_objects.items():
        field = value[key]
        require(isinstance(field, str) and len(field) == 40 and all(c in "0123456789abcdef" for c in field), f"source-review {key} malformed")
        require(field == expected, f"source-review {key} mismatch")
    require(value["authority_sha256"] == request.authority_sha256, "source-review authority digest mismatch")
    require(value["schedule_sha256"] == request.schedule_sha256, "source-review schedule mismatch")
    require(value["target_namespace_sha256"] == request.namespace_sha256, "source-review namespace mismatch")
    require(value["execution_bundle_sha256"] == request.execution_bundle_sha256, "source-review bundle mismatch")
    require(value["deterministic_archive_sha256"] == manifest.get("deterministic_archive_sha256"), "source-review archive mismatch")
    require(value["target_identity_sha256"] == manifest.get("target_identity_stdout_sha256"), "source-review target identity mismatch")
    require(value["target"] == request.target, "source-review target mismatch")
    require(value["remote_execution_root"] == request.remote_execution_root and value["remote_output_root"] == request.remote_output_root, "source-review root mismatch")
    expected_identities = []
    for entry in sorted(manifest.get("files", []), key=lambda item: item["package_path"]):
        expected_identities.append({
            "role": entry["role"],
            "package_path": entry["package_path"],
            "source_repository_path": entry["source_repository_path"],
            "git_blob_sha1": entry["git_blob_sha1"],
            "git_mode": entry["git_mode"],
            "sha256": entry["sha256"],
            "byte_size": entry["byte_size"],
        })
    require(expected_identities and value["source_identities"] == expected_identities, "source-review source identities mismatch")


def validate_cleanup_result(value: dict[str, Any], *, request: HostExecutionRequest) -> None:
    expected_keys = {
        "cleanup_return_code", "cleanup_mode", "cleanup_runner_stdout",
        "cleanup_runner_stderr", "execution_root_absent", "output_root_absent",
        "stage_absent", "authority_absent", "archive_absent", "receipt_absent",
        "target_inventory_absent", "claim_retained", "claim_sha256",
        "execution_started_sha256",
    }
    require(set(value) == expected_keys, "cleanup result key set mismatch")
    require(value["cleanup_return_code"] == 0, "remote cleanup returned nonzero")
    require(value["cleanup_mode"] in {"verified_copyback", "no_output_created"}, "remote cleanup mode is not closed")
    require(isinstance(value["cleanup_runner_stdout"], str) and isinstance(value["cleanup_runner_stderr"], str), "cleanup raw streams malformed")
    for key in (
        "execution_root_absent", "output_root_absent", "stage_absent",
        "authority_absent", "archive_absent", "receipt_absent",
        "target_inventory_absent",
    ):
        require(value[key] is True, f"cleanup absence proof failed: {key}")
    expected_claim_sha256 = _canonical_line_sha256(_expected_claim(request))
    expected_started_sha256 = _canonical_line_sha256(_expected_execution_started(request))
    require(value["claim_retained"] is True and value["claim_sha256"] == expected_claim_sha256, "durable authority claim was not retained exactly")
    if value["cleanup_mode"] == "verified_copyback":
        require(value["execution_started_sha256"] == expected_started_sha256, "execution-start marker was not retained exactly")
        require(value["cleanup_runner_stderr"] == "", "cleanup runner wrote stderr")
        try:
            inner = json.loads(value["cleanup_runner_stdout"])
        except json.JSONDecodeError as exc:
            raise TransportError("cleanup runner stdout is malformed") from exc
        expected_inner = {
            "status": "GATE_A_CLEANUP_COMPLETE_AFTER_VERIFIED_COPY_BACK",
            "remote_output_root": request.remote_output_root,
            "claim_retained": True,
            "claim_sha256": expected_claim_sha256,
            "execution_started_sha256": expected_started_sha256,
        }
        require(inner == expected_inner, "cleanup runner receipt binding mismatch")
    else:
        require(value["cleanup_runner_stdout"] == "" and value["cleanup_runner_stderr"] == "", "no-output cleanup unexpectedly invoked a runner")
        require(value["execution_started_sha256"] in {None, expected_started_sha256}, "no-output execution marker mismatch")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        names: set[str] = set()
        for member in archive.getmembers():
            require(member.isfile(), f"non-file evidence member rejected: {member.name}")
            require(member.name not in names, f"duplicate evidence member: {member.name}")
            names.add(member.name)
            target = (destination / member.name).resolve()
            require(os.path.commonpath((str(root), str(target))) == str(root), f"evidence member escapes root: {member.name}")
            require(not target.exists(), f"duplicate evidence target: {member.name}")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = archive.extractfile(member)
            require(source is not None, f"evidence member unreadable: {member.name}")
            _exclusive_bytes(target, source.read())
    _fsync_directory(destination)


class HostEvidencePacket:
    REQUIRED_SUCCESS_FILES = frozenset({
        "AUTHORITY_ARTIFACT.json",
        "SCHEDULE.json",
        "EXECUTION_BUNDLE_MANIFEST.json",
        "SOURCE_REVIEW_BINDING.json",
        "HOST_COMMANDS.jsonl",
        "TARGET_EXECUTION_RECEIPT.json",
        "TARGET_EVIDENCE_INVENTORY.json",
        "COPY_BACK_RECEIPT.json",
        "POST_RUNTIME_PROCESS_RECEIPT.json",
        "POST_CLEANUP_PROCESS_RECEIPT.json",
        "CLEANUP_RECEIPT.json",
        "FINAL_EVIDENCE_INVENTORY.json",
        "FINAL_BINDINGS.json",
    })

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._ledger_path = root / "HOST_COMMANDS.jsonl"
        self._ledger = self._ledger_path.open("x", encoding="utf-8", newline="\n")
        self._ledger_closed = False
        _fsync_directory(root)

    def write_bytes(self, name: str, data: bytes) -> None:
        require("/" not in name and "\\" not in name, "top-level packet name required")
        _exclusive_bytes(self.root / name, data)

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        self.write_bytes(name, json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")

    def exists(self, name: str) -> bool:
        path = self.root / name
        return path.is_file() and not path.is_symlink()

    def append_command(self, value: dict[str, Any]) -> None:
        require(not self._ledger_closed, "host command ledger already closed")
        self._ledger.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        self._ledger.flush()
        os.fsync(self._ledger.fileno())

    def close_ledger(self) -> None:
        if not self._ledger_closed:
            self._ledger.flush()
            os.fsync(self._ledger.fileno())
            self._ledger.close()
            self._ledger_closed = True
            _fsync_directory(self.root)

    def file_sha256(self, name: str) -> str:
        path = self.root / name
        require(path.is_file() and not path.is_symlink(), f"packet file missing: {name}")
        return _sha256_bytes(path.read_bytes())


def validate_final_packet(root: Path) -> dict[str, Any]:
    inventory_path = root / "FINAL_EVIDENCE_INVENTORY.json"
    require(inventory_path.is_file() and not inventory_path.is_symlink(), "final evidence inventory missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(set(inventory) == {"schema_id", "self_exclusion", "files"}, "final inventory key set mismatch")
    require(inventory["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_FINAL_EVIDENCE_INVENTORY_V1", "final inventory schema mismatch")
    require(inventory["self_exclusion"] == "FINAL_EVIDENCE_INVENTORY.json", "final inventory self-exclusion mismatch")
    expected, digest = _inventory(root, exclude=frozenset({"FINAL_EVIDENCE_INVENTORY.json"}))
    require(inventory["files"] == expected["files"], "final evidence inventory does not close over packet")
    present = {path.name for path in root.iterdir() if path.is_file()}
    require(HostEvidencePacket.REQUIRED_SUCCESS_FILES <= present, "final host packet is incomplete")
    return {"status": "GATE_A_FINAL_PACKET_VALID", "final_inventory_sha256": digest, "file_count": len(inventory["files"])}


class SshScpTransport:
    """One no-retry state machine instantiated only after exact authority custody."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner = _run_command,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        failure_injector: FailureInjector | None = None,
    ):
        self._run = command_runner
        self._clock_ns = clock_ns
        self._monotonic_ns = monotonic_ns
        self._inject = failure_injector
        self._used = False
        self._sequence = 0
        self._attempted_commands: set[str] = set()
        self._completed_stages: list[str] = []
        self._runner_start_count = 0
        self._packet: HostEvidencePacket | None = None
        self._last_operation = "authority_validated"

    def _checkpoint(self, name: str) -> None:
        self._last_operation = name
        if self._inject is not None:
            self._inject(name)

    def _complete(self, stage: TransportStage) -> None:
        if stage.value not in self._completed_stages:
            self._completed_stages.append(stage.value)

    def _invoke(
        self,
        stage: TransportStage,
        argv: list[str],
        *,
        input_text: str | None = None,
        timeout: int = 120,
        check: bool = True,
        operation: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        require(self._packet is not None, "host packet unavailable")
        operation_key = operation or stage.value
        require(operation_key not in self._attempted_commands, f"transport operation repeated: {operation_key}")
        self._attempted_commands.add(operation_key)
        self._last_operation = operation_key
        self._checkpoint(operation_key)
        self._sequence += 1
        start_utc = self._clock_ns()
        start_mono = self._monotonic_ns()
        timed_out = False
        return_code: int | None = None
        stdout = ""
        stderr = ""
        failure: str | None = None
        try:
            completed = self._run(argv, input_text=input_text, timeout=timeout)
            return_code = completed.returncode
            stdout = _stream_text(completed.stdout)
            stderr = _stream_text(completed.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _stream_text(exc.stdout)
            stderr = _stream_text(exc.stderr)
            failure = "COMMAND_TIMEOUT"
            completed = subprocess.CompletedProcess(argv, -1, stdout=stdout, stderr=stderr)
        except OSError as exc:
            failure = f"COMMAND_UNOBSERVABLE:{type(exc).__name__}"
            stderr = str(exc)
            completed = subprocess.CompletedProcess(argv, -1, stdout="", stderr=stderr)
        end_mono = self._monotonic_ns()
        end_utc = self._clock_ns()
        record = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_HOST_COMMAND_V1",
            "sequence": self._sequence,
            "stage": stage.value,
            "operation": operation_key,
            "command": argv,
            "command_sha256": _sha256_bytes(_canonical_bytes(argv)),
            "stdin_sha256": _sha256_text(input_text or ""),
            "stdin_size": len((input_text or "").encode("utf-8")),
            "start_utc_ns": start_utc,
            "end_utc_ns": end_utc,
            "start_monotonic_ns": start_mono,
            "end_monotonic_ns": end_mono,
            "timeout_seconds": timeout,
            "timed_out": timed_out,
            "return_code": return_code,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
            "stdout_sha256": _sha256_text(stdout),
            "stderr_sha256": _sha256_text(stderr),
            "failure": failure,
        }
        self._packet.append_command(record)
        if failure is not None:
            raise TransportError(f"{operation_key} failed closed: {failure}")
        if check:
            require(return_code == 0, f"{operation_key} failed: {stderr.strip()}")
        return completed

    @staticmethod
    def _json_stdout(completed: subprocess.CompletedProcess[str], context: str) -> dict[str, Any]:
        require(completed.returncode == 0, f"{context} returned nonzero")
        try:
            value = json.loads(_stream_text(completed.stdout))
        except json.JSONDecodeError as exc:
            raise TransportError(f"{context} returned malformed JSON") from exc
        require(isinstance(value, dict), f"{context} must return an object")
        return value

    def _seed_packet(self, request: HostExecutionRequest) -> None:
        require(not request.local_evidence_root.exists(), "local evidence root must be absent")
        require(request.authority_bytes and _sha256_bytes(request.authority_bytes) == request.authority_sha256, "retained authority bytes mismatch")
        require(request.schedule_bytes and request.manifest_bytes, "retained schedule/manifest bytes missing")
        require(isinstance(request.source_review_binding, dict), "source-review binding missing")
        schedule = json.loads(request.schedule_bytes.decode("utf-8"))
        manifest = json.loads(request.manifest_bytes.decode("utf-8"))
        require(schedule.get("schedule_sha256") == request.schedule_sha256, "retained schedule digest mismatch")
        require(manifest.get("execution_bundle_sha256") == request.execution_bundle_sha256, "retained manifest bundle mismatch")
        require(manifest.get("target_namespace_sha256") == request.namespace_sha256, "retained manifest namespace mismatch")
        validate_source_review_binding(request.source_review_binding, request=request, manifest=manifest)
        self._packet = HostEvidencePacket(request.local_evidence_root)
        self._packet.write_bytes("AUTHORITY_ARTIFACT.json", request.authority_bytes)
        self._packet.write_bytes("SCHEDULE.json", request.schedule_bytes)
        self._packet.write_bytes("EXECUTION_BUNDLE_MANIFEST.json", request.manifest_bytes)
        self._packet.write_json("SOURCE_REVIEW_BINDING.json", request.source_review_binding)
        self._complete(TransportStage.AUTHORITY_VALIDATED)

    @staticmethod
    def _paths(request: HostExecutionRequest) -> dict[str, str]:
        prefix = f"/root/.catcas_gate_a_{request.authority_sha256[:16]}"
        return {
            "prefix": prefix,
            "stage": prefix + ".bundle.tar",
            "authority": prefix + ".authority.json",
            "archive": prefix + ".evidence.tar",
            "receipt": prefix + ".copy_back.json",
            "target_inventory": prefix + ".target_inventory.json",
            "claim": f"/root/.catcas_gate_a_claim_{request.authority_sha256}",
        }

    def _namespace_preflight(self, request: HostExecutionRequest, paths: dict[str, str]) -> None:
        script = f'''import json, os\ndef state(path):\n    try: os.lstat(path)\n    except FileNotFoundError: return "absent"\n    except OSError as exc: return "unobservable:" + type(exc).__name__\n    return "present"\ntry:\n    matches = sorted(os.path.join("/root", n) for n in os.listdir("/root") if os.path.join("/root", n).startswith({paths["prefix"]!r}))\nexcept OSError as exc:\n    print(json.dumps({{"inspection_complete":False,"error":type(exc).__name__}})); raise SystemExit(0)\nprint(json.dumps({{"inspection_complete":True,"execution_root":state({request.remote_execution_root!r}),"output_root":state({request.remote_output_root!r}),"stage":state({paths["stage"]!r}),"authority":state({paths["authority"]!r}),"archive":state({paths["archive"]!r}),"receipt":state({paths["receipt"]!r}),"target_inventory":state({paths["target_inventory"]!r}),"claim":state({paths["claim"]!r}),"prefix_matches":matches}},sort_keys=True))\n'''
        value = self._json_stdout(
            self._invoke(
                TransportStage.REMOTE_NAMESPACE_INSPECTED,
                ["ssh", request.target, "python3", "-"],
                input_text=script,
                operation="remote_namespace_inspected",
            ),
            "remote namespace preflight",
        )
        expected = {"inspection_complete", "execution_root", "output_root", "stage", "authority", "archive", "receipt", "target_inventory", "claim", "prefix_matches"}
        require(set(value) == expected, "preflight key set mismatch")
        require(value["inspection_complete"] is True, "remote namespace unobservable")
        for key in expected - {"inspection_complete", "prefix_matches"}:
            require(value[key] == "absent", f"remote {key} is not absent")
        require(value["prefix_matches"] == [], "authority-bound remote prefix collision")
        self._complete(TransportStage.REMOTE_NAMESPACE_INSPECTED)

    def _claim_authority(self, request: HostExecutionRequest, paths: dict[str, str]) -> dict[str, Any]:
        claim = _expected_claim(request)
        script = f'''import hashlib, json, os, pathlib\nroot=pathlib.Path({paths["claim"]!r})\nroot.mkdir(mode=0o700,parents=False,exist_ok=False)\nparent_fd=os.open(str(root.parent),os.O_RDONLY)\ntry: os.fsync(parent_fd)\nfinally: os.close(parent_fd)\npayload=(json.dumps({claim!r},sort_keys=True,separators=(",",":"))+"\\n").encode()\nfd=os.open(str(root/"CLAIM.json"),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)\ntry:\n    os.write(fd,payload); os.fsync(fd)\nfinally: os.close(fd)\ndir_fd=os.open(str(root),os.O_RDONLY)\ntry: os.fsync(dir_fd)\nfinally: os.close(dir_fd)\nprint(json.dumps({{"claim_created":True,"claim_root":str(root),"claim_sha256":hashlib.sha256(payload).hexdigest()}},sort_keys=True))\n'''
        value = self._json_stdout(
            self._invoke(
                TransportStage.AUTHORITY_CLAIMED,
                ["ssh", request.target, "python3", "-"],
                input_text=script,
                operation="authority_claimed",
            ),
            "durable authority claim",
        )
        require(set(value) == {"claim_created", "claim_root", "claim_sha256"}, "authority claim receipt mismatch")
        require(value["claim_created"] is True and value["claim_root"] == paths["claim"], "authority claim was not created")
        require(value["claim_sha256"] == _canonical_line_sha256(claim), "authority claim digest mismatch")
        self._packet.write_json("AUTHORITY_CLAIM_RECEIPT.json", value)  # type: ignore[union-attr]
        self._complete(TransportStage.AUTHORITY_CLAIMED)
        return value

    @staticmethod
    def _execute_script(request: HostExecutionRequest, paths: dict[str, str]) -> str:
        return f'''import hashlib, json, os, pathlib, signal, subprocess, sys, tarfile\nroot=pathlib.Path({request.remote_execution_root!r}); stage=pathlib.Path({paths["stage"]!r}); output=pathlib.Path({request.remote_output_root!r}); archive_path=pathlib.Path({paths["archive"]!r}); inventory_path=pathlib.Path({paths["target_inventory"]!r})\nroot.mkdir(mode=0o700,parents=False,exist_ok=False)\nwith tarfile.open(stage,"r") as archive:\n    base=root.resolve()\n    for member in archive.getmembers():\n        target=(root/member.name).resolve()\n        if not member.isfile() or os.path.commonpath((str(base),str(target))) != str(base): raise SystemExit("unsafe deployment member")\n    archive.extractall(root)\ncmd=[sys.executable,"-B",str(root/"adapter/gate_a_target_runner.py"),"--execute-authorized","--authority-artifact",{paths["authority"]!r},"--authority-sha256",{request.authority_sha256!r},"--execution-bundle-sha256",{request.execution_bundle_sha256!r},"--source-head",{request.reviewed_adapter_head!r},"--independent-review-id",{str(request.independent_review_id)!r},"--schedule-sha256",{request.schedule_sha256!r},"--target",{request.target!r},"--namespace-sha256",{request.namespace_sha256!r},"--output-root",str(output),"--transport-claim-root",{paths["claim"]!r}]\nprocess=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)\ntimed_out=False\ntry: out_b,err_b=process.communicate(timeout=45)\nexcept subprocess.TimeoutExpired:\n    timed_out=True; os.killpg(process.pid,signal.SIGTERM)\n    try: out_b,err_b=process.communicate(timeout=2)\n    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); out_b,err_b=process.communicate()\nsys.path.insert(0,str(root/"adapter"))\nimport gate_a_process_custody as pc\npost_path=output/"POST_RUNTIME_PROCESS_RECEIPT.json"\nif post_path.is_file(): post=json.loads(post_path.read_text())\nelse:\n    post=pc.scan_processes("post_runtime")\n    if output.is_dir():\n        fd=os.open(str(post_path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); data=(json.dumps(post,sort_keys=True,indent=2)+"\\n").encode()\n        try: os.write(fd,data); os.fsync(fd)\n        finally: os.close(fd)\ndef inventory(base):\n    files=[]\n    for path in sorted(base.rglob("*")):\n        if path.is_dir() and not path.is_symlink(): continue\n        if not path.is_file() or path.is_symlink(): raise RuntimeError("unsafe evidence path")\n        data=path.read_bytes(); files.append({{"path":path.relative_to(base).as_posix(),"size":len(data),"sha256":hashlib.sha256(data).hexdigest()}})\n    value={{"schema_id":"CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1","files":files}}; digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest(); return value,digest\ninv=None; inv_digest=None\nif output.is_dir():\n    inv,inv_digest=inventory(output); inventory_path.write_text(json.dumps(inv,sort_keys=True,indent=2)+"\\n")\n    with tarfile.open(archive_path,"w") as archive:\n        for path in sorted(output.rglob("*")):\n            if path.is_file() and not path.is_symlink(): archive.add(path,arcname=path.relative_to(output).as_posix(),recursive=False)\nprint(json.dumps({{"runner_command":cmd,"runner_return_code":process.returncode,"runner_stdout":out_b.decode("utf-8","replace"),"runner_stderr":err_b.decode("utf-8","replace"),"runner_stdout_sha256":hashlib.sha256(out_b).hexdigest(),"runner_stderr_sha256":hashlib.sha256(err_b).hexdigest(),"target_timeout":timed_out,"evidence_archive_created":archive_path.is_file(),"target_evidence_inventory":inv,"target_evidence_inventory_sha256":inv_digest,"post_runtime_process_receipt":post}},sort_keys=True))\n'''

    @staticmethod
    def _recovery_archive_script(request: HostExecutionRequest, paths: dict[str, str]) -> str:
        return f'''import hashlib,json,os,pathlib,tarfile
root=pathlib.Path({request.remote_execution_root!r}); output=pathlib.Path({request.remote_output_root!r}); archive_path=pathlib.Path({paths["archive"]!r}); inventory_path=pathlib.Path({paths["target_inventory"]!r})
created=False; inv=None; digest=None; post=None
if output.is_dir() and not output.is_symlink():
    files=[]
    for path in sorted(output.rglob("*")):
        if path.is_dir() and not path.is_symlink(): continue
        if not path.is_file() or path.is_symlink(): raise RuntimeError("unsafe recovery evidence path")
        data=path.read_bytes(); files.append({{"path":path.relative_to(output).as_posix(),"size":len(data),"sha256":hashlib.sha256(data).hexdigest()}})
    if not files: raise RuntimeError("recovery evidence is empty")
    inv={{"schema_id":"CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1","files":files}}; digest=hashlib.sha256(json.dumps(inv,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    post_path=output/"POST_RUNTIME_PROCESS_RECEIPT.json"
    if post_path.is_file() and not post_path.is_symlink():
        try: post=json.loads(post_path.read_text())
        except Exception as exc: post={{"schema_id":"CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_UNOBSERVABLE_V1","phase":"post_runtime","scan_complete":False,"failure":"RECOVERED_RECEIPT_UNREADABLE:"+type(exc).__name__}}
    candidate=root/".gate_a_evidence_recovery.tar"
    if candidate.exists(): raise RuntimeError("recovery archive candidate collision")
    with tarfile.open(candidate,"x") as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.is_symlink(): archive.add(path,arcname=path.relative_to(output).as_posix(),recursive=False)
    fd=os.open(str(candidate),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    os.replace(candidate,archive_path)
    inventory_candidate=root/".gate_a_inventory_recovery.json"
    payload=(json.dumps(inv,sort_keys=True,indent=2)+"\\n").encode()
    fd=os.open(str(inventory_candidate),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try: os.write(fd,payload); os.fsync(fd)
    finally: os.close(fd)
    os.replace(inventory_candidate,inventory_path)
    parent_fd=os.open(str(archive_path.parent),os.O_RDONLY)
    try: os.fsync(parent_fd)
    finally: os.close(parent_fd)
    created=True
print(json.dumps({{"evidence_archive_created":created,"created_by_recovery":created,"target_evidence_inventory":inv,"target_evidence_inventory_sha256":digest,"post_runtime_process_receipt":post}},sort_keys=True))
'''

    def _cleanup_script(self, request: HostExecutionRequest, paths: dict[str, str]) -> str:
        return f'''import hashlib,json,pathlib,shutil,subprocess,sys
root=pathlib.Path({request.remote_execution_root!r}); output=pathlib.Path({request.remote_output_root!r}); receipt=pathlib.Path({paths["receipt"]!r}); claim=pathlib.Path({paths["claim"]!r}); archive=pathlib.Path({paths["archive"]!r}); target_inventory=pathlib.Path({paths["target_inventory"]!r}); runner=root/"adapter/gate_a_target_runner.py"
allow_no_output_cleanup={self._runner_start_count == 0!r}
rc=70; out=""; err=""; cleanup_mode="blocked_unverified_copyback"
if output.exists() and runner.is_file() and receipt.is_file():
    cmd=[sys.executable,"-B",str(runner),"--cleanup-after-verified-copy","--output-root",str(output),"--authority-sha256",{request.authority_sha256!r},"--execution-bundle-sha256",{request.execution_bundle_sha256!r},"--copy-back-receipt",str(receipt),"--transport-claim-root",str(claim)]
    completed=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False); rc,out,err=completed.returncode,completed.stdout,completed.stderr; cleanup_mode="verified_copyback" if rc==0 else "blocked_unverified_copyback"
elif not output.exists() and not allow_no_output_cleanup: rc=72; cleanup_mode="blocked_runner_start_unresolved"
elif not output.exists() and (archive.exists() or target_inventory.exists()): rc=71; cleanup_mode="blocked_available_evidence"
elif not output.exists(): rc=0; cleanup_mode="no_output_created"
if rc==0:
    if root.exists(): shutil.rmtree(root)
    for value in ({paths["stage"]!r},{paths["authority"]!r},{paths["archive"]!r},{paths["receipt"]!r},{paths["target_inventory"]!r}):
        try: pathlib.Path(value).unlink()
        except FileNotFoundError: pass
claim_file=claim/"CLAIM.json"; started_file=claim/"EXECUTION_STARTED.json"
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else None
print(json.dumps({{"cleanup_return_code":rc,"cleanup_mode":cleanup_mode,"cleanup_runner_stdout":out,"cleanup_runner_stderr":err,"execution_root_absent":not root.exists(),"output_root_absent":not output.exists(),"stage_absent":not pathlib.Path({paths["stage"]!r}).exists(),"authority_absent":not pathlib.Path({paths["authority"]!r}).exists(),"archive_absent":not archive.exists(),"receipt_absent":not receipt.exists(),"target_inventory_absent":not target_inventory.exists(),"claim_retained":claim.is_dir() and not claim.is_symlink() and claim_file.is_file() and not claim_file.is_symlink(),"claim_sha256":digest(claim_file),"execution_started_sha256":digest(started_file)}},sort_keys=True))
'''

    def _retain_execution_receipt(self, value: dict[str, Any]) -> None:
        require(self._packet is not None, "host packet unavailable")
        require(set(value) == {
            "runner_command", "runner_return_code", "runner_stdout",
            "runner_stderr", "runner_stdout_sha256", "runner_stderr_sha256",
            "target_timeout", "evidence_archive_created",
            "target_evidence_inventory", "target_evidence_inventory_sha256",
            "post_runtime_process_receipt",
        }, "target execution receipt key set mismatch")
        require(isinstance(value["runner_command"], list) and value["runner_command"], "target runner command missing")
        require(isinstance(value["runner_stdout"], str) and isinstance(value["runner_stderr"], str), "target runner streams malformed")
        require(value["runner_stdout_sha256"] == _sha256_text(value["runner_stdout"]), "target runner stdout digest mismatch")
        require(value["runner_stderr_sha256"] == _sha256_text(value["runner_stderr"]), "target runner stderr digest mismatch")
        require(isinstance(value["target_timeout"], bool) and isinstance(value["evidence_archive_created"], bool), "target execution state malformed")
        self._packet.write_json("TARGET_EXECUTION_RECEIPT.json", value)
        post = value.get("post_runtime_process_receipt")
        require(isinstance(post, dict), "post-runtime process receipt missing")
        self._packet.write_json("POST_RUNTIME_PROCESS_RECEIPT.json", post)
        process_custody.validate_process_receipt(post, expected_phase="post_runtime")
        self._complete(TransportStage.POST_RUNTIME_PROCESS_SCANNED)

    def _copy_back(
        self,
        request: HostExecutionRequest,
        paths: dict[str, str],
        copied_archive: Path,
        execution: dict[str, Any],
        *,
        operation_prefix: str = "",
    ) -> dict[str, Any]:
        require(self._packet is not None, "host packet unavailable")
        operation = operation_prefix + "evidence_download"
        self._invoke(
            TransportStage.EVIDENCE_COPIED_BACK,
            ["scp", f"{request.target}:{paths['archive']}", str(copied_archive)],
            timeout=120,
            operation=operation,
        )
        target_root = self._packet.root / "TARGET_OUTPUT"
        self._checkpoint(operation_prefix + "safe_extract")
        _safe_extract(copied_archive, target_root)
        downloaded_inventory, downloaded_digest = _inventory(target_root)
        target_inventory = execution.get("target_evidence_inventory")
        target_digest = execution.get("target_evidence_inventory_sha256")
        self._checkpoint(operation_prefix + "target_inventory_verification")
        require(isinstance(target_inventory, dict), "target evidence inventory missing")
        require(_validate_inventory_shape(target_inventory) == target_digest, "target evidence inventory digest mismatch")
        require(target_inventory == downloaded_inventory and target_digest == downloaded_digest, "downloaded evidence differs from target inventory")
        self._packet.write_json("TARGET_EVIDENCE_INVENTORY.json", target_inventory)
        self._complete(TransportStage.EVIDENCE_COPIED_BACK)
        archive_sha256 = _sha256_bytes(copied_archive.read_bytes())
        receipt = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1",
            "remote_output_root": request.remote_output_root,
            "authority_sha256": request.authority_sha256,
            "execution_bundle_sha256": request.execution_bundle_sha256,
            "retained_evidence_custody_verified": True,
            "evidence_inventory_sha256": downloaded_digest,
            "target_evidence_inventory_sha256": target_digest,
            "downloaded_evidence_inventory_sha256": downloaded_digest,
            "archive_sha256": archive_sha256,
            "copy_back_complete": True,
        }
        self._checkpoint(operation_prefix + "copy_back_receipt_persist")
        self._packet.write_json("COPY_BACK_RECEIPT.json", receipt)
        self._complete(TransportStage.COPY_BACK_VERIFIED)
        return receipt

    def _upload_copyback_receipt(self, request: HostExecutionRequest, paths: dict[str, str], *, operation: str = "copy_back_receipt_upload") -> None:
        require(self._packet is not None, "host packet unavailable")
        self._invoke(
            TransportStage.COPY_BACK_RECEIPT_UPLOADED,
            ["scp", str(self._packet.root / "COPY_BACK_RECEIPT.json"), f"{request.target}:{paths['receipt']}"],
            timeout=120,
            operation=operation,
        )
        self._complete(TransportStage.COPY_BACK_RECEIPT_UPLOADED)

    def _attempt_cleanup(self, request: HostExecutionRequest, paths: dict[str, str]) -> dict[str, Any]:
        require(self._packet is not None, "host packet unavailable")
        value: dict[str, Any]
        error: Exception | None = None
        try:
            completed = self._invoke(
                TransportStage.REMOTE_CLEANUP_ATTEMPTED,
                ["ssh", request.target, "python3", "-"],
                input_text=self._cleanup_script(request, paths),
                timeout=120,
                operation="remote_cleanup_attempted",
                check=False,
            )
            value = self._json_stdout(completed, "remote cleanup")
            receipt = {
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V1",
                "raw_response": completed.stdout,
                "raw_response_sha256": _sha256_text(completed.stdout),
                "raw_stderr": completed.stderr,
                "raw_stderr_sha256": _sha256_text(completed.stderr),
                "parsed": value,
            }
        except Exception as exc:
            error = exc
            value = {"cleanup_return_code": None, "cleanup_error": str(exc)}
            receipt = {
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V1",
                "raw_response": "",
                "raw_response_sha256": _sha256_text(""),
                "raw_stderr": str(exc),
                "raw_stderr_sha256": _sha256_text(str(exc)),
                "parsed": value,
            }
        if not self._packet.exists("CLEANUP_RECEIPT.json"):
            self._packet.write_json("CLEANUP_RECEIPT.json", receipt)
        if error is None:
            try:
                validate_cleanup_result(value, request=request)
            except Exception as validation_exc:
                error = validation_exc
        if error is not None:
            raise TransportError(f"remote cleanup failed closed: {error}") from error
        self._complete(TransportStage.REMOTE_CLEANUP_ATTEMPTED)
        return value

    def _post_cleanup_scan(self, request: HostExecutionRequest) -> dict[str, Any]:
        require(self._packet is not None, "host packet unavailable")
        try:
            completed = self._invoke(
                TransportStage.POST_CLEANUP_PROCESS_SCANNED,
                ["ssh", request.target, "python3", "-"],
                input_text=process_custody.render_remote_scan_script("post_cleanup"),
                timeout=30,
                operation="post_cleanup_process_scan",
            )
            receipt = self._json_stdout(completed, "post-cleanup process scan")
        except Exception as exc:
            receipt = {
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_UNOBSERVABLE_V1",
                "phase": "post_cleanup",
                "scan_complete": False,
                "failure": str(exc),
            }
        if not self._packet.exists("POST_CLEANUP_PROCESS_RECEIPT.json"):
            self._packet.write_json("POST_CLEANUP_PROCESS_RECEIPT.json", receipt)
        process_custody.validate_process_receipt(receipt, expected_phase="post_cleanup")
        self._complete(TransportStage.POST_CLEANUP_PROCESS_SCANNED)
        return receipt

    def _placeholder(self, name: str, *, failed_stage: str, reason: str) -> None:
        require(self._packet is not None, "host packet unavailable")
        if not self._packet.exists(name):
            self._packet.write_json(name, {
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_NOT_REACHED_V1",
                "artifact": name,
                "failed_stage": failed_stage,
                "reason": reason,
                "automatic_retry": False,
            })

    def _seal(
        self,
        request: HostExecutionRequest,
        *,
        status: str,
        failed_stage: str | None,
        primary_error: str | None,
        secondary_errors: list[str],
    ) -> dict[str, Any]:
        require(self._packet is not None, "host packet unavailable")
        self._checkpoint("final_local_seal")
        self._packet.close_ledger()
        bound_names = [
            "AUTHORITY_ARTIFACT.json", "SCHEDULE.json",
            "EXECUTION_BUNDLE_MANIFEST.json", "SOURCE_REVIEW_BINDING.json",
            "HOST_COMMANDS.jsonl", "TARGET_EXECUTION_RECEIPT.json",
            "TARGET_EVIDENCE_INVENTORY.json", "COPY_BACK_RECEIPT.json",
            "POST_RUNTIME_PROCESS_RECEIPT.json", "POST_CLEANUP_PROCESS_RECEIPT.json",
            "CLEANUP_RECEIPT.json",
        ]
        for optional_name in ("AUTHORITY_CLAIM_RECEIPT.json", "TRANSPORT_FAILURE_RECEIPT.json"):
            if self._packet.exists(optional_name):
                bound_names.append(optional_name)
        bindings = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FINAL_BINDINGS_V1",
            "status": status,
            "authority_sha256": request.authority_sha256,
            "execution_bundle_sha256": request.execution_bundle_sha256,
            "schedule_sha256": request.schedule_sha256,
            "namespace_sha256": request.namespace_sha256,
            "reviewed_source_commit": request.reviewed_adapter_head,
            "independent_review_id": request.independent_review_id,
            "authority_bearing_execution_commit": request.authority_bearing_execution_commit,
            "completed_stages": self._completed_stages,
            "failed_stage": failed_stage,
            "primary_error": primary_error,
            "secondary_errors": secondary_errors,
            "runner_start_count": self._runner_start_count,
            "transport_execution_count": 1,
            "retry_count": 0,
            "automatic_retry": False,
            "artifact_sha256": {name: self._packet.file_sha256(name) for name in bound_names},
        }
        self._packet.write_json("FINAL_BINDINGS.json", bindings)
        inventory, inventory_sha256 = _inventory(
            self._packet.root,
            exclude=frozenset({"FINAL_EVIDENCE_INVENTORY.json"}),
        )
        final_inventory = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FINAL_EVIDENCE_INVENTORY_V1",
            "self_exclusion": "FINAL_EVIDENCE_INVENTORY.json",
            "files": inventory["files"],
        }
        self._packet.write_json("FINAL_EVIDENCE_INVENTORY.json", final_inventory)
        self._complete(TransportStage.FINAL_LOCAL_SEALED)
        validation = validate_final_packet(self._packet.root)
        validation["inventory_content_sha256"] = inventory_sha256
        validation["final_inventory_sha256"] = self._packet.file_sha256("FINAL_EVIDENCE_INVENTORY.json")
        return validation

    def execute(self, request: HostExecutionRequest) -> dict[str, Any]:
        require(not self._used, "transport may execute only once")
        self._used = True
        require(request.target == "root@192.168.137.100", "target mismatch")
        require(len(request.authority_sha256) == 64, "authority digest malformed")
        require(len(request.authority_bearing_execution_commit) == 40, "authority-bearing commit missing")
        paths = self._paths(request)
        mutation_attempted = False
        claim_state = "not_attempted"
        claim_preserved_verified = False
        copy_back_verified = False
        execution: dict[str, Any] | None = None
        cleanup: dict[str, Any] | None = None
        primary: Exception | None = None
        failed_stage: str | None = None
        secondary: list[str] = []
        sealed: dict[str, Any] | None = None

        self._seed_packet(request)
        with tempfile.TemporaryDirectory(prefix="gate_a_authorized_transport_") as tmp:
            temp = Path(tmp)
            deployment = temp / "bundle.tar"
            copied = temp / "evidence.tar"
            try:
                self._checkpoint("deployment_archive_build")
                bundle.write_deployment_archive(deployment, request.authority_bearing_execution_commit)
                self._namespace_preflight(request, paths)
                mutation_attempted = True
                claim_state = "uncertain"
                self._claim_authority(request, paths)
                claim_state = "confirmed"
                self._invoke(
                    TransportStage.BUNDLE_STAGED,
                    ["scp", str(deployment), f"{request.target}:{paths['stage']}"],
                    timeout=120,
                    operation="bundle_staged",
                )
                self._complete(TransportStage.BUNDLE_STAGED)
                self._invoke(
                    TransportStage.AUTHORITY_STAGED,
                    ["scp", str(self._packet.root / "AUTHORITY_ARTIFACT.json"), f"{request.target}:{paths['authority']}"],  # type: ignore[union-attr]
                    timeout=120,
                    operation="authority_staged",
                )
                self._complete(TransportStage.AUTHORITY_STAGED)
                self._runner_start_count += 1
                require(self._runner_start_count == 1, "target runner start attempted more than once")
                execution = self._json_stdout(
                    self._invoke(
                        TransportStage.TARGET_RUNNER_STARTED,
                        ["ssh", request.target, "python3", "-"],
                        input_text=self._execute_script(request, paths),
                        timeout=180,
                        operation="target_runner_started",
                    ),
                    "authorized target execution",
                )
                self._complete(TransportStage.TARGET_RUNNER_STARTED)
                self._retain_execution_receipt(execution)
                self._checkpoint("evidence_archive_creation")
                require(execution.get("evidence_archive_created") is True, "target evidence archive missing")
                self._complete(TransportStage.EVIDENCE_ARCHIVED)
                receipt = self._copy_back(request, paths, copied, execution)
                copy_back_verified = True
                self._upload_copyback_receipt(request, paths)
                cleanup = self._attempt_cleanup(request, paths)
                claim_preserved_verified = True
                self._post_cleanup_scan(request)
                self._checkpoint("cleanup_verification")
                require(cleanup.get("cleanup_mode") == "verified_copyback", "successful execution requires verified-copyback cleanup")
                self._checkpoint("target_result_verification")
                require(execution.get("target_timeout") is False, "target-local timeout fired")
                require(execution.get("runner_return_code") == 0, "authorized runtime failed after evidence custody")
                try:
                    runtime_result = json.loads(execution["runner_stdout"])
                except (KeyError, json.JSONDecodeError) as exc:
                    raise TransportError("target runner result malformed") from exc
                sealed = self._seal(
                    request,
                    status="GATE_A_AUTHORIZED_TRANSPORT_COMPLETE",
                    failed_stage=None,
                    primary_error=None,
                    secondary_errors=[],
                )
                return {
                    "status": "GATE_A_AUTHORIZED_TRANSPORT_COMPLETE",
                    "target_runner_result": runtime_result,
                    "copy_back_inventory_sha256": receipt["evidence_inventory_sha256"],
                    "final_inventory_sha256": sealed["final_inventory_sha256"],
                    "cleanup_verified": True,
                    "authority_claim_retained": True,
                    "automatic_retry": False,
                    "retry_count": 0,
                    "transport_execution_count": 1,
                }
            except Exception as exc:
                primary = exc
                failed_stage = self._last_operation

            if mutation_attempted:
                if not copy_back_verified:
                    try:
                        recovered = self._json_stdout(
                            self._invoke(
                                TransportStage.EVIDENCE_ARCHIVED,
                                ["ssh", request.target, "python3", "-"],
                                input_text=self._recovery_archive_script(request, paths),
                                timeout=120,
                                operation="recovery_archive",
                            ),
                            "recovery archive",
                        )
                        if recovered.get("evidence_archive_created") is True:
                            require(set(recovered) == {
                                "evidence_archive_created", "created_by_recovery",
                                "target_evidence_inventory", "target_evidence_inventory_sha256",
                                "post_runtime_process_receipt",
                            }, "recovery archive receipt key set mismatch")
                            require(recovered.get("created_by_recovery") is True, "recovery did not rebuild the evidence archive")
                            self._complete(TransportStage.EVIDENCE_ARCHIVED)
                            if execution is None:
                                execution = {
                                    "schema_id": "CAT_CAS_PHASE6B6_GATE_A_RECOVERED_TARGET_EXECUTION_V1",
                                    "runner_return_code": None,
                                    "runner_stdout": "",
                                    "runner_stderr": str(primary),
                                    "target_timeout": False,
                                    "evidence_archive_created": True,
                                    "target_evidence_inventory": recovered.get("target_evidence_inventory"),
                                    "target_evidence_inventory_sha256": recovered.get("target_evidence_inventory_sha256"),
                                    "post_runtime_process_receipt": recovered.get("post_runtime_process_receipt"),
                                }
                                if not self._packet.exists("TARGET_EXECUTION_RECEIPT.json"):  # type: ignore[union-attr]
                                    self._packet.write_json("TARGET_EXECUTION_RECEIPT.json", execution)  # type: ignore[union-attr]
                            recovery_execution = dict(execution)
                            recovery_execution["target_evidence_inventory"] = recovered.get("target_evidence_inventory")
                            recovery_execution["target_evidence_inventory_sha256"] = recovered.get("target_evidence_inventory_sha256")
                            self._copy_back(request, paths, copied, recovery_execution, operation_prefix="recovery_")
                            copy_back_verified = True
                            recovered_post = recovered.get("post_runtime_process_receipt")
                            require(isinstance(recovered_post, dict), "recovered post-runtime process receipt missing")
                            process_custody.validate_process_receipt(recovered_post, expected_phase="post_runtime")
                            if not self._packet.exists("POST_RUNTIME_PROCESS_RECEIPT.json"):  # type: ignore[union-attr]
                                self._packet.write_json("POST_RUNTIME_PROCESS_RECEIPT.json", recovered_post)  # type: ignore[union-attr]
                            self._upload_copyback_receipt(request, paths, operation="recovery_copy_back_receipt_upload")
                    except Exception as recovery_exc:
                        secondary.append(f"recovery:{recovery_exc}")
                if "remote_cleanup_attempted" not in self._attempted_commands:
                    try:
                        cleanup = self._attempt_cleanup(request, paths)
                        claim_preserved_verified = True
                    except Exception as cleanup_exc:
                        secondary.append(f"cleanup:{cleanup_exc}")
                if "post_cleanup_process_scan" not in self._attempted_commands:
                    try:
                        self._post_cleanup_scan(request)
                    except Exception as scan_exc:
                        secondary.append(f"post_cleanup_scan:{scan_exc}")

            assert primary is not None
            reason = str(primary)
            for name in (
                "TARGET_EXECUTION_RECEIPT.json",
                "TARGET_EVIDENCE_INVENTORY.json",
                "COPY_BACK_RECEIPT.json",
                "POST_RUNTIME_PROCESS_RECEIPT.json",
                "POST_CLEANUP_PROCESS_RECEIPT.json",
                "CLEANUP_RECEIPT.json",
            ):
                self._placeholder(name, failed_stage=failed_stage or "unknown", reason=reason)
            self._packet.write_json("TRANSPORT_FAILURE_RECEIPT.json", {  # type: ignore[union-attr]
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_TRANSPORT_FAILURE_V1",
                "failed_stage": failed_stage,
                "primary_error": reason,
                "secondary_errors": secondary,
                "mutation_attempted": mutation_attempted,
                "authority_claim_state": claim_state,
                "copy_back_verified": copy_back_verified,
                "cleanup_attempted": "remote_cleanup_attempted" in self._attempted_commands,
                "runner_start_count": self._runner_start_count,
                "retry_count": 0,
                "automatic_retry": False,
                "authority_claim_preserved": claim_preserved_verified,
            })
            try:
                sealed = self._seal(
                    request,
                    status="GATE_A_AUTHORIZED_TRANSPORT_FAILED_CLOSED",
                    failed_stage=failed_stage,
                    primary_error=reason,
                    secondary_errors=secondary,
                )
            except Exception as seal_exc:
                raise TransportError(f"transport failed ({reason}); local failure seal failed ({seal_exc})") from primary
            raise TransportError(
                f"transport failed closed at {failed_stage}: {reason}; final_inventory={sealed['final_inventory_sha256']}"
            ) from primary

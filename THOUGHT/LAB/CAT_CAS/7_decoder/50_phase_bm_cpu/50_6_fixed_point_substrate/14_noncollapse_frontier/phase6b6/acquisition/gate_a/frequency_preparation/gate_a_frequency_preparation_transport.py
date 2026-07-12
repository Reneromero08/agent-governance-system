#!/usr/bin/env python3
"""One-shot SSH/SCP transport for Gate A frequency preparation qualification.

Nothing executes at import time.  The state machine is dependency-injected for
zero-contact qualification.  A durable claim outside cleanup roots consumes the
authority before any target-side transaction can begin.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gate_a_frequency_preparation_authority as authority
import gate_a_frequency_preparation_bundle as target_bundle

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes"]
SCP_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes"]


class TransportError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise TransportError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(path: Path, data: bytes, *, exclusive: bool = True) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            require(written > 0, f"short durable write: {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(path.parent)


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    write_durable(path, canonical_line(value), exclusive=exclusive)


@dataclass(frozen=True)
class TransportRequest:
    permit: authority.PreparationPermit
    authority_path: Path
    authority_bytes: bytes
    manifest: dict[str, Any]
    manifest_bytes: bytes
    deployment_archive: bytes
    local_evidence_root: Path
    source_review_binding: dict[str, Any]


Runner = Callable[..., subprocess.CompletedProcess[Any]]


def default_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(argv, check=False, **kwargs)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class CommandLedger:
    def __init__(self, path: Path, runner: Runner):
        self.path = path
        self.runner = runner
        self.sequence = 0

    def run(self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 120) -> subprocess.CompletedProcess[Any]:
        self.sequence += 1
        start = time.time_ns()
        timed_out = False
        try:
            result = self.runner(
                argv,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            result = subprocess.CompletedProcess(argv, 124, stdout=exc.stdout or b"", stderr=exc.stderr or b"")
        end = time.time_ns()
        stdout = result.stdout if isinstance(result.stdout, bytes) else _text(result.stdout).encode("utf-8")
        stderr = result.stderr if isinstance(result.stderr, bytes) else _text(result.stderr).encode("utf-8")
        record = {
            "sequence": self.sequence,
            "argv": argv,
            "start_time_ns": start,
            "end_time_ns": end,
            "timeout_s": timeout,
            "timed_out": timed_out,
            "returncode": int(result.returncode),
            "stdin_size": 0 if input_bytes is None else len(input_bytes),
            "stdin_sha256": sha256_bytes(b"" if input_bytes is None else input_bytes),
            "stdout_size": len(stdout),
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_size": len(stderr),
            "stderr_sha256": sha256_bytes(stderr),
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        with self.path.open("ab") as stream:
            stream.write(canonical_line(record))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.path.parent)
        if timed_out:
            raise TransportError(f"command timed out: {argv[0]}")
        return result


def ssh_argv(target: str, *remote: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, target, *remote]


def scp_to_argv(local: Path, target: str, remote: str) -> list[str]:
    return ["scp", *SCP_OPTIONS, str(local), f"{target}:{remote}"]


def scp_from_argv(target: str, remote: str, local: Path) -> list[str]:
    return ["scp", *SCP_OPTIONS, f"{target}:{remote}", str(local)]


def _remote_python(script: str, *args: str) -> list[str]:
    return ["python3", "-c", script, *args]


PREFLIGHT_SCRIPT = r'''
import json, os, pathlib, sys
paths=[pathlib.Path(x) for x in sys.argv[1:]]
state=[]
for p in paths:
    try:
        os.lstat(p)
    except FileNotFoundError:
        state.append({"path":str(p),"state":"absent"})
    except OSError as exc:
        print(json.dumps({"status":"unobservable","path":str(p),"error":str(exc)},sort_keys=True)); raise SystemExit(2)
    else:
        print(json.dumps({"status":"collision","path":str(p)},sort_keys=True)); raise SystemExit(3)
print(json.dumps({"status":"all_absent","paths":state},sort_keys=True))
'''.strip()

CLAIM_SCRIPT = r'''
import base64, json, os, pathlib, sys
root=pathlib.Path(sys.argv[1]); data=base64.b64decode(sys.argv[2].encode("ascii"),validate=True)
root.mkdir(mode=0o700,parents=False,exist_ok=False)
p=root/"CLAIM.json"
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),0o600)
try:
 v=memoryview(data)
 while v:
  n=os.write(fd,v)
  if n<=0: raise OSError("short claim write")
  v=v[n:]
 os.fsync(fd)
finally: os.close(fd)
d=os.open(root,os.O_RDONLY)
try: os.fsync(d)
finally: os.close(d)
print(json.dumps({"status":"claim_created"},sort_keys=True))
'''.strip()

EXTRACT_SCRIPT = r'''
import os, pathlib, tarfile, sys
archive=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
root.mkdir(mode=0o700,parents=False,exist_ok=False)
with tarfile.open(archive,"r:") as tf:
 for member in tf.getmembers():
  p=pathlib.PurePosixPath(member.name)
  if member.isdir() or member.issym() or member.islnk() or member.isdev() or p.is_absolute() or ".." in p.parts: raise SystemExit(4)
  if not member.isfile(): raise SystemExit(4)
  dest=root.joinpath(*p.parts); dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
  src=tf.extractfile(member)
  if src is None: raise SystemExit(4)
  data=src.read(); fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0),0o600)
  try:
   v=memoryview(data)
   while v:
    n=os.write(fd,v)
    if n<=0: raise OSError("short extract write")
    v=v[n:]
   os.fsync(fd)
  finally: os.close(fd)
print("EXTRACTED")
'''.strip()

ARCHIVE_SCRIPT = r'''
import pathlib, tarfile, sys
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
if not root.is_dir(): raise SystemExit(3)
with tarfile.open(out,"w",format=tarfile.PAX_FORMAT) as tf:
 for p in sorted(root.rglob("*")):
  if p.is_dir() and not p.is_symlink(): continue
  if not p.is_file() or p.is_symlink(): raise SystemExit(4)
  tf.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print("ARCHIVED")
'''.strip()

CLEANUP_SCRIPT = r'''
import json, pathlib, shutil, sys
for raw in sys.argv[1:]:
 p=pathlib.Path(raw)
 if p.is_symlink(): raise SystemExit(4)
 if p.is_dir(): shutil.rmtree(p)
 elif p.exists(): p.unlink()
print(json.dumps({"status":"cleanup_complete"},sort_keys=True))
'''.strip()

PROCESS_SCRIPT = r'''
import json, os, subprocess
p=subprocess.run(["ps","-ww","-eo","pid=,comm=,args="],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
text=p.stdout.decode("utf-8","strict")
tokens=["gate_a_"+"frequency_"+"preparation_"+"target.py","catcas_"+"phase6b6_"+"gate_a_"+"freqprep_"]
me=os.getpid(); hits=[]
for line in text.splitlines():
 parts=line.split(None,1)
 if not parts: continue
 try: pid=int(parts[0])
 except ValueError: raise SystemExit(6)
 if pid==me or "ps -ww -eo" in line: continue
 if any(token in line for token in tokens): hits.append(line)
print(json.dumps({"status":"complete","returncode":p.returncode,"hits":hits,"stdout_sha256":__import__("hashlib").sha256(p.stdout).hexdigest(),"stderr_sha256":__import__("hashlib").sha256(p.stderr).hexdigest()},sort_keys=True))
raise SystemExit(0 if p.returncode==0 and not hits else 5)
'''.strip()

ABSENCE_SCRIPT = r'''
import json, os, pathlib, sys
states=[]
for raw in sys.argv[1:]:
 p=pathlib.Path(raw)
 try: os.lstat(p)
 except FileNotFoundError: states.append({"path":raw,"state":"absent"})
 except OSError as exc: print(json.dumps({"status":"unobservable","path":raw,"error":str(exc)},sort_keys=True)); raise SystemExit(2)
 else: print(json.dumps({"status":"present","path":raw},sort_keys=True)); raise SystemExit(3)
print(json.dumps({"status":"all_absent","states":states},sort_keys=True))
'''.strip()


def claim_value(request: TransportRequest) -> dict[str, Any]:
    permit = authority.require_permit(request.permit)
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_CLAIM_V1",
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "bundle_sha256": permit.bundle_sha256,
        "maximum_transaction_count": 1,
        "automatic_retry": False,
    }


def parse_process_receipt(stdout: bytes | str) -> dict[str, Any]:
    value = json.loads(_text(stdout))
    require(isinstance(value, dict) and value.get("status") == "complete", "process receipt malformed")
    require(value.get("returncode") == 0 and value.get("hits") == [], "target writer still active")
    return value


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), "unsafe evidence archive member")
            require(not path.is_absolute() and ".." not in path.parts, "unsafe evidence archive path")
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            require(stream is not None, "evidence archive member unreadable")
            write_durable(target, stream.read(), exclusive=True)


def validate_evidence_root(root: Path) -> dict[str, Any]:
    inventory_path = root / "FINAL_INVENTORY.json"
    require(inventory_path.is_file() and not inventory_path.is_symlink(), "target final inventory missing")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    require(set(inventory) == {"schema_id", "files"}, "target inventory key set mismatch")
    require(inventory["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_EVIDENCE_INVENTORY_V1", "target inventory schema mismatch")
    expected = {entry["path"]: entry for entry in inventory["files"]}
    observed: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid local evidence path: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "FINAL_INVENTORY.json":
            continue
        data = path.read_bytes()
        observed[relative] = {"path": relative, "size": len(data), "sha256": sha256_bytes(data)}
    require(observed == expected, "target evidence inventory mismatch")
    return inventory


def final_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid host evidence path: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "FINAL_HOST_INVENTORY.json":
            continue
        data = path.read_bytes()
        files.append({"path": relative, "size": len(data), "sha256": sha256_bytes(data)})
    return {"schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_HOST_INVENTORY_V1", "files": files}


def run_transport(request: TransportRequest, *, runner: Runner = default_runner) -> dict[str, Any]:
    permit = authority.require_permit(request.permit)
    root = request.local_evidence_root
    require(not root.exists(), "local evidence root must be absent")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    ledger = CommandLedger(root / "HOST_COMMANDS.jsonl", runner)
    write_durable(root / "AUTHORITY_ARTIFACT.json", request.authority_bytes)
    write_durable(root / "BUNDLE_MANIFEST.json", request.manifest_bytes)
    write_json(root / "SOURCE_REVIEW_BINDING.json", request.source_review_binding)
    archive_path = root / "DEPLOYMENT_BUNDLE.tar"
    write_durable(archive_path, request.deployment_archive)

    target = permit.target
    remote_paths = [
        permit.remote_claim_root,
        permit.remote_execution_root,
        permit.remote_stage_archive,
        permit.remote_evidence_archive,
    ]
    result: dict[str, Any] = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_TRANSPORT_RESULT_V1",
        "authority_id": permit.authority_id,
        "authority_sha256": permit.authority_sha256,
        "target": target,
        "target_contacts": 0,
        "transaction_invocations": 0,
        "retry_count": 0,
        "automatic_retry": False,
        "claim_retained": False,
        "copy_back_verified": False,
        "cleanup_complete": False,
        "remote_writer_absent": False,
        "status": None,
        "failure": None,
    }
    mutated = False
    writer_absent = False
    copy_verified = False

    def contact(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        result["target_contacts"] += 1
        return ledger.run(argv, **kwargs)

    try:
        preflight = contact(ssh_argv(target, *_remote_python(PREFLIGHT_SCRIPT, *remote_paths)))
        require(preflight.returncode == 0, "remote namespace preflight failed")

        claim_bytes = canonical_line(claim_value(request))
        claim_b64 = base64.b64encode(claim_bytes).decode("ascii")
        claim = contact(ssh_argv(target, *_remote_python(CLAIM_SCRIPT, permit.remote_claim_root, claim_b64)))
        require(claim.returncode == 0, "durable claim creation failed")
        mutated = True
        result["claim_retained"] = True

        stage = contact(scp_to_argv(archive_path, target, permit.remote_stage_archive))
        require(stage.returncode == 0, "bundle staging failed")
        extract = contact(ssh_argv(target, *_remote_python(EXTRACT_SCRIPT, permit.remote_stage_archive, permit.remote_execution_root)))
        require(extract.returncode == 0, "bundle extraction failed")

        authority_local = root / "AUTHORITY.upload.json"
        write_durable(authority_local, request.authority_bytes)
        authority_remote = f"{permit.remote_execution_root}/AUTHORITY.json"
        upload_auth = contact(scp_to_argv(authority_local, target, authority_remote))
        require(upload_auth.returncode == 0, "authority staging failed")

        manifest_remote = f"{permit.remote_execution_root}/{target_bundle.MANIFEST_FILENAME}"
        runner_remote = f"{permit.remote_execution_root}/gate_a_frequency_preparation_target.py"
        result["transaction_invocations"] = 1
        invoke = contact(
            ssh_argv(
                target,
                "python3",
                "-B",
                runner_remote,
                "--bundle-root",
                permit.remote_execution_root,
                "--manifest",
                manifest_remote,
                "--authority",
                authority_remote,
                "--authority-sha256",
                permit.authority_sha256,
                "--reviewed-source-commit",
                permit.reviewed_source_commit,
                "--independent-review-id",
                str(permit.independent_review_id),
                "--claim-root",
                permit.remote_claim_root,
                "--output-root",
                permit.remote_output_root,
            ),
            timeout=90,
        )
        require(invoke.returncode in {0, 1}, "target runner returned invalid status")

        scan = contact(ssh_argv(target, *_remote_python(PROCESS_SCRIPT)))
        process_receipt = parse_process_receipt(scan.stdout)
        write_json(root / "POST_RUNTIME_PROCESS_RECEIPT.json", process_receipt)
        writer_absent = True
        result["remote_writer_absent"] = True

        archive = contact(ssh_argv(target, *_remote_python(ARCHIVE_SCRIPT, permit.remote_output_root, permit.remote_evidence_archive)))
        require(archive.returncode == 0, "target evidence archiving failed")
        local_archive = root / "TARGET_EVIDENCE.tar"
        copied = contact(scp_from_argv(target, permit.remote_evidence_archive, local_archive))
        require(copied.returncode == 0, "target evidence copy-back failed")
        extracted = root / "TARGET_OUTPUT"
        safe_extract_archive(local_archive, extracted)
        target_inventory = validate_evidence_root(extracted)
        write_json(root / "COPY_BACK_RECEIPT.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_COPY_BACK_V1",
            "archive_sha256": sha256_bytes(local_archive.read_bytes()),
            "target_inventory_sha256": sha256_bytes(canonical_bytes(target_inventory)),
            "verified": True,
        })
        copy_verified = True
        result["copy_back_verified"] = True

        cleanup = contact(ssh_argv(target, *_remote_python(CLEANUP_SCRIPT, permit.remote_execution_root, permit.remote_stage_archive, permit.remote_evidence_archive)))
        require(cleanup.returncode == 0, "remote cleanup failed")
        absence = contact(ssh_argv(target, *_remote_python(ABSENCE_SCRIPT, permit.remote_execution_root, permit.remote_stage_archive, permit.remote_evidence_archive)))
        require(absence.returncode == 0, "post-cleanup absence failed")
        post = contact(ssh_argv(target, *_remote_python(PROCESS_SCRIPT)))
        post_receipt = parse_process_receipt(post.stdout)
        write_json(root / "POST_CLEANUP_PROCESS_RECEIPT.json", post_receipt)
        write_json(root / "CLEANUP_RECEIPT.json", {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_CLEANUP_V1",
            "verified_copy_back": True,
            "claim_retained": True,
            "transient_paths_absent": True,
        })
        result["cleanup_complete"] = True
        target_result = json.loads((extracted / "RESULT.json").read_text(encoding="utf-8"))
        result["target_result"] = target_result
        result["status"] = "SUCCESS" if target_result["status"] == "QUALIFIED_PREPARATION_AND_RESTORATION" else "TARGET_FAILED_CLOSED"
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "FAILED_CLOSED_TRANSPORT"
        if mutated and not writer_absent:
            try:
                scan = contact(ssh_argv(target, *_remote_python(PROCESS_SCRIPT)))
                process_receipt = parse_process_receipt(scan.stdout)
                write_json(root / "FAILURE_PROCESS_RECEIPT.json", process_receipt)
                writer_absent = True
                result["remote_writer_absent"] = True
            except BaseException as scan_exc:
                write_json(root / "PROCESS_CUSTODY_FAILURE.json", {"failure": f"{type(scan_exc).__name__}: {scan_exc}"})
        if mutated and writer_absent and not copy_verified:
            try:
                archive = contact(ssh_argv(target, *_remote_python(ARCHIVE_SCRIPT, permit.remote_output_root, permit.remote_evidence_archive)))
                if archive.returncode == 0:
                    local_archive = root / "TARGET_EVIDENCE.partial.tar"
                    copied = contact(scp_from_argv(target, permit.remote_evidence_archive, local_archive))
                    if copied.returncode == 0:
                        extracted = root / "TARGET_OUTPUT_PARTIAL"
                        safe_extract_archive(local_archive, extracted)
                        validate_evidence_root(extracted)
                        copy_verified = True
                        result["copy_back_verified"] = True
            except BaseException as recovery_exc:
                write_json(root / "COPY_BACK_RECOVERY_FAILURE.json", {"failure": f"{type(recovery_exc).__name__}: {recovery_exc}"})
        if mutated and writer_absent and copy_verified:
            try:
                cleanup = contact(ssh_argv(target, *_remote_python(CLEANUP_SCRIPT, permit.remote_execution_root, permit.remote_stage_archive, permit.remote_evidence_archive)))
                absence = contact(ssh_argv(target, *_remote_python(ABSENCE_SCRIPT, permit.remote_execution_root, permit.remote_stage_archive, permit.remote_evidence_archive)))
                result["cleanup_complete"] = cleanup.returncode == 0 and absence.returncode == 0
            except BaseException as cleanup_exc:
                write_json(root / "CLEANUP_FAILURE.json", {"failure": f"{type(cleanup_exc).__name__}: {cleanup_exc}"})

    require(result["transaction_invocations"] <= 1, "transaction invoked more than once")
    require(result["retry_count"] == 0 and result["automatic_retry"] is False, "retry boundary violated")
    write_json(root / "TRANSPORT_RESULT.json", result)
    write_json(root / "FINAL_HOST_INVENTORY.json", final_inventory(root))
    return result

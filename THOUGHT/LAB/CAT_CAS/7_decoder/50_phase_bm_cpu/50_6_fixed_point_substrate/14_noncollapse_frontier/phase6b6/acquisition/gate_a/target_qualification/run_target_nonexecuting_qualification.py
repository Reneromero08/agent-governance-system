#!/usr/bin/env python3
"""Host orchestrator for the Gate A target non-executing qualification.

This driver deploys the exact reviewed deterministic Gate A execution bundle to
the authorized Phenom target, validates it in an isolated Git-free namespace,
runs the target runner's no-drive qualification exactly once, preserves complete
evidence, verifies copy-back, and cleans the target namespace.

Hard rules enforced here:
  - Git is never used on the target.
  - No package installation, no hardware driving, no probing, no MSR access.
  - The qualification command runs at most once, with a 180s timeout.
  - Only positively established path absence permits namespace creation.
  - Cleanup runs only after copy-back is cryptographically verified.

It is a lab tool, not a packaged bundle payload. It performs no hardware drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GATE_A = HERE.parent
ADAPTER = GATE_A / "adapter"
PHASE6B6 = HERE.parents[2]
REPO_ROOT = PHASE6B6.parents[7]
EVIDENCE_PARENT = PHASE6B6 / "evidence"
HISTORICAL_EVID = EVIDENCE_PARENT / "gate_a_target_nonexecuting_qualification_6f243b1a_bundle_abc9e50a"
HISTORICAL_AUTHORIZATION = HERE / "GATE_A_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json"
PREDECESSOR_IDENTITY = PHASE6B6 / "evidence" / "nonhardware_qualification_3c6a5dd3_subject_d351a62f" / "target" / "logs" / "016_target_identity.stdout.txt"

SSH_TARGET = "root@192.168.137.100"
TARGET_HOSTNAME = "catcas"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes"]
SCP_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes"]

# These values are intentionally unset until a new, closed, exact owner
# authorization artifact has been validated.  There is no default that can
# accidentally reuse the consumed historical authority or evidence namespace.
EVID: Path
EXEC_ROOT: str
EVIDENCE_ROOT: str
TRANSFER_STAGE: str
EV_ARCHIVE: str
TP: str
QUAL_STDOUT: str
QUAL_STDERR: str
REPLACEMENT_AUTHORITY: dict[str, Any]
REPLACEMENT_AUTHORITY_PATH: Path
REPLACEMENT_AUTHORITY_SHA256: str
REPLACEMENT_AUTHORITY_GIT_BLOB_SHA1: str

EXPECTED_EXECUTION_BUNDLE = "abc9e50a517d764c553adc5096378992028b29a8f62480a9ae217ebbd5202bba"
EXPECTED_ARCHIVE = "04eaf73336f373865f4e837baca9ff4fe893d3b1b16dd8b8288af1259ff96f9c"
EXPECTED_MANIFEST_FILE = "ccb7866db67170083cb00d546c334b61772c8ef909131ec9c62ed21115facc94"
PREDECESSOR_ID_SHA = "10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4"
EXPECTED_HOSTNAME = "catcas"
EXPECTED_ARCH = "x86_64"
EXPECTED_CPU = "AMD Phenom(tm) II X6 1090T Processor"

LOCAL_HOST = socket.gethostname()

REPLACEMENT_AUTHORITY_SCHEMA_ID = (
    "CAT_CAS_PHASE6B6_GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION_V1"
)
REPLACEMENT_AUTHORITY_DECISION = (
    "AUTHORIZED_FOR_ONE_REPLACEMENT_GATE_A_TARGET_NONEXECUTING_QUALIFICATION"
)
HISTORICAL_EVIDENCE_REL = HISTORICAL_EVID.relative_to(REPO_ROOT).as_posix()
HISTORICAL_AUTHORIZATION_REL = HISTORICAL_AUTHORIZATION.relative_to(REPO_ROOT).as_posix()

REPLACEMENT_AUTHORITY_KEYS = {
    "schema_id",
    "decision",
    "project_owner",
    "owner_instruction",
    "authority_id",
    "authorized_source_commit",
    "historical_authorization_path",
    "historical_authority_consumed",
    "historical_evidence_dir",
    "local_evidence_dir",
    "ssh_target",
    "expected_hostname",
    "expected_architecture",
    "expected_cpu_model",
    "remote_execution_root",
    "remote_evidence_root",
    "remote_transfer_stage",
    "remote_evidence_archive",
    "remote_temp_prefix",
    "execution_bundle_sha256",
    "deterministic_archive_sha256",
    "bundle_manifest_sha256",
    "maximum_target_qualification_executions",
    "automatic_retry",
    "replacement_qualification_authorized",
    "ssh_authorized",
    "copy_authorized",
    "target_filesystem_staging_authorized",
    "compile_validate_only_authorized",
    "no_drive_runner_authorized",
    "probe_authorized",
    "engineering_smoke_authorized",
    "hardware_execution_authorized",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "execution_authority_artifact_creation_authorized",
}

NARROW_TRUE_FIELDS = (
    "replacement_qualification_authorized",
    "ssh_authorized",
    "copy_authorized",
    "target_filesystem_staging_authorized",
    "compile_validate_only_authorized",
    "no_drive_runner_authorized",
)

DOWNSTREAM_FALSE_FIELDS = (
    "probe_authorized",
    "engineering_smoke_authorized",
    "hardware_execution_authorized",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "execution_authority_artifact_creation_authorized",
)


class QualError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one replacement Gate A target non-executing qualification only "
            "under a new exact project-owner authorization artifact."
        )
    )
    parser.add_argument(
        "--replacement-authorization",
        required=True,
        type=Path,
        help=(
            "Path to GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json; "
            "the consumed historical authorization is rejected"
        ),
    )
    return parser.parse_args(argv)


def current_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise QualError(f"cannot resolve exact source commit: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def load_replacement_authority(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved == HISTORICAL_AUTHORIZATION.resolve():
        raise QualError("the historical target-qualification authority is consumed and cannot be reused")
    if resolved.name != "GATE_A_REPLACEMENT_TARGET_NONEXECUTING_QUALIFICATION_AUTHORIZATION.json":
        raise QualError("replacement authorization must use the exact required artifact name")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualError(f"cannot load replacement authorization: {exc}") from exc
    if not isinstance(value, dict):
        raise QualError("replacement authorization must be a JSON object")
    return value


def validate_replacement_authority(
    authority: dict[str, Any],
    authority_path: Path,
    *,
    source_commit: str,
) -> Path:
    """Validate closed replacement authority before any SSH or SCP is possible."""
    if set(authority) != REPLACEMENT_AUTHORITY_KEYS:
        missing = sorted(REPLACEMENT_AUTHORITY_KEYS - set(authority))
        extra = sorted(set(authority) - REPLACEMENT_AUTHORITY_KEYS)
        raise QualError(f"replacement authorization key closure failed: missing={missing}, extra={extra}")
    expected_scalars = {
        "schema_id": REPLACEMENT_AUTHORITY_SCHEMA_ID,
        "decision": REPLACEMENT_AUTHORITY_DECISION,
        "project_owner": "Raúl Romero",
        "authorized_source_commit": source_commit,
        "historical_authorization_path": HISTORICAL_AUTHORIZATION_REL,
        "historical_evidence_dir": HISTORICAL_EVIDENCE_REL,
        "ssh_target": SSH_TARGET,
        "expected_hostname": EXPECTED_HOSTNAME,
        "expected_architecture": EXPECTED_ARCH,
        "expected_cpu_model": EXPECTED_CPU,
        "execution_bundle_sha256": EXPECTED_EXECUTION_BUNDLE,
        "deterministic_archive_sha256": EXPECTED_ARCHIVE,
        "bundle_manifest_sha256": EXPECTED_MANIFEST_FILE,
        "maximum_target_qualification_executions": 1,
        "automatic_retry": False,
        "historical_authority_consumed": True,
    }
    for key, expected in expected_scalars.items():
        if authority[key] != expected:
            raise QualError(f"replacement authorization {key} mismatch")
    if not isinstance(authority["owner_instruction"], str) or not authority["owner_instruction"].strip():
        raise QualError("replacement authorization owner_instruction must be nonempty")
    for key in NARROW_TRUE_FIELDS:
        if authority[key] is not True:
            raise QualError(f"replacement authorization {key} must be true")
    for key in DOWNSTREAM_FALSE_FIELDS:
        if authority[key] is not False:
            raise QualError(f"replacement authorization {key} must be false")

    authority_id = authority["authority_id"]
    if not isinstance(authority_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{7,63}", authority_id) is None:
        raise QualError("replacement authority_id must be a safe 8-64 character lowercase identifier")
    if authority_id in HISTORICAL_EVID.name:
        raise QualError("replacement authority_id must not reuse the historical namespace")

    expected_local_rel = (
        EVIDENCE_PARENT.relative_to(REPO_ROOT).as_posix()
        + f"/gate_a_target_nonexecuting_qualification_replacement_{authority_id}"
    )
    if authority["local_evidence_dir"] != expected_local_rel:
        raise QualError("replacement local evidence namespace is not exactly authority-bound")
    evidence_path = (REPO_ROOT / expected_local_rel).resolve()
    if evidence_path == HISTORICAL_EVID.resolve():
        raise QualError("historical evidence namespace cannot be reused")
    if evidence_path.parent != EVIDENCE_PARENT.resolve():
        raise QualError("replacement evidence namespace escaped the evidence parent")

    expected_remote = {
        "remote_execution_root": f"/root/catcas_phase6b6_gate_a_target_nonexec_{authority_id}",
        "remote_evidence_root": f"/root/catcas_phase6b6_gate_a_target_nonexec_{authority_id}/evidence",
        "remote_transfer_stage": f"/tmp/catcas_gate_a_bundle_{authority_id}.deploy.tar",
        "remote_evidence_archive": f"/tmp/catcas_gate_a_evidence_{authority_id}.tar",
        "remote_temp_prefix": f"/tmp/catcas_gate_a_tq_{authority_id}_",
    }
    for key, expected in expected_remote.items():
        if authority[key] != expected:
            raise QualError(f"replacement authorization {key} is not exactly authority-bound")

    if authority_path.resolve() == HISTORICAL_AUTHORIZATION.resolve():
        raise QualError("historical authorization cannot be reused")
    return evidence_path


def ensure_new_evidence_namespace(path: Path) -> None:
    if not EVIDENCE_PARENT.is_dir() or EVIDENCE_PARENT.is_symlink():
        raise QualError(f"evidence parent must be an existing real directory: {EVIDENCE_PARENT}")
    if path.exists() or path.is_symlink():
        raise QualError(f"replacement evidence namespace must be absent: {path}")


def validate_replacement_authority_custody(authority_path: Path, source_commit: str) -> str:
    """Bind the owner artifact and runner bytes to the exact authorized commit."""
    resolved = authority_path.resolve()
    try:
        authority_rel = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise QualError("replacement authorization must be committed inside the repository") from exc
    runner_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    for rel in (authority_rel, runner_rel):
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}:{rel}"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if tracked.returncode != 0:
            raise QualError(f"authorized source commit does not contain exact required path: {rel}")
    clean = subprocess.run(
        ["git", "diff", "--exit-code", source_commit, "--", authority_rel, runner_rel],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if clean.returncode != 0:
        raise QualError("replacement authorization or runner differs from the authorized source commit")
    blob = subprocess.run(
        ["git", "rev-parse", f"{source_commit}:{authority_rel}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if blob.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", blob.stdout.strip()) is None:
        raise QualError("cannot bind replacement authorization Git blob")
    return blob.stdout.strip()


def configure_runtime(authority: dict[str, Any], authority_path: Path) -> None:
    global EVID, EXEC_ROOT, EVIDENCE_ROOT, TRANSFER_STAGE, EV_ARCHIVE, TP
    global QUAL_STDOUT, QUAL_STDERR, REPLACEMENT_AUTHORITY, REPLACEMENT_AUTHORITY_PATH
    global REPLACEMENT_AUTHORITY_SHA256, REPLACEMENT_AUTHORITY_GIT_BLOB_SHA1

    source_commit = current_head()
    evidence_path = validate_replacement_authority(
        authority,
        authority_path,
        source_commit=source_commit,
    )
    authority_blob = validate_replacement_authority_custody(authority_path, source_commit)
    ensure_new_evidence_namespace(evidence_path)
    EVID = evidence_path
    EXEC_ROOT = authority["remote_execution_root"]
    EVIDENCE_ROOT = authority["remote_evidence_root"]
    TRANSFER_STAGE = authority["remote_transfer_stage"]
    EV_ARCHIVE = authority["remote_evidence_archive"]
    TP = authority["remote_temp_prefix"]
    QUAL_STDOUT = TP + "qual.stdout"
    QUAL_STDERR = TP + "qual.stderr"
    REPLACEMENT_AUTHORITY = authority
    REPLACEMENT_AUTHORITY_PATH = authority_path.resolve()
    REPLACEMENT_AUTHORITY_SHA256 = sha256_file(REPLACEMENT_AUTHORITY_PATH)
    REPLACEMENT_AUTHORITY_GIT_BLOB_SHA1 = authority_blob


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class Recorder:
    def __init__(self, evid: Path) -> None:
        self.evid = evid
        self.seq = 0
        self.commands_path = evid / "COMMANDS.jsonl"
        self.commands_path.write_text("", encoding="utf-8")

    def _write(self, subdir: str, name: str, suffix: str, data: bytes) -> tuple[str, str]:
        d = self.evid / subdir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.seq:03d}_{name}.{suffix}"
        p.write_bytes(data)
        return str(p.relative_to(self.evid).as_posix()), sha256_bytes(data)

    def record(self, *, subdir: str, name: str, argv: list[str], hostname: str, cwd: str,
               started: float, finished: float, exit_code: int, stdout: bytes, stderr: bytes,
               environment: dict[str, str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.seq += 1
        out_path, out_sha = self._write(subdir, name, "stdout.txt", stdout)
        err_path, err_sha = self._write(subdir, name, "stderr.txt", stderr)
        entry = {
            "sequence": self.seq,
            "environment": environment,
            "argv": argv,
            "working_directory": cwd,
            "started_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
            "elapsed_seconds": round(finished - started, 3),
            "exit_code": exit_code,
            "stdout_path": out_path,
            "stdout_sha256": out_sha,
            "stderr_path": err_path,
            "stderr_sha256": err_sha,
            "hostname": hostname,
        }
        if extra:
            entry.update(extra)
        with self.commands_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


REC: Recorder


def run_local(name: str, argv: list[str], *, subdir: str = "local", cwd: Path | None = None,
              env_over: dict[str, str] | None = None, timeout: int | None = None, check: bool = True) -> dict[str, Any]:
    env = os.environ.copy()
    if env_over:
        env.update(env_over)
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
        rc = proc.returncode
        out, err = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or b""
        err = (exc.stderr or b"") + b"\n[timeout]"
    finished = time.time()
    entry = REC.record(subdir=subdir, name=name, argv=argv, hostname=LOCAL_HOST,
                       cwd=str(cwd) if cwd else os.getcwd(), started=started, finished=finished,
                       exit_code=rc, stdout=out, stderr=err, environment=env_over or {})
    entry["_stdout"] = out
    entry["_stderr"] = err
    if check and rc != 0:
        raise QualError(f"local command failed: {name} rc={rc}: {err.decode('utf-8','replace')[:400]}")
    return entry


def run_ssh(name: str, remote_cmd: str, *, subdir: str = "target", timeout: int = 120, check: bool = True) -> dict[str, Any]:
    argv = ["ssh", *SSH_OPTS, SSH_TARGET, remote_cmd]
    started = time.time()
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or b""
        err = (exc.stderr or b"") + b"\n[timeout]"
    finished = time.time()
    entry = REC.record(subdir=subdir, name=name, argv=argv, hostname=TARGET_HOSTNAME, cwd="(remote)",
                       started=started, finished=finished, exit_code=rc, stdout=out, stderr=err, environment={})
    entry["_stdout"] = out
    entry["_stderr"] = err
    if check and rc != 0:
        raise QualError(f"ssh command failed: {name} rc={rc}: {err.decode('utf-8','replace')[:400]}")
    return entry


def run_ssh_py(name: str, script: str, remote_env: dict[str, str] | None = None, *, subdir: str = "target",
               timeout: int = 120, check: bool = True) -> dict[str, Any]:
    prefix = ""
    if remote_env:
        prefix = " ".join(f"{k}='{v}'" for k, v in remote_env.items()) + " "
    remote_cmd = prefix + "python3 -"
    argv = ["ssh", *SSH_OPTS, SSH_TARGET, remote_cmd]
    script_bytes = script.encode("utf-8")
    # persist the transmitted script as an artifact
    d = REC.evid / subdir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(argv, input=script_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or b""
        err = (exc.stderr or b"") + b"\n[timeout]"
    finished = time.time()
    script_path = d / f"{REC.seq + 1:03d}_{name}.script.py"
    script_path.write_bytes(script_bytes)
    entry = REC.record(subdir=subdir, name=name, argv=argv, hostname=TARGET_HOSTNAME, cwd="(remote)",
                       started=started, finished=finished, exit_code=rc, stdout=out, stderr=err,
                       environment=remote_env or {},
                       extra={"stdin_script_path": str(script_path.relative_to(REC.evid).as_posix()),
                              "stdin_script_sha256": sha256_bytes(script_bytes)})
    entry["_stdout"] = out
    entry["_stderr"] = err
    if check and rc != 0:
        raise QualError(f"ssh python failed: {name} rc={rc}: {err.decode('utf-8','replace')[:400]}")
    return entry


def _scp(name: str, argv: list[str], *, subdir: str, direction: str) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = exc.stdout or b""
        err = (exc.stderr or b"") + b"\n[timeout]"
    except OSError as exc:
        rc = 127
        out = b""
        err = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
    finished = time.time()
    entry = REC.record(subdir=subdir, name=name, argv=argv, hostname=LOCAL_HOST, cwd=os.getcwd(),
                       started=started, finished=finished, exit_code=rc,
                       stdout=out, stderr=err, environment={})
    entry["_stdout"] = out
    entry["_stderr"] = err
    if rc != 0:
        raise QualError(f"scp {direction} failed: {name} rc={rc}: {err.decode('utf-8','replace')[:400]}")
    return entry


def scp_to(name: str, local: Path, remote: str) -> dict[str, Any]:
    argv = ["scp", *SCP_OPTS, str(local), f"{SSH_TARGET}:{remote}"]
    return _scp(name, argv, subdir="transfer", direction="upload")


def scp_from(name: str, remote: str, local: Path, *, subdir: str = "copy_back") -> dict[str, Any]:
    argv = ["scp", *SCP_OPTS, f"{SSH_TARGET}:{remote}", str(local)]
    return _scp(name, argv, subdir=subdir, direction="download")


# ---- target-side python programs (Git-free) ----

TREE_LIB = r'''
import os, json, hashlib, stat as st
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(65536),b''):
            h.update(c)
    return h.hexdigest()
def tree(root):
    out=[]
    for dp,dns,fns in os.walk(root):
        dns.sort()
        for n in sorted(dns):
            fp=os.path.join(dp,n); rel=os.path.relpath(fp,root).replace(os.sep,'/')
            lm=os.lstat(fp)
            out.append({"path":rel,"type":"symlink" if st.S_ISLNK(lm.st_mode) else "dir","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
        for n in sorted(fns):
            fp=os.path.join(dp,n); rel=os.path.relpath(fp,root).replace(os.sep,'/')
            lm=os.lstat(fp)
            if st.S_ISLNK(lm.st_mode):
                out.append({"path":rel,"type":"symlink","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
            elif st.S_ISREG(lm.st_mode):
                out.append({"path":rel,"type":"file","mode":oct(lm.st_mode & 0o7777),"size":lm.st_size,"sha256":sha256_file(fp)})
            else:
                out.append({"path":rel,"type":"special","mode":oct(lm.st_mode & 0o7777),"size":0,"sha256":""})
    out.sort(key=lambda e:e["path"])
    return out
'''


def target_identity_script() -> str:
    return r'''
import os, json, subprocess, platform
def first_line(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode('utf-8','replace').splitlines()[0]
    except Exception as e:
        return "ERR:%s"%e
cpu=""
ncpu=0
with open('/proc/cpuinfo') as f:
    for line in f:
        if line.startswith('model name'):
            cpu=line.split(':',1)[1].strip()
        if line.startswith('processor'):
            ncpu+=1
ident={
 "hostname": platform.node(),
 "architecture": platform.machine(),
 "cpu_model": cpu,
 "cpu_count": ncpu,
 "kernel": first_line(['uname','-a']),
 "cc_version_first_line": first_line(['cc','--version']),
 "python_version": first_line(['python3','--version']),
}
text=json.dumps(ident, sort_keys=True, indent=2)
out=os.environ.get("OUT")
if out:
    with open(out,'w') as f: f.write(text+"\n")
print(text)
'''


def absence_script() -> str:
    return r'''
import os, json
def state(p):
    try:
        os.lstat(p); return "PRESENT"
    except FileNotFoundError:
        return "ABSENT"
    except OSError as e:
        return "UNOBSERVABLE:%s"%type(e).__name__
res={"execution_root":state(os.environ["ROOT"]),"transfer_stage":state(os.environ["STAGE"])}
print(json.dumps(res, sort_keys=True))
'''


def members_script() -> str:
    return r'''
import os, json, tarfile
stage=os.environ["STAGE"]
bad=[]; members=[]
with tarfile.open(stage,'r:') as t:
    seen=set(); lower=set()
    for m in t.getmembers():
        members.append({"name":m.name,"type":("file" if m.isreg() else "dir" if m.isdir() else "sym" if m.issym() else "lnk" if m.islnk() else "chr" if m.ischr() else "blk" if m.isblk() else "fifo" if m.isfifo() else "other"),"size":m.size,"mode":oct(m.mode)})
        n=m.name
        if n.startswith('/') or n.startswith('\\'): bad.append("absolute:"+n)
        if '..' in n.replace('\\','/').split('/'): bad.append("traversal:"+n)
        if n.strip()=='' : bad.append("empty")
        if n in seen: bad.append("duplicate:"+n)
        seen.add(n)
        if n.lower() in lower: bad.append("case_collision:"+n)
        lower.add(n.lower())
        if m.issym(): bad.append("symlink:"+n)
        if m.islnk(): bad.append("hardlink:"+n)
        if m.ischr() or m.isblk(): bad.append("device:"+n)
        if m.isfifo(): bad.append("fifo:"+n)
        if not (m.isreg() or m.isdir()): bad.append("special:"+n)
print(json.dumps({"members":members,"violations":bad}, sort_keys=True))
'''


def extract_script() -> str:
    return r'''
import os, json, tarfile
root=os.environ["ROOT"]; stage=os.environ["STAGE"]
os.umask(0o022)
if os.path.lexists(root):
    raise SystemExit("execution root already exists")
os.makedirs(root, mode=0o755)
with tarfile.open(stage,'r:') as t:
    t.extractall(root, filter='data')
print(json.dumps({"extracted_to":root,"status":"EXTRACTED"}, sort_keys=True))
'''


def custody_script(phase: str) -> str:
    return (r'''
import os, json, sys, hashlib, stat as st, pathlib
sys.dont_write_bytecode=True
root=os.environ["ROOT"]
root_p=pathlib.Path(root)
sys.path.insert(0, os.path.join(root,'adapter'))
''' + TREE_LIB + r'''
report={"phase":os.environ["PHASE"]}
# git absence
report["git_absent"]= not os.path.lexists(os.path.join(root,'.git'))
# authority artifact absence anywhere under root
auth=[]
for dp,dns,fns in os.walk(root):
    for n in fns:
        if n=='GATE_A_EXECUTION_AUTHORITY.json':
            auth.append(os.path.relpath(os.path.join(dp,n),root))
report["authority_artifact_absent"]= (len(auth)==0)
report["authority_artifact_hits"]=sorted(auth)
# strict git-free validation
import gate_a_target_bundle as tb
manifest=tb.load_manifest(root_p)
val=tb.validate_extracted_bundle(root_p, manifest, strict=True)
report["validation"]=val
# tree inventory
tv=tree(root)
report["tree"]=tv
report["tree_canonical_sha256"]=hashlib.sha256(json.dumps(tv, sort_keys=True, separators=(',',':')).encode()).hexdigest()
# generated-file forbidden check
forbidden_files=[e for e in tv if ('__pycache__' in e["path"].split('/')) or e["path"].endswith('.pyc') or e["path"].endswith('.pyo') or e["path"].endswith('gate_a_worker') or e["path"].endswith('gate_a_worker_asan') or e["path"].endswith('gate_a_worker_ubsan')]
report["forbidden_generated_files"]=[e["path"] for e in forbidden_files]
text=json.dumps(report, sort_keys=True)
out=os.environ.get("OUT")
if out:
    with open(out,'w') as f: f.write(text)
print(text)
''')


def process_script() -> str:
    return r'''
import os, json, subprocess, hashlib, base64
pats=["combined_pdn_runner","run_combined_campaign","explicit_slot_runtime","wrmsr","rdmsr","cpupower","turbostat","gate_a_worker"]
cmd=json.loads(os.environ.get("PROCESS_SCAN_COMMAND_JSON", '["ps","-eo","pid,comm,args"]'))
proc=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
raw_stdout=proc.stdout.decode('utf-8','replace')
raw_stderr=proc.stderr.decode('utf-8','replace')
receipt={
 "command":cmd,
 "command_sha256":hashlib.sha256(json.dumps(cmd,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest(),
 "return_code":proc.returncode,
 "stdout_sha256":hashlib.sha256(proc.stdout).hexdigest(),
 "stderr_sha256":hashlib.sha256(proc.stderr).hexdigest(),
 "raw_process_listing":raw_stdout,
 "raw_process_listing_base64":base64.b64encode(proc.stdout).decode('ascii'),
 "raw_process_listing_sha256":hashlib.sha256(proc.stdout).hexdigest(),
 "raw_process_stderr":raw_stderr,
 "raw_process_stderr_base64":base64.b64encode(proc.stderr).decode('ascii'),
 "ps_executed_successfully":False,
 "raw_process_listing_preserved":True,
 "forbidden_process_filter_evaluated":False,
 "forbidden_process_hits":[],
 "scan_complete":False,
}
def emit(exit_code):
    text=json.dumps(receipt, sort_keys=True)
    out_path=os.environ.get("OUT")
    if out_path:
        with open(out_path,'w') as f: f.write(text)
    print(text)
    raise SystemExit(exit_code)
if proc.returncode != 0:
    receipt["failure"]="ps returned nonzero"
    emit(70)
lines=raw_stdout.splitlines()
if not lines or "PID" not in lines[0].upper():
    receipt["failure"]="ps output missing expected header"
    emit(71)
hits=[]
for line in lines[1:]:
    for p in pats:
        if p in line and 'ps -eo' not in line:
            hits.append(line.strip())
receipt["forbidden_process_hits"]=hits
receipt["ps_executed_successfully"]=True
receipt["forbidden_process_filter_evaluated"]=True
receipt["scan_complete"]=True
text=json.dumps(receipt, sort_keys=True)
out_path=os.environ.get("OUT")
if out_path:
    with open(out_path,'w') as f: f.write(text)
print(text)
'''


def validate_process_scan(scan: dict[str, Any], context: str) -> None:
    import base64

    expected_command = ["ps", "-eo", "pid,comm,args"]
    checks = {
        "command": expected_command,
        "return_code": 0,
        "ps_executed_successfully": True,
        "raw_process_listing_preserved": True,
        "forbidden_process_filter_evaluated": True,
        "scan_complete": True,
    }
    for key, expected in checks.items():
        if scan.get(key) != expected:
            raise QualError(f"{context} process scan {key}={scan.get(key)!r} != {expected!r}")
    raw_stdout = scan.get("raw_process_listing")
    raw_stderr = scan.get("raw_process_stderr")
    if not isinstance(raw_stdout, str) or not isinstance(raw_stderr, str):
        raise QualError(f"{context} process scan did not preserve raw stdout/stderr")
    try:
        raw_stdout_bytes = base64.b64decode(scan.get("raw_process_listing_base64", ""), validate=True)
        raw_stderr_bytes = base64.b64decode(scan.get("raw_process_stderr_base64", ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise QualError(f"{context} process scan raw base64 invalid: {exc}") from exc
    if raw_stdout_bytes.decode("utf-8", "replace") != raw_stdout:
        raise QualError(f"{context} process scan raw stdout representation mismatch")
    if raw_stderr_bytes.decode("utf-8", "replace") != raw_stderr:
        raise QualError(f"{context} process scan raw stderr representation mismatch")
    expected_command_sha = sha256_bytes(canonical(expected_command))
    if scan.get("command_sha256") != expected_command_sha:
        raise QualError(f"{context} process scan command hash mismatch")
    if scan.get("stdout_sha256") != sha256_bytes(raw_stdout_bytes):
        raise QualError(f"{context} process scan stdout hash mismatch")
    if scan.get("raw_process_listing_sha256") != sha256_bytes(raw_stdout_bytes):
        raise QualError(f"{context} process scan raw-listing hash mismatch")
    if scan.get("stderr_sha256") != sha256_bytes(raw_stderr_bytes):
        raise QualError(f"{context} process scan stderr hash mismatch")
    if not isinstance(scan.get("forbidden_process_hits"), list):
        raise QualError(f"{context} process scan hits must be a list")


def parse_json_stdout(entry: dict[str, Any]) -> Any:
    txt = entry["_stdout"].decode("utf-8", "replace").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError as exc:
        raise QualError(f"malformed JSON from {entry.get('argv')}: {exc}: {txt[:300]}")


def parse_predecessor_identity(text: str) -> dict[str, str]:
    ident = {"hostname": "", "architecture": "", "cpu_model": ""}
    for line in text.splitlines():
        s = line.strip()
        if s == "catcas" and not ident["hostname"]:
            ident["hostname"] = "catcas"
        if s.startswith("HOSTNAME="):
            ident["hostname"] = s.split("=", 1)[1].strip()
        if s.startswith("architecture="):
            ident["architecture"] = s.split("=", 1)[1].strip()
        if s.startswith("cpu_model="):
            ident["cpu_model"] = s.split("=", 1)[1].strip()
    return ident


def main(argv: list[str] | None = None) -> int:
    global REC
    args = parse_args(argv)
    authority_path = args.replacement_authorization.resolve()
    authority = load_replacement_authority(authority_path)
    configure_runtime(authority, authority_path)

    # Authority and namespace closure are complete before any evidence path is
    # created and before any SSH/SCP helper can be called.
    EVID.mkdir(parents=False, exist_ok=False)
    for sub in ("local", "transfer", "target", "copy_back", "cleanup"):
        (EVID / sub / "logs").mkdir(parents=True, exist_ok=True)
    (EVID / "target" / "results").mkdir(parents=True, exist_ok=True)
    REC = Recorder(EVID)

    bindings: dict[str, Any] = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXEC_ORCHESTRATION_V2",
        "replacement_authority_schema_id": authority["schema_id"],
        "replacement_authority_id": authority["authority_id"],
        "replacement_authority_path": str(authority_path),
        "replacement_authority_sha256": REPLACEMENT_AUTHORITY_SHA256,
        "replacement_authority_git_blob_sha1": REPLACEMENT_AUTHORITY_GIT_BLOB_SHA1,
        "historical_authority_consumed": True,
        "historical_evidence_dir": HISTORICAL_EVIDENCE_REL,
        "replacement_evidence_dir": EVID.relative_to(REPO_ROOT).as_posix(),
        "maximum_target_qualification_executions": 1,
        "automatic_retry": False,
    }
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_a_deploy_"))
    deploy = tmpdir / Path(TRANSFER_STAGE).name

    try:
        # ---- Phase 1: host reconstruction ----
        run_local("host_build_compare_twice", [sys.executable, str(ADAPTER / "build_gate_a_execution_bundle.py"), "--compare-twice", "--quiet"], subdir="local")
        run_local("host_verify_adapter", [sys.executable, str(ADAPTER / "verify_gate_a_adapter_qualification.py")], subdir="local")
        run_local("host_adapter_no_drive", [sys.executable, str(ADAPTER / "gate_a_hardware_adapter.py"), "--qualify-no-drive"], subdir="local")
        run_local("host_emit_archive", [sys.executable, str(ADAPTER / "build_gate_a_execution_bundle.py"), "--emit-archive", str(deploy), "--quiet"], subdir="local")
        transfer_digest = sha256_file(deploy)
        deploy_size = deploy.stat().st_size
        bindings["deployment_archive_local_path"] = str(deploy)
        bindings["deployment_archive_size"] = deploy_size
        bindings["deployment_archive_host_sha256"] = transfer_digest
        bindings["execution_bundle_sha256"] = EXPECTED_EXECUTION_BUNDLE
        bindings["deterministic_archive_sha256"] = EXPECTED_ARCHIVE
        bindings["bundle_manifest_file_sha256"] = EXPECTED_MANIFEST_FILE

        # ---- Phase 2: preflight + identity ----
        id_before = run_ssh_py("target_identity_before", target_identity_script(), {"OUT": TP + "id_before.json"}, subdir="target")
        ident_before = parse_json_stdout(id_before)
        id_before_sha = id_before["stdout_sha256"]
        pred_ident = parse_predecessor_identity(PREDECESSOR_IDENTITY.read_text(encoding="utf-8"))
        for key in ("hostname", "architecture", "cpu_model"):
            if ident_before.get(key) != pred_ident.get(key):
                raise QualError(f"identity mismatch vs predecessor {key}: {ident_before.get(key)!r} != {pred_ident.get(key)!r}")
        if ident_before["hostname"] != EXPECTED_HOSTNAME or ident_before["architecture"] != EXPECTED_ARCH or ident_before["cpu_model"] != EXPECTED_CPU:
            raise QualError(f"identity does not match expected target: {ident_before}")
        bindings["predecessor_identity"] = pred_ident
        bindings["predecessor_identity_sha256"] = PREDECESSOR_ID_SHA
        bindings["current_identity_before"] = ident_before
        bindings["current_identity_before_sha256"] = id_before_sha

        # ---- Phase 3: prove namespace absence ----
        absent = parse_json_stdout(run_ssh_py("prove_absence", absence_script(), {"ROOT": EXEC_ROOT, "STAGE": TRANSFER_STAGE}, subdir="target"))
        bindings["execution_root_predeploy_state"] = absent["execution_root"]
        bindings["transfer_stage_predeploy_state"] = absent["transfer_stage"]
        if absent["execution_root"] != "ABSENT" or absent["transfer_stage"] != "ABSENT":
            raise QualError(f"namespace not provably absent: {absent}")

        # ---- Phase 4: transfer + verify + members + extract ----
        scp_to("upload_deploy_archive", deploy, TRANSFER_STAGE)
        tgt_digest_entry = run_ssh("verify_transfer_digest", f"python3 -c \"import hashlib;print(hashlib.sha256(open('{TRANSFER_STAGE}','rb').read()).hexdigest())\"", subdir="transfer")
        target_transfer_digest = tgt_digest_entry["_stdout"].decode().strip()
        bindings["deployment_archive_target_sha256"] = target_transfer_digest
        bindings["transfer_digest_match"] = (target_transfer_digest == transfer_digest)
        if target_transfer_digest != transfer_digest:
            raise QualError(f"transfer digest mismatch: target {target_transfer_digest} != host {transfer_digest}")
        members = parse_json_stdout(run_ssh_py("inspect_members", members_script(), {"STAGE": TRANSFER_STAGE}, subdir="transfer"))
        if members["violations"]:
            raise QualError(f"archive member violations: {members['violations']}")
        run_ssh_py("extract_bundle", extract_script(), {"ROOT": EXEC_ROOT, "STAGE": TRANSFER_STAGE}, subdir="target")

        # ---- Phase 5: pre-run strict custody ----
        before = parse_json_stdout(run_ssh_py("custody_before", custody_script("before"), {"ROOT": EXEC_ROOT, "PHASE": "before", "PYTHONDONTWRITEBYTECODE": "1", "OUT": TP + "before.json"}, subdir="target"))
        if not before["git_absent"]:
            raise QualError(".git present in extracted bundle")
        if not before["authority_artifact_absent"]:
            raise QualError(f"authority artifact present: {before['authority_artifact_hits']}")
        if before["validation"]["status"] != "GATE_A_TARGET_BUNDLE_VALIDATED" or before["validation"]["strict"] is not True:
            raise QualError(f"pre-run strict validation failed: {before['validation']}")
        if before["forbidden_generated_files"]:
            raise QualError(f"pre-run generated files present: {before['forbidden_generated_files']}")
        proc_before = parse_json_stdout(run_ssh_py("process_before", process_script(), {"OUT": TP + "proc_before.json"}, subdir="target"))
        validate_process_scan(proc_before, "before qualification")
        if proc_before["forbidden_process_hits"]:
            raise QualError(f"forbidden processes before: {proc_before['forbidden_process_hits']}")
        bindings["process_scan_before"] = proc_before
        bindings["strict_validation_before"] = before["validation"]
        bindings["before_tree_canonical_sha256"] = before["tree_canonical_sha256"]

        # ---- Phase 6: ONE no-drive qualification ----
        qual_cmd = (
            f"cd {EXEC_ROOT} && env PYTHONPATH= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
            f"timeout 180s python3 adapter/gate_a_target_runner.py --qualify-no-drive "
            f">{QUAL_STDOUT} 2>{QUAL_STDERR}"
        )
        qual = run_ssh("qualification_no_drive", qual_cmd, subdir="target", timeout=200, check=False)
        bindings["qualification_execution_count"] = 1
        bindings["qualification_exit_code"] = qual["exit_code"]
        # read back the captured stdout/stderr from the target temp files
        qual_out = run_ssh("qualification_stdout", f"cat {QUAL_STDOUT}", subdir="target", check=False)
        qual_err = run_ssh("qualification_stderr", f"cat {QUAL_STDERR}", subdir="target", check=False)
        (EVID / "target" / "results" / "TARGET_QUALIFICATION_RESULT.json").write_bytes(qual_out["_stdout"])
        (EVID / "target" / "results" / "TARGET_QUALIFICATION.stderr").write_bytes(qual_err["_stdout"])
        if qual["exit_code"] != 0:
            raise QualError(f"qualification returned rc={qual['exit_code']}; stopping (no retry)")
        qjson = parse_json_stdout(qual_out)
        req = {
            "status": "GATE_A_TARGET_RUNNER_NO_DRIVE_QUALIFIED",
            "git_free": True,
            "compiled": True,
        }
        for k, v in req.items():
            if qjson.get(k) != v:
                raise QualError(f"qualification field {k}={qjson.get(k)!r} != {v!r}")
        lbv = qjson["local_bundle_validation"]
        if lbv["status"] != "GATE_A_TARGET_BUNDLE_VALIDATED" or lbv["strict"] is not True:
            raise QualError(f"qualification local_bundle_validation weak: {lbv}")
        if lbv["execution_bundle_sha256"] != EXPECTED_EXECUTION_BUNDLE or lbv["deterministic_archive_sha256"] != EXPECTED_ARCHIVE:
            raise QualError(f"qualification bundle digests mismatch: {lbv}")
        if qjson["worker_validate_only"]["status"] != "GATE_A_WORKER_VALIDATE_ONLY_OK":
            raise QualError("worker validate-only not OK")
        for counter in ("network_connections_opened", "hardware_probes", "sender_starts", "receiver_captures", "control_writes", "msr_accesses", "hardware_executions"):
            if qjson.get(counter) != 0:
                raise QualError(f"nonzero counter {counter}={qjson.get(counter)}")
        bindings["qualification_status"] = qjson["status"]
        bindings["worker_validate_only_status"] = qjson["worker_validate_only"]["status"]
        bindings["qualification_json"] = qjson

        # ---- Phase 7: post-run custody ----
        after = parse_json_stdout(run_ssh_py("custody_after", custody_script("after"), {"ROOT": EXEC_ROOT, "PHASE": "after", "PYTHONDONTWRITEBYTECODE": "1", "OUT": TP + "after.json"}, subdir="target"))
        if after["validation"]["status"] != "GATE_A_TARGET_BUNDLE_VALIDATED" or after["validation"]["strict"] is not True:
            raise QualError(f"post-run strict validation failed: {after['validation']}")
        if after["forbidden_generated_files"]:
            raise QualError(f"post-run generated files present: {after['forbidden_generated_files']}")
        if after["tree"] != before["tree"]:
            raise QualError("bundle tree changed after qualification")
        if after["tree_canonical_sha256"] != before["tree_canonical_sha256"]:
            raise QualError("bundle tree canonical digest changed after qualification")
        proc_after = parse_json_stdout(run_ssh_py("process_after", process_script(), {"OUT": TP + "proc_after.json"}, subdir="target"))
        validate_process_scan(proc_after, "after qualification")
        if proc_after["forbidden_process_hits"]:
            raise QualError(f"forbidden processes after: {proc_after['forbidden_process_hits']}")
        bindings["process_scan_after"] = proc_after
        id_after = run_ssh_py("target_identity_after", target_identity_script(), {"OUT": TP + "id_after.json"}, subdir="target")
        ident_after = parse_json_stdout(id_after)
        for key in ("hostname", "architecture", "cpu_model"):
            if ident_after.get(key) != ident_before.get(key):
                raise QualError(f"identity changed after run {key}")
        bindings["current_identity_after"] = ident_after
        bindings["current_identity_after_sha256"] = id_after["stdout_sha256"]
        bindings["strict_validation_after"] = after["validation"]
        bindings["after_tree_canonical_sha256"] = after["tree_canonical_sha256"]
        bindings["bundle_tree_unchanged"] = True

        # ---- Phase 8: remote evidence package ----
        evidence_assemble = build_evidence_script()
        ev_env = {
            "ROOT": EXEC_ROOT, "EVROOT": EVIDENCE_ROOT, "EVARCHIVE": EV_ARCHIVE, "TP": TP,
            "TRANSFER_DIGEST": transfer_digest,
        }
        ev = parse_json_stdout(run_ssh_py("assemble_evidence", evidence_assemble, ev_env, subdir="target", timeout=120))
        target_evidence_archive_sha = ev["archive_sha256"]
        bindings["target_evidence_archive_sha256"] = target_evidence_archive_sha
        bindings["target_evidence_inventory"] = ev["inventory"]

        # ---- Phase 9: copy-back verification ----
        local_ev_archive = EVID / "copy_back" / "target_evidence.tar"
        scp_from("download_evidence_archive", EV_ARCHIVE, local_ev_archive, subdir="copy_back")
        local_ev_sha = sha256_file(local_ev_archive)
        bindings["copied_back_archive_sha256"] = local_ev_sha
        if local_ev_sha != target_evidence_archive_sha:
            raise QualError(f"copy-back archive digest mismatch: {local_ev_sha} != {target_evidence_archive_sha}")
        # extract and verify inventory
        cb_dir = EVID / "copy_back" / "target_evidence"
        cb_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local_ev_archive, "r:") as t:
            t.extractall(cb_dir, filter="data")
        unexpected: list[str] = []
        inv = ev["inventory"]
        inv_names = {e["path"] for e in inv}
        for entry in inv:
            fp = cb_dir / entry["path"]
            if not fp.is_file():
                raise QualError(f"copy-back missing inventory file: {entry['path']}")
            if fp.stat().st_size != entry["size"]:
                raise QualError(f"copy-back size mismatch: {entry['path']}")
            if sha256_file(fp) != entry["sha256"]:
                raise QualError(f"copy-back sha256 mismatch: {entry['path']}")
        for fp in sorted(cb_dir.rglob("*")):
            if fp.is_file():
                rel = fp.relative_to(cb_dir).as_posix()
                if rel not in inv_names:
                    unexpected.append(rel)
        if unexpected:
            raise QualError(f"copy-back unexpected files: {unexpected}")
        copy_back_receipt = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1",
            "retained_evidence_custody_verified": True,
            "target_evidence_archive_sha256": local_ev_sha,
            "inventory_verified": True,
            "unexpected_entries": [],
            "inventory_entry_count": len(inv),
        }
        (EVID / "COPY_BACK_RECEIPT.json").write_text(json.dumps(copy_back_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        bindings["copy_back_verified"] = True

        # ---- Phase 10: cleanup (only after verified copy-back) ----
        cleanup = parse_json_stdout(run_ssh_py("cleanup_namespace", cleanup_script(), {"ROOT": EXEC_ROOT, "STAGE": TRANSFER_STAGE, "EVARCHIVE": EV_ARCHIVE, "TP": TP}, subdir="cleanup"))
        if not (cleanup["exact_execution_root_removed"] and cleanup["exact_transfer_stage_removed"] and cleanup["execution_root_absence_proven"] and cleanup["transfer_stage_absence_proven"]):
            raise QualError(f"cleanup incomplete: {cleanup}")
        cleanup_scan = parse_json_stdout(
            run_ssh_py("process_after_cleanup", process_script(), subdir="cleanup")
        )
        validate_process_scan(cleanup_scan, "after cleanup")
        if cleanup_scan["forbidden_process_hits"]:
            raise QualError(f"forbidden processes after cleanup: {cleanup_scan['forbidden_process_hits']}")
        cleanup_receipt = {
            "schema_id": "CAT_CAS_PHASE6B6_GATE_A_CLEANUP_RECEIPT_V2",
            "exact_execution_root_removed": True,
            "exact_transfer_stage_removed": True,
            "execution_root_absence_proven": True,
            "transfer_stage_absence_proven": True,
            "process_scan": cleanup_scan,
            "forbidden_processes_remaining": cleanup_scan["forbidden_process_hits"],
        }
        (EVID / "CLEANUP_RECEIPT.json").write_text(json.dumps(cleanup_receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        bindings["cleanup_verified"] = True
        bindings["execution_root_final_state"] = cleanup["execution_root_final_state"]
        bindings["transfer_stage_final_state"] = cleanup["transfer_stage_final_state"]
        bindings["process_scan_after_cleanup"] = cleanup_scan

        bindings["overall_status"] = "SUCCESS"
    except Exception as exc:  # noqa: BLE001 - deterministic fail-closed capture
        bindings["overall_status"] = "FAILED"
        bindings["failure"] = f"{type(exc).__name__}: {exc}"
        _finalize(bindings)
        print(json.dumps({"status": "FAILED", "error": bindings["failure"]}, indent=2))
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    _finalize(bindings)
    print(json.dumps({"status": "SUCCESS", "final_bindings_sha256": bindings["_final_bindings_sha256"], "evidence_inventory_sha256": bindings["_evidence_inventory_sha256"]}, indent=2))
    return 0


def build_evidence_script() -> str:
    return (r'''
import os, json, hashlib, tarfile
evroot=os.environ["EVROOT"]; evarchive=os.environ["EVARCHIVE"]; tp=os.environ["TP"]
os.umask(0o022)
os.makedirs(evroot, mode=0o755)
def rd(p):
    with open(p,'rb') as f: return f.read()
def w(name, data):
    p=os.path.join(evroot,name)
    if isinstance(data,str): data=data.encode("utf-8")
    with open(p,'wb') as f: f.write(data)
def pj(name, obj):
    w(name, json.dumps(obj, sort_keys=True, indent=2)+"\n")
before=json.loads(rd(tp+"before.json")); after=json.loads(rd(tp+"after.json"))
id_before=json.loads(rd(tp+"id_before.json")); id_after=json.loads(rd(tp+"id_after.json"))
proc_before=json.loads(rd(tp+"proc_before.json")); proc_after=json.loads(rd(tp+"proc_after.json"))
w("TARGET_IDENTITY_BEFORE.json", rd(tp+"id_before.json"))
w("TARGET_IDENTITY_AFTER.json", rd(tp+"id_after.json"))
w("TARGET_IDENTITY_BEFORE.stdout", rd(tp+"id_before.json"))
w("TARGET_IDENTITY_AFTER.stdout", rd(tp+"id_after.json"))
w("TARGET_QUALIFICATION_RESULT.json", rd(tp+"qual.stdout"))
w("TARGET_QUALIFICATION.stderr", rd(tp+"qual.stderr"))
pj("TARGET_BUNDLE_VALIDATION_BEFORE.json", before["validation"])
pj("TARGET_BUNDLE_VALIDATION_AFTER.json", after["validation"])
tb=before["tree"]; ta=after["tree"]
pj("TARGET_TREE_BEFORE.json", tb)
pj("TARGET_TREE_AFTER.json", ta)
pj("TARGET_TREE_COMPARISON.json", {"identical": tb==ta, "before_count": len(tb), "after_count": len(ta), "before_canonical_sha256": before["tree_canonical_sha256"], "after_canonical_sha256": after["tree_canonical_sha256"]})
w("TARGET_PROCESS_STATE_BEFORE.txt", json.dumps(proc_before, sort_keys=True, indent=2)+"\n")
w("TARGET_PROCESS_STATE_AFTER.txt", json.dumps(proc_after, sort_keys=True, indent=2)+"\n")
w("TARGET_TRANSFER_DIGEST.txt", os.environ["TRANSFER_DIGEST"]+"\n")
# inventory of evidence files
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(65536),b''): h.update(c)
    return h.hexdigest()
inv=[]
for n in sorted(os.listdir(evroot)):
    fp=os.path.join(evroot,n)
    if os.path.isfile(fp):
        stt=os.stat(fp)
        inv.append({"path":n,"mode":oct(stt.st_mode & 0o7777),"size":stt.st_size,"sha256":sha256_file(fp)})
pj("TARGET_EVIDENCE_INVENTORY.json", inv)
# rebuild inventory including the inventory file itself
inv=[]
for n in sorted(os.listdir(evroot)):
    fp=os.path.join(evroot,n)
    if os.path.isfile(fp):
        stt=os.stat(fp)
        inv.append({"path":n,"mode":oct(stt.st_mode & 0o7777),"size":stt.st_size,"sha256":sha256_file(fp)})
# deterministic archive of evidence dir (sorted members, fixed metadata)
if os.path.lexists(evarchive): raise SystemExit("evidence archive already exists")
with tarfile.open(evarchive,'w:') as t:
    for e in sorted(inv, key=lambda x:x["path"]):
        fp=os.path.join(evroot,e["path"])
        ti=tarfile.TarInfo(name=e["path"]); ti.size=e["size"]; ti.mode=0o644; ti.mtime=0; ti.uid=0; ti.gid=0; ti.uname=""; ti.gname=""
        with open(fp,'rb') as fh: t.addfile(ti, fh)
archive_sha=sha256_file(evarchive)
print(json.dumps({"inventory":inv,"archive_sha256":archive_sha}, sort_keys=True))
''')


def cleanup_script() -> str:
    return f'''
import os, json, shutil, glob
root=os.environ["ROOT"]; stage=os.environ["STAGE"]; evarchive=os.environ["EVARCHIVE"]; tp=os.environ["TP"]
res={{}}
# exact-string guards
assert root=={EXEC_ROOT!r}, "root guard"
assert stage=={TRANSFER_STAGE!r}, "stage guard"
assert evarchive=={EV_ARCHIVE!r}, "evidence archive guard"
assert tp=={TP!r}, "tp guard"
def state(p):
    try:
        os.lstat(p); return "PRESENT"
    except FileNotFoundError:
        return "ABSENT"
    except OSError as e:
        return "UNOBSERVABLE:%s"%type(e).__name__
if os.path.lexists(root) and not os.path.islink(root):
    shutil.rmtree(root)
elif os.path.islink(root):
    os.unlink(root)
if os.path.lexists(stage):
    os.unlink(stage)
if os.path.lexists(evarchive):
    os.unlink(evarchive)
for f in glob.glob(tp+"*"):
    try: os.unlink(f)
    except OSError: pass
res["execution_root_final_state"]=state(root)
res["transfer_stage_final_state"]=state(stage)
res["exact_execution_root_removed"]= state(root)=="ABSENT"
res["exact_transfer_stage_removed"]= state(stage)=="ABSENT"
res["execution_root_absence_proven"]= state(root)=="ABSENT"
res["transfer_stage_absence_proven"]= state(stage)=="ABSENT"
print(json.dumps(res, sort_keys=True))
'''


def _finalize(bindings: dict[str, Any]) -> None:
    readme = EVID / "README.md"
    readme.write_text(
        "# Gate A target non-executing qualification evidence\n\n"
        f"Overall status: {bindings.get('overall_status')}\n\n"
        "This directory contains the deterministic evidence for one owner-authorized\n"
        "Gate A target non-executing qualification. No hardware was driven, no probe\n"
        "ran, no execute-authorized path ran, and no execution authority artifact was\n"
        "created. See FINAL_BINDINGS.json, COMMANDS.jsonl, COPY_BACK_RECEIPT.json,\n"
        "CLEANUP_RECEIPT.json, and EVIDENCE_INVENTORY.json.\n",
        encoding="utf-8",
    )
    # evidence inventory of the repository-side evidence tree
    inv = []
    for fp in sorted(EVID.rglob("*")):
        if fp.is_file() and fp.name not in ("EVIDENCE_INVENTORY.json", "FINAL_BINDINGS.json"):
            inv.append({"path": fp.relative_to(EVID).as_posix(), "size": fp.stat().st_size, "sha256": sha256_file(fp)})
    inv_obj = {"schema_id": "CAT_CAS_PHASE6B6_GATE_A_TARGET_NONEXEC_EVIDENCE_INVENTORY_V1", "files": inv, "file_count": len(inv)}
    (EVID / "EVIDENCE_INVENTORY.json").write_text(json.dumps(inv_obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    bindings["_evidence_inventory_sha256"] = sha256_file(EVID / "EVIDENCE_INVENTORY.json")
    fb_bytes = json.dumps(bindings, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    (EVID / "FINAL_BINDINGS.json").write_bytes(fb_bytes)
    bindings["_final_bindings_sha256"] = sha256_bytes(fb_bytes)


if __name__ == "__main__":
    raise SystemExit(main())

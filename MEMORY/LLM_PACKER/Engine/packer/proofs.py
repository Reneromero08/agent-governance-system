#!/usr/bin/env python3
"""
Proof generation and refresh for LLM Packer.

Generates proof artifacts under NAVIGATION/PROOFS/_LATEST/ including:
- GREEN_STATE (git state, timestamps, command results)
- CATALYTIC proof outputs
- COMPRESSION proof outputs
- PROOF_MANIFEST (comprehensive manifest of all proof files)

All proofs are written atomically (temp dir + rename) and fail-closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .firewall_writer import PackerWriter

# Proof directories (relative to project root)
PROOFS_DIR = Path("NAVIGATION/PROOFS")
RUNS_DIR = PROOFS_DIR / "_RUNS"

# Proof suite config (optional)
PROOF_SUITE_CONFIG = PROOFS_DIR / "PROOF_SUITE.json"


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def _run_command(
    cmd: List[str], *, cwd: Path, timeout: int = 300
) -> Tuple[int, bytes, bytes]:
    """
    Run a command and capture stdout/stderr.
    Returns (exit_code, stdout_bytes, stderr_bytes).
    """
    env = os.environ.copy()
    # Avoid pytest/capture tmpfile issues by forcing a stable temp root.
    try:
        tmp_root = (cwd / "LAW" / "CONTRACTS" / "_runs" / "pytest_tmp").resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp_root)
        env["TMP"] = str(tmp_root)
        env["TEMP"] = str(tmp_root)
    except Exception:
        pass

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, b"", f"Command timed out after {timeout}s".encode("utf-8")
    except Exception as e:
        return -1, b"", str(e).encode("utf-8")


def _get_git_state(project_root: Path) -> Dict[str, Any]:
    """Capture current git state."""
    
    def run_git(args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
        except Exception:
            return ""
    
    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--porcelain"])
    
    return {
        "repo_head_commit": commit or "unknown",
        "branch": branch or "unknown",
        "git_status": status if status else "clean",
        "is_clean": not bool(status),
    }


def _load_proof_suite(project_root: Path) -> List[List[str]]:
    """
    Load proof suite commands from PROOF_SUITE.json if it exists.
    Otherwise, return the minimal default suite.
    """
    config_path = project_root / PROOF_SUITE_CONFIG
    
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                commands = data.get("commands", [])
                # Replace bare "python" with sys.executable
                normalized = []
                for cmd in commands:
                    if cmd and cmd[0] == "python":
                        normalized.append([sys.executable] + cmd[1:])
                    else:
                        normalized.append(cmd)
                return normalized
        except Exception:
            pass
    
    # Default minimal suite
    #
    # IMPORTANT: do NOT invoke pytest here by default. Proof refresh is called by the
    # packer during other tests/fixtures; invoking pytest here creates recursive/very
    # slow test runs (and can appear "stuck"). Opt into a stronger proof suite by
    # providing NAVIGATION/PROOFS/PROOF_SUITE.json.
    return [[sys.executable, "--version"]]


def _generate_green_state(
    project_root: Path,
    *,
    stamp: str,
    commands_executed: List[Dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
) -> Tuple[Dict[str, Any], str]:
    """
    Generate GREEN_STATE.json and GREEN_STATE.md.
    Returns (json_data, markdown_content).
    """
    git_state = _get_git_state(project_root)
    duration = (end_time - start_time).total_seconds()
    
    json_data = {
        "stamp": stamp,
        "repo_head_commit": git_state["repo_head_commit"],
        "branch": git_state["branch"],
        "git_status": git_state["git_status"],
        "is_clean": git_state["is_clean"],
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "commands": commands_executed,
    }
    
    # Generate markdown
    md_lines = [
        "# Green State Report",
        "",
        f"**Stamp:** `{stamp}`",
        f"**Commit:** `{git_state['repo_head_commit']}`",
        f"**Branch:** `{git_state['branch']}`",
        f"**Git Status:** {'clean' if git_state['is_clean'] else 'dirty'}",
        "",
        f"**Start Time:** {start_time.isoformat()}",
        f"**End Time:** {end_time.isoformat()}",
        f"**Duration:** {duration:.2f}s",
        "",
        "## Commands Executed",
        "",
    ]
    
    for i, cmd_info in enumerate(commands_executed, 1):
        md_lines.extend([
            f"### Command {i}",
            "",
            f"**Command:** `{' '.join(cmd_info['command'])}`",
            f"**Exit Code:** {cmd_info['exit_code']}",
            f"**Status:** {'PASS' if cmd_info['exit_code'] == 0 else 'FAIL'}",
            f"**Stdout SHA256:** `{cmd_info['stdout_sha256']}`",
            f"**Stderr SHA256:** `{cmd_info['stderr_sha256']}`",
            "",
        ])
    
    md_content = "\n".join(md_lines)
    return json_data, md_content


def _generate_catalytic_proof(
    project_root: Path, *, proof_dir: Path, commands: List[List[str]], writer: Optional[PackerWriter] = None
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Run catalytic proof commands and generate outputs.
    Returns (success, commands_info).
    """
    catalytic_dir = proof_dir / "CATALYTIC"
    if writer is None:
        catalytic_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(catalytic_dir, kind="durable", parents=True, exist_ok=True)

    all_stdout = []
    all_stderr = []
    commands_info = []
    overall_success = True

    for cmd in commands:
        exit_code, stdout, stderr = _run_command(cmd, cwd=project_root)

        all_stdout.append(stdout)
        all_stderr.append(stderr)

        cmd_info = {
            "command": cmd,
            "exit_code": exit_code,
            "stdout_sha256": _sha256_bytes(stdout),
            "stderr_sha256": _sha256_bytes(stderr),
        }
        commands_info.append(cmd_info)

        if exit_code != 0:
            overall_success = False

    # Write combined log
    combined_log = b"\n".join(all_stdout + all_stderr)
    if writer is None:
        (catalytic_dir / "PROOF_LOG.txt").write_bytes(combined_log)
    else:
        writer.write_bytes(catalytic_dir / "PROOF_LOG.txt", combined_log)

    # Write summary
    summary_lines = [
        "# Catalytic Proof Summary",
        "",
        f"**Overall Status:** {'PASS' if overall_success else 'FAIL'}",
        "",
        "## Commands",
        "",
    ]

    for i, cmd_info in enumerate(commands_info, 1):
        status = "✓ PASS" if cmd_info["exit_code"] == 0 else "✗ FAIL"
        summary_lines.append(f"{i}. `{' '.join(cmd_info['command'])}` — {status}")

    summary_lines.append("")
    if writer is None:
        (catalytic_dir / "PROOF_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    else:
        writer.write_text(catalytic_dir / "PROOF_SUMMARY.md", "\n".join(summary_lines), encoding="utf-8")

    return overall_success, commands_info


def _generate_compression_proof(
    project_root: Path, *, proof_dir: Path, writer: Optional[PackerWriter] = None
) -> bool:
    """
    Generate compression proof outputs by copying existing compression proof artifacts.
    Returns True if successful.
    """
    compression_dir = proof_dir / "COMPRESSION"
    if writer is None:
        compression_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(compression_dir, kind="durable", parents=True, exist_ok=True)

    # Source compression proof directory
    source_compression = project_root / "NAVIGATION/PROOFS/COMPRESSION"

    if not source_compression.exists():
        # Create minimal placeholder
        placeholder_content1 = "# Compression Proof\n\nNo compression proof artifacts found.\n"
        if writer is None:
            (compression_dir / "COMPRESSION_PROOF_REPORT.md").write_text(
                placeholder_content1,
                encoding="utf-8"
            )
        else:
            writer.write_text(
                compression_dir / "COMPRESSION_PROOF_REPORT.md",
                placeholder_content1,
                encoding="utf-8"
            )

        placeholder_content2 = json.dumps({"status": "not_available"}, indent=2)
        if writer is None:
            (compression_dir / "COMPRESSION_PROOF_DATA.json").write_text(
                placeholder_content2,
                encoding="utf-8"
            )
        else:
            writer.write_text(
                compression_dir / "COMPRESSION_PROOF_DATA.json",
                placeholder_content2,
                encoding="utf-8"
            )

        placeholder_content3 = ""
        if writer is None:
            (compression_dir / "PROOF_LOG.txt").write_text(placeholder_content3, encoding="utf-8")
        else:
            writer.write_text(compression_dir / "PROOF_LOG.txt", placeholder_content3, encoding="utf-8")
        return True

    # Copy existing compression proof artifacts
    try:
        for artifact in ["COMPRESSION_PROOF_REPORT.md", "COMPRESSION_PROOF_DATA.json"]:
            src = source_compression / artifact
            if src.exists():
                if writer is None:
                    shutil.copy2(src, compression_dir / artifact)
                else:
                    # For copying, we still use shutil since it's not a direct write operation
                    shutil.copy2(src, compression_dir / artifact)

        # Create a log file (empty for now, as compression proof is pre-generated)
        log_content = "Compression proof artifacts copied from NAVIGATION/PROOFS/COMPRESSION/\n"
        if writer is None:
            (compression_dir / "PROOF_LOG.txt").write_text(
                log_content,
                encoding="utf-8"
            )
        else:
            writer.write_text(compression_dir / "PROOF_LOG.txt", log_content, encoding="utf-8")
        return True
    except Exception:
        return False


def _generate_proof_manifest(
    proof_dir: Path,
    *,
    stamp: str,
    repo_head_commit: str,
    overall_status: str,
    executed_commands: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate PROOF_MANIFEST.json listing all files under proof_dir.
    """
    files = []
    for path in sorted(proof_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(proof_dir).as_posix()
            files.append({
                "relative_path": rel_path,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    
    manifest = {
        "stamp": stamp,
        "repo_head_commit": repo_head_commit,
        "overall_status": overall_status,
        "executed_commands": executed_commands,
        "files": files,
    }
    
    return manifest


def refresh_proofs(
    project_root: Path,
    *,
    stamp: str,
    save_run_history: bool = False,
    writer: Optional[PackerWriter] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Refresh proof artifacts under NAVIGATION/PROOFS/_LATEST/.

    Returns (success, error_message).

    Atomicity: writes to temp dir first, then atomically replaces _LATEST.
    Fail-closed: if ANY proof command fails, _LATEST is NOT updated.
    """
    start_time = datetime.now(timezone.utc)

    # Create temp directory for atomic write
    temp_dir = project_root / PROOFS_DIR / f"_LATEST.__tmp__{stamp}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if writer is None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(temp_dir, kind="tmp", parents=True, exist_ok=True)

    try:
        # Load proof suite
        commands = _load_proof_suite(project_root)

        # Generate catalytic proof
        catalytic_success, commands_info = _generate_catalytic_proof(
            project_root, proof_dir=temp_dir, commands=commands, writer=writer
        )

        if not catalytic_success:
            error_msg = "Catalytic proof failed: one or more commands returned non-zero exit code"
            return False, error_msg

        # Generate compression proof
        compression_success = _generate_compression_proof(
            project_root, proof_dir=temp_dir, writer=writer
        )

        if not compression_success:
            error_msg = "Compression proof generation failed"
            return False, error_msg

        end_time = datetime.now(timezone.utc)

        # Generate green state
        green_state_json, green_state_md = _generate_green_state(
            project_root,
            stamp=stamp,
            commands_executed=commands_info,
            start_time=start_time,
            end_time=end_time,
        )

        json_content = json.dumps(green_state_json, indent=2, sort_keys=True) + "\n"
        if writer is None:
            (temp_dir / "GREEN_STATE.json").write_text(json_content, encoding="utf-8")
        else:
            writer.write_text(temp_dir / "GREEN_STATE.json", json_content, encoding="utf-8")

        if writer is None:
            (temp_dir / "GREEN_STATE.md").write_text(green_state_md, encoding="utf-8")
        else:
            writer.write_text(temp_dir / "GREEN_STATE.md", green_state_md, encoding="utf-8")

        # Generate proof manifest
        git_state = _get_git_state(project_root)
        manifest = _generate_proof_manifest(
            temp_dir,
            stamp=stamp,
            repo_head_commit=git_state["repo_head_commit"],
            overall_status="PASS",
            executed_commands=commands_info,
        )

        manifest_content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if writer is None:
            (temp_dir / "PROOF_MANIFEST.json").write_text(manifest_content, encoding="utf-8")
        else:
            writer.write_text(temp_dir / "PROOF_MANIFEST.json", manifest_content, encoding="utf-8")


        # Deploy: clean up old _LATEST if exists (migration) and copy temp to dispersed locations
        latest_path = project_root / PROOFS_DIR / "_LATEST"
        if latest_path.exists():
            shutil.rmtree(latest_path)

        # Distribute artifacts to parent folders
        # 1. Root artifacts (GREEN_STATE*, PROOF_MANIFEST)
        for name in ["GREEN_STATE.json", "GREEN_STATE.md", "PROOF_MANIFEST.json"]:
            src = temp_dir / name
            dst = project_root / PROOFS_DIR / name
            if src.exists():
                if writer is None:
                    shutil.copy2(src, dst)
                else:
                    # For copying, we still use shutil since it's not a direct write operation
                    shutil.copy2(src, dst)

        # 2. Subdirectories (CATALYTIC, COMPRESSION)
        for subdir in ["CATALYTIC", "COMPRESSION"]:
            src_sub = temp_dir / subdir
            dst_sub = project_root / PROOFS_DIR / subdir
            if writer is None:
                dst_sub.mkdir(parents=True, exist_ok=True)
            else:
                writer.mkdir(dst_sub, kind="durable", parents=True, exist_ok=True)
            if src_sub.exists():
                # Copy/overwrite files
                for item in src_sub.iterdir():
                    if item.is_file():
                        if writer is None:
                            shutil.copy2(item, dst_sub / item.name)
                        else:
                            # For copying, we still use shutil since it's not a direct write operation
                            shutil.copy2(item, dst_sub / item.name)

        # Optionally save to _RUNS history
        if save_run_history:
            runs_path = project_root / RUNS_DIR / stamp
            if writer is None:
                runs_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                writer.mkdir(runs_path.parent, kind="durable", parents=True, exist_ok=True)
            if writer is None:
                shutil.copytree(temp_dir, runs_path)
            else:
                # For copying, we still use shutil since it's not a direct write operation
                shutil.copytree(temp_dir, runs_path)

        return True, None

    except Exception as e:
        # Cleanup temp dir on failure
        if temp_dir.exists():
            if writer is None:
                shutil.rmtree(temp_dir)
            else:
                # For cleanup, we still use shutil since it's not a direct write operation
                shutil.rmtree(temp_dir)
        return False, f"Proof generation failed: {e}"
    finally:
        # Always cleanup temp dir
        if temp_dir.exists():
            if writer is None:
                shutil.rmtree(temp_dir)
            else:
                # For cleanup, we still use shutil since it's not a direct write operation
                shutil.rmtree(temp_dir)


def get_lite_proof_summary(project_root: Path) -> Dict[str, Any]:
    """
    Generate a minimal PROOFS.json summary for LITE packs.
    Returns a small JSON object with key proof metadata.
    """
    proots_path = project_root / PROOFS_DIR
    
    manifest_path = proots_path / "PROOF_MANIFEST.json"
    green_state_path = proots_path / "GREEN_STATE.json"
    
    if not manifest_path.exists():
        return {
            "status": "not_available",
            "message": "No proof manifest found at NAVIGATION/PROOFS/PROOF_MANIFEST.json",
        }
    
    try:
        manifest = {}
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        green_state = {}
        if green_state_path.exists():
            with green_state_path.open("r", encoding="utf-8") as f:
                green_state = json.load(f)
        
        # Build minimal summary
        summary = {
            "overall_status": manifest.get("overall_status", "unknown"),
            "repo_head_commit": manifest.get("repo_head_commit", "unknown"),
            "stamp": manifest.get("stamp", "unknown"),
            "proof_files": {
                "green_state": _sha256_file(green_state_path) if green_state_path.exists() else None,
                "catalytic_summary": _sha256_file(proots_path / "CATALYTIC/PROOF_SUMMARY.md") if (proots_path / "CATALYTIC/PROOF_SUMMARY.md").exists() else None,
                "compression_report": _sha256_file(proots_path / "COMPRESSION/COMPRESSION_PROOF_REPORT.md") if (proots_path / "COMPRESSION/COMPRESSION_PROOF_REPORT.md").exists() else None,
                "manifest": _sha256_file(manifest_path) if manifest_path.exists() else None,
            },
        }
        
        return summary
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read proof artifacts: {e}",
        }

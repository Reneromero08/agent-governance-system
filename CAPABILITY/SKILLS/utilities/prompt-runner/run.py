#!/usr/bin/env python3
"""Prompt runner skill.

Parse a prompt file, enforce prompt canon gates, optionally run commands,
write receipt/report artifacts, and emit a deterministic summary JSON.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALLOWED_OUTPUT_ROOT = (PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs").resolve()
POLICY_CANON_PATH = PROJECT_ROOT / "NAVIGATION" / "PROMPTS" / "1_PROMPT_POLICY_CANON.md"
GUIDE_CANON_PATH = PROJECT_ROOT / "NAVIGATION" / "PROMPTS" / "2_PROMPT_GENERATOR_GUIDE_FINAL.md"
SECTION_INDEX_PATHS = [
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated" / "SECTION_INDEX.json",
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "meta" / "SECTION_INDEX.json",
]
LINT_COMMAND = "bash CAPABILITY/TOOLS/linters/lint_prompt_pack.sh NAVIGATION/PROMPTS"

NON_PLANNER_MODELS = {
    "Gemini Flash",
    "Gemini Pro Low",
    "GPT Codex",
    "Grok Code Fast",
}

REQUIRED_HEADER_FIELDS = [
    "phase",
    "task_id",
    "slug",
    "policy_canon_sha256",
    "guide_canon_sha256",
    "depends_on",
    "primary_model",
    "fallback_chain",
    "receipt_path",
    "report_path",
    "max_report_lines",
]

REQUIRED_SECTIONS = [
    "ROLE + MODEL",
    "GOAL",
    "SCOPE (WRITE ALLOWLIST)",
    "REQUIRED FACTS",
    "PLAN",
    "VALIDATION",
    "ARTIFACTS",
    "EXIT CRITERIA",
]


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _parse_frontmatter(lines: List[str]) -> Tuple[Dict[str, Any], int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0

    frontmatter: Dict[str, Any] = {}
    for idx in range(1, len(lines)):
        line = lines[idx].rstrip("\n")
        if line.strip() == "---":
            return frontmatter, idx + 1
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed = _parse_value(value.strip())
        frontmatter[key.strip()] = parsed
    return frontmatter, 0


def _parse_value(raw: str) -> Any:
    if not raw:
        return ""
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip().strip("\"").strip("'") for p in inner.split(",")]
            return [p for p in parts if p]
    lower = value.lower().strip("\"").strip("'")
    if lower in {"true", "false"}:
        return lower == "true"
    if value.isdigit():
        try:
            return int(value)
        except ValueError:
            return value.strip("\"").strip("'")
    return value.strip("\"").strip("'")


def _extract_headings(lines: List[str]) -> List[str]:
    headings: List[str] = []
    for line in lines:
        if line.startswith("## "):
            headings.append(line[3:].strip())
    return headings


def _find_fill_me_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    marker = "FILL_ME__"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            break
        end = idx + len(marker)
        if end < len(text) and text[end] == "<":
            start = end + 1
            continue
        while end < len(text) and (text[end].isalnum() or text[end] in "_-"):
            end += 1
        token = text[idx:end]
        if token != marker and token not in tokens:
            tokens.append(token)
        start = end
    return tokens


def _extract_section(lines: List[str], title: str) -> List[str]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == f"## {title}".lower():
            start = idx + 1
            break
    if start is None:
        return []
    section_lines: List[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        section_lines.append(line)
    return section_lines


def _parse_allowlist(lines: List[str]) -> Tuple[List[str], List[str]]:
    allowed_writes: List[str] = []
    allowed_deletes: List[str] = []
    current: Optional[List[str]] = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("- allowed writes"):
            current = allowed_writes
            continue
        if stripped.lower().startswith("- allowed deletes") or stripped.lower().startswith("- allowed delete"):
            current = allowed_deletes
            continue
        if stripped.startswith("-") and "forbidden" in stripped.lower():
            current = None
            continue

        if current is not None and stripped.startswith("-"):
            item = stripped.lstrip("-").strip()
            if " (" in item:
                item = item.split(" (", 1)[0].strip()
            if item and item.lower() not in {"none", "n/a"}:
                current.append(item)

    return allowed_writes, allowed_deletes


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _resolve_repo_path(path_str: str) -> Path:
    path = (PROJECT_ROOT / path_str).resolve()
    if not str(path).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path escapes repo root: {path_str}")
    return path


def _ensure_allowed_output(path: Path) -> None:
    if not str(path).startswith(str(ALLOWED_OUTPUT_ROOT)):
        raise ValueError(f"Output path must be under LAW/CONTRACTS/_runs: {path}")


def _is_path_allowed(path: Path, allowlist: List[str]) -> bool:
    normalized = _normalize_path(str(path.relative_to(PROJECT_ROOT)))
    for entry in allowlist:
        entry_norm = _normalize_path(entry)
        if normalized == entry_norm:
            return True
        if entry_norm.endswith("/"):
            if normalized.startswith(entry_norm):
                return True
        else:
            if normalized.startswith(entry_norm.rstrip("/") + "/"):
                return True
    return False


def _truncate(text: str, max_bytes: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    truncated = data[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n... [TRUNCATED]"


def _run_command(command: str, timeout_sec: int, max_output_bytes: int) -> Dict[str, Any]:
    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        creationflags=creationflags,
    )

    stdout = _truncate(result.stdout or "", max_output_bytes)
    stderr = _truncate(result.stderr or "", max_output_bytes)

    return {
        "command": command,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _resolve_artifact_path(frontmatter: Dict[str, Any], key: str, default_path: Path) -> Path:
    value = frontmatter.get(key)
    if not value:
        return default_path
    resolved = _resolve_repo_path(str(value))
    _ensure_allowed_output(resolved)
    return resolved


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_section_index(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_section_index(prompt_path: str) -> Optional[Path]:
    normalized = _normalize_path(prompt_path)
    for index_path in SECTION_INDEX_PATHS:
        if not index_path.exists():
            continue
        try:
            sections = _load_section_index(index_path)
        except json.JSONDecodeError:
            continue
        for record in sections:
            record_path = str(record.get("path", "")) if "path" in record else ""
            anchor = str(record.get("anchor", "")) if "anchor" in record else ""
            anchor_path = anchor.split("#", 1)[0] if anchor else ""
            if _normalize_path(record_path) == normalized or _normalize_path(anchor_path) == normalized:
                return index_path
    return None


def _match_required_sections(headings: List[str]) -> List[str]:
    missing = []
    for required in REQUIRED_SECTIONS:
        required_lower = required.strip().lower()
        found = False
        for heading in headings:
            if heading.strip().lower().startswith(required_lower):
                found = True
                break
        if not found:
            missing.append(required)
    return missing


def _validate_plan_ref(primary_model: str, plan_ref: str) -> bool:
    for model in NON_PLANNER_MODELS:
        if model.lower() in (primary_model or "").lower():
            return bool(plan_ref)
    return True


def _build_report(
    *,
    status: str,
    prompt_path: str,
    prompt_sha256: str,
    policy_hash: str,
    guide_hash: str,
    lint_info: Dict[str, Any],
    fill_tokens: List[str],
    allowed_writes: List[str],
    allowed_deletes: List[str],
    commands_run: List[Dict[str, Any]],
    errors: List[str],
    cortex_index: str,
) -> str:
    lines: List[str] = []
    lines.append("# Prompt Runner Report")
    lines.append("")
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("## Prompt")
    lines.append(f"Path: {prompt_path}")
    lines.append(f"SHA256: {prompt_sha256}")
    lines.append(f"Cortex index: {cortex_index or '(none)'}")
    lines.append("")
    lines.append("## Canon Hashes")
    lines.append(f"policy_canon_sha256: {policy_hash}")
    lines.append(f"guide_canon_sha256: {guide_hash}")
    lines.append("")
    lines.append("## Lint")
    lines.append(f"command: {lint_info.get('command')}")
    lines.append(f"exit_code: {lint_info.get('exit_code')}")
    lines.append(f"result: {lint_info.get('result')}")
    lines.append("")
    lines.append("## FILL_ME__ Tokens")
    lines.append(", ".join(fill_tokens) if fill_tokens else "(none)")
    lines.append("")
    lines.append("## Allowlist")
    lines.append("Allowed writes:")
    if allowed_writes:
        for item in allowed_writes:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    lines.append("Allowed deletes/renames:")
    if allowed_deletes:
        for item in allowed_deletes:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Commands")
    if commands_run:
        for item in commands_run:
            lines.append(f"- {item.get('command')} (exit {item.get('exit_code')})")
    else:
        lines.append("- (none)")
    if errors:
        lines.append("")
        lines.append("## Errors")
        for err in errors:
            lines.append(f"- {err}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        data = _load_json(input_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Failed to load input: {exc}")
        return 1

    errors: List[str] = []
    policy_breach = False
    verification_failed = False
    blocked_unknown = False

    task_id = data.get("task_id")
    prompt_path = data.get("prompt_path")
    output_dir = data.get("output_dir")

    if not isinstance(task_id, str) or not task_id:
        errors.append("task_id is required")
        policy_breach = True
    if not isinstance(prompt_path, str) or not prompt_path:
        errors.append("prompt_path is required")
        policy_breach = True
    if not isinstance(output_dir, str) or not output_dir:
        errors.append("output_dir is required")
        policy_breach = True

    emit_data = bool(data.get("emit_data", False))
    commands = data.get("commands", [])
    max_output_bytes = int(data.get("max_output_bytes", 100000))
    plan_ref = str(data.get("plan_ref", "") or "")
    manifest_path = data.get("manifest_path")

    prompt_text = ""
    frontmatter: Dict[str, Any] = {}
    headings: List[str] = []
    fill_me_tokens: List[str] = []
    prompt_sha256 = ""

    allowed_writes: List[str] = []
    allowed_deletes: List[str] = []
    cortex_index_used = ""

    lint_info = {
        "command": LINT_COMMAND,
        "exit_code": -1,
        "result": "NOT_RUN",
        "stdout": "",
        "stderr": "",
    }

    receipt_path = None
    report_path = None
    data_path = None

    if not errors:
        try:
            output_dir_path = _resolve_repo_path(output_dir)
            _ensure_allowed_output(output_dir_path)
            default_receipt = output_dir_path / "receipt.json"
            default_report = output_dir_path / "REPORT.md"
            receipt_path = default_receipt
            report_path = default_report

            prompt_file = _resolve_repo_path(prompt_path)
            if not prompt_file.exists():
                errors.append(f"prompt_path not found: {prompt_path}")
                policy_breach = True
            else:
                index_path = _find_section_index(prompt_path)
                if not index_path:
                    errors.append(f"prompt_path not in cortex index: {prompt_path}")
                    policy_breach = True
                else:
                    cortex_index_used = _normalize_path(str(index_path.relative_to(PROJECT_ROOT)))
                    prompt_text = prompt_file.read_text(encoding="utf-8")
                    prompt_sha256 = _sha256_text(prompt_text)
                    lines = prompt_text.splitlines()
                    frontmatter, _ = _parse_frontmatter(lines)
                    headings = _extract_headings(lines)
                    fill_me_tokens = _find_fill_me_tokens(prompt_text)

                if prompt_text:
                    missing_headers = [f for f in REQUIRED_HEADER_FIELDS if f not in frontmatter]
                    if missing_headers:
                        errors.append(f"missing required header fields: {', '.join(missing_headers)}")
                        policy_breach = True

                    missing_sections = _match_required_sections(headings)
                    if missing_sections:
                        errors.append(f"missing required sections: {', '.join(missing_sections)}")
                        policy_breach = True

                    if fill_me_tokens:
                        errors.append("FILL_ME__ tokens unresolved")
                        blocked_unknown = True

                    if POLICY_CANON_PATH.exists() and GUIDE_CANON_PATH.exists():
                        policy_hash = _sha256_file(POLICY_CANON_PATH)
                        guide_hash = _sha256_file(GUIDE_CANON_PATH)
                        expected_policy = str(frontmatter.get("policy_canon_sha256", ""))
                        expected_guide = str(frontmatter.get("guide_canon_sha256", ""))
                        if expected_policy and expected_policy != policy_hash:
                            errors.append("policy_canon_sha256 mismatch")
                            policy_breach = True
                        if expected_guide and expected_guide != guide_hash:
                            errors.append("guide_canon_sha256 mismatch")
                            policy_breach = True
                    else:
                        errors.append("prompt canon files missing")
                        policy_breach = True

                    if not _validate_plan_ref(str(frontmatter.get("primary_model", "")), plan_ref):
                        errors.append("plan_ref required for non-planner model")
                        policy_breach = True

                    allowed_writes, allowed_deletes = _parse_allowlist(lines)
                    if not allowed_writes:
                        errors.append("allowed writes list missing or empty")
                        policy_breach = True

                    receipt_path = _resolve_artifact_path(frontmatter, "receipt_path", default_receipt)
                    report_path = _resolve_artifact_path(frontmatter, "report_path", default_report)

                    if emit_data:
                        default_data = output_dir_path / "DATA.json"
                        data_path_value = data.get("data_path")
                        if data_path_value:
                            data_path = _resolve_repo_path(str(data_path_value))
                            _ensure_allowed_output(data_path)
                        else:
                            data_path = default_data

                    if allowed_writes:
                        if receipt_path and not _is_path_allowed(receipt_path, allowed_writes):
                            errors.append("receipt_path not in allowed writes")
                            policy_breach = True
                        if report_path and not _is_path_allowed(report_path, allowed_writes):
                            errors.append("report_path not in allowed writes")
                            policy_breach = True
                        if data_path and not _is_path_allowed(data_path, allowed_writes):
                            errors.append("data_path not in allowed writes")
                            policy_breach = True

                    depends_on = frontmatter.get("depends_on", [])
                    if depends_on and not manifest_path:
                        errors.append("depends_on present but manifest_path missing")
                        policy_breach = True

        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            policy_breach = True

    lint_output: Optional[Dict[str, Any]] = None
    if not errors:
        lint_output = _run_command(LINT_COMMAND, 120, max_output_bytes)
        lint_info["exit_code"] = lint_output["exit_code"]
        lint_info["stdout"] = lint_output["stdout"]
        lint_info["stderr"] = lint_output["stderr"]
        if lint_output["exit_code"] == 0:
            lint_info["result"] = "PASS"
        elif lint_output["exit_code"] == 2:
            lint_info["result"] = "WARNING"
        else:
            lint_info["result"] = "FAIL"
            verification_failed = True
            errors.append("prompt lint failed")

    if not errors and manifest_path:
        try:
            manifest_file = _resolve_repo_path(str(manifest_path))
            manifest = _load_json(manifest_file)
            task_index = manifest.get("task_index", [])
            receipt_map = {item.get("task_id"): item.get("receipt_path") for item in task_index}
            depends_on = frontmatter.get("depends_on", [])
            for dep in depends_on:
                receipt_rel = receipt_map.get(dep)
                if not receipt_rel:
                    errors.append(f"dependency receipt missing for {dep}")
                    verification_failed = True
                    continue
                receipt_full = _resolve_repo_path(str(receipt_rel))
                if not receipt_full.exists():
                    errors.append(f"dependency receipt not found for {dep}")
                    verification_failed = True
                    continue
                dep_receipt = _load_json(receipt_full)
                if dep_receipt.get("result") != "OK":
                    errors.append(f"dependency receipt not OK for {dep}")
                    verification_failed = True
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"dependency check failed: {exc}")
            verification_failed = True

    commands_run: List[Dict[str, Any]] = []
    if not errors and commands:
        if not isinstance(commands, list):
            errors.append("commands must be a list")
            policy_breach = True
        else:
            for item in commands:
                if not isinstance(item, dict):
                    errors.append("command entry must be an object")
                    policy_breach = True
                    continue
                command = item.get("command")
                if not isinstance(command, str) or not command:
                    errors.append("command is required for each entry")
                    policy_breach = True
                    continue
                timeout_sec = int(item.get("timeout_sec", 60))
                allow_failure = bool(item.get("allow_failure", False))
                try:
                    result = _run_command(command, timeout_sec, max_output_bytes)
                    result["allow_failure"] = allow_failure
                    commands_run.append(result)
                    if result["exit_code"] != 0 and not allow_failure:
                        errors.append(f"command failed: {command}")
                        verification_failed = True
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"command error: {command}: {exc}")
                    verification_failed = True

    status = "success" if not errors else "error"
    result = "OK"
    if blocked_unknown:
        result = "BLOCKED_UNKNOWN"
    elif policy_breach:
        result = "POLICY_BREACH"
    elif verification_failed:
        result = "VERIFICATION_FAILED"
    elif errors:
        result = "INTERNAL_ERROR"

    policy_hash = _sha256_file(POLICY_CANON_PATH) if POLICY_CANON_PATH.exists() else ""
    guide_hash = _sha256_file(GUIDE_CANON_PATH) if GUIDE_CANON_PATH.exists() else ""

    report_body = _build_report(
        status=status,
        prompt_path=prompt_path or "",
        prompt_sha256=prompt_sha256,
        policy_hash=policy_hash,
        guide_hash=guide_hash,
        lint_info=lint_info,
        fill_tokens=fill_me_tokens,
        allowed_writes=allowed_writes,
        allowed_deletes=allowed_deletes,
        commands_run=commands_run,
        errors=errors,
        cortex_index=cortex_index_used,
    )

    max_report_lines = frontmatter.get("max_report_lines") if frontmatter else None
    if isinstance(max_report_lines, int) and max_report_lines > 0:
        report_lines = report_body.splitlines()
        if len(report_lines) > max_report_lines:
            report_body = "\n".join(report_lines[:max_report_lines]) + "\n"

    outputs: List[Dict[str, str]] = []
    if report_path:
        _atomic_write_text(report_path, report_body)
        outputs.append({"path": _normalize_path(str(report_path.relative_to(PROJECT_ROOT))), "sha256": _sha256_file(report_path)})
    if data_path:
        _atomic_write_bytes(data_path, _canonical_json_bytes({"status": "template"}))
        outputs.append({"path": _normalize_path(str(data_path.relative_to(PROJECT_ROOT))), "sha256": _sha256_file(data_path)})

    inputs_list = [
        {
            "path": _normalize_path(prompt_path or ""),
            "sha256": prompt_sha256,
        }
    ]
    if manifest_path:
        manifest_sha = ""
        try:
            manifest_sha = _sha256_file(_resolve_repo_path(str(manifest_path)))
        except (OSError, ValueError):
            manifest_sha = ""
        inputs_list.append(
            {
                "path": _normalize_path(str(manifest_path)),
                "sha256": manifest_sha,
            }
        )

    receipt_payload = {
        "task_id": str(frontmatter.get("task_id", task_id or "")),
        "timestamp_utc": _timestamp_utc(),
        "primary_model": str(frontmatter.get("primary_model", "")),
        "fallback_chain": frontmatter.get("fallback_chain", []),
        "policy_canon_sha256": str(frontmatter.get("policy_canon_sha256", "")),
        "guide_canon_sha256": str(frontmatter.get("guide_canon_sha256", "")),
        "depends_on": frontmatter.get("depends_on", []),
        "receipt_path": _normalize_path(str(receipt_path.relative_to(PROJECT_ROOT))) if receipt_path else "",
        "report_path": _normalize_path(str(report_path.relative_to(PROJECT_ROOT))) if report_path else "",
        "allowed_writes": allowed_writes,
        "allowed_deletes_renames": allowed_deletes,
        "unknowns_or_missing_inputs": fill_me_tokens,
        "commands_run": [item.get("command", "") for item in commands_run],
        "validations": [
            {
                "name": "prompt_lint",
                "command": LINT_COMMAND,
                "exit_code": lint_info.get("exit_code"),
            }
        ],
        "inputs": inputs_list,
        "outputs": outputs,
        "result": result,
        "plan_ref": plan_ref,
        "lint_command": LINT_COMMAND,
        "lint_exit_code": lint_info.get("exit_code"),
        "lint_result": lint_info.get("result"),
    }

    if receipt_path:
        _atomic_write_bytes(receipt_path, _canonical_json_bytes(receipt_payload))

    output = {
        "task_id": task_id,
        "status": status,
        "result": result,
        "prompt_path": prompt_path,
        "prompt_sha256": prompt_sha256,
        "frontmatter": frontmatter,
        "headings": headings,
        "fill_me_tokens": fill_me_tokens,
        "cortex_index": cortex_index_used,
        "output_dir": output_dir,
        "artifacts": {
            "receipt_path": str(receipt_path) if receipt_path else "",
            "report_path": str(report_path) if report_path else "",
            "data_path": str(data_path) if data_path else "",
        },
        "lint": lint_info,
        "commands_run": commands_run,
        "errors": errors,
    }

    try:
        _atomic_write_bytes(output_path, _canonical_json_bytes(output))
    except OSError as exc:
        print(f"ERROR: Failed to write output: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

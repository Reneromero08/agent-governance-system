"""Phase 6B.6 non-hardware qualification contract and CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - older distro jsonschema used by local WSL
    from jsonschema import Draft7Validator as Draft202012Validator  # type: ignore


QUALIFICATION_ROOT = Path(__file__).resolve().parent
PHASE6B6_ROOT = QUALIFICATION_ROOT.parent
FRONTIER_ROOT = PHASE6B6_ROOT.parent
REPO_ROOT = PHASE6B6_ROOT.parents[7]
if str(PHASE6B6_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE6B6_ROOT))

from contracts.contract import AUTHORITY, digest as phase6b6_digest  # noqa: E402
from contracts.schedule import campaign_schedule  # noqa: E402
from contracts.v2_interface import (  # noqa: E402
    PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT,
    QUALIFIED_V2_SOURCE,
    TONE_CODEWORD_TABLE,
)
from runtime.explicit_slot_runtime import AUTHORITY_ERROR as RUNTIME_AUTHORITY_ERROR  # noqa: E402
from runtime.explicit_slot_runtime import run_mock  # noqa: E402


SCHEMA_DIR = QUALIFICATION_ROOT / "schemas"
EXPECTED_REVIEWED_IMPLEMENTATION_HEAD = "e33cb2d4b895746d7ca45e1aa2e6fde673fac20f"
EXPECTED_MERGED_MAIN_HEAD = "d351a62f4f211589d57359d872734757b6e280d9"
SOURCE_REVIEW = 4596915321
QUALIFIED_REVIEWED_SOURCE = "ba48125d15009a044bb869b5716c412b1a8baa1b"
QUALIFICATION_SCHEMA_ID = "CAT_CAS_PHASE6B6_NONHARDWARE_QUALIFICATION_CONTRACT_V1"
REFERENCE_SCHEMA_ID = "CAT_CAS_PHASE6B6_C_REFERENCE_TABLE_V1"
AUTHORITY_FALSE_FIELDS = (
    "hardware_ran",
    "authorization_artifact_created",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
)
AUTHORITY_TRUE_FIELDS = (
    "phase6b6_entry_approved",
    "phase6b6_entered",
    "implementation_authorized",
    "software_qualification_authorized",
    "non_hardware_target_qualification_authorized",
)
FORBIDDEN_AUTHORITY_OPTIONS = {
    "--hardware",
    "--acquire",
    "--calibrate",
    "--run-campaign",
}
STRICT_C_FLAGS = (
    "-std=gnu11",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-ffunction-sections",
    "-fdata-sections",
)


class QualificationError(ValueError):
    """Raised when non-hardware qualification fails closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wsl_path(path: Path) -> str:
    result = subprocess.run(
        ["wsl", "-e", "wslpath", "-a", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise QualificationError((result.stderr or result.stdout or "wslpath failed").strip())
    return result.stdout.strip()


def _wsl_gcc_available() -> bool:
    if os.name != "nt" or shutil.which("wsl") is None:
        return False
    result = subprocess.run(
        ["wsl", "-e", "sh", "-lc", "command -v gcc >/dev/null 2>&1"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def validate_schema(name: str, payload: dict[str, Any]) -> None:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(json.loads(json.dumps(payload)))


def _qualified_v2_source_path() -> Path:
    return FRONTIER_ROOT / QUALIFIED_V2_SOURCE["physical_interface_source_path"].split("/", 1)[1]


def qualification_authority_state() -> dict[str, Any]:
    return {
        "phase6b6_entry_approved": True,
        "phase6b6_entered": True,
        **AUTHORITY,
    }


def _check_authority(authority: dict[str, Any]) -> None:
    for field in AUTHORITY_TRUE_FIELDS:
        if authority.get(field) is not True:
            raise QualificationError(f"authority invariant failed: {field}")
    for field in AUTHORITY_FALSE_FIELDS:
        if authority.get(field) is not False:
            raise QualificationError(f"authority invariant failed: {field}")


@lru_cache(maxsize=1)
def _schedule_and_mock_digests() -> tuple[str, str]:
    schedule = campaign_schedule()
    custody = run_mock(schedule)
    return schedule["schedule_sha256"], custody["custody_sha256"]


def qualification_contract(
    *,
    reviewed_implementation_head: str = EXPECTED_REVIEWED_IMPLEMENTATION_HEAD,
    merged_main_head: str = EXPECTED_MERGED_MAIN_HEAD,
) -> dict[str, Any]:
    if reviewed_implementation_head != EXPECTED_REVIEWED_IMPLEMENTATION_HEAD:
        raise QualificationError("reviewed implementation head mismatch")
    if merged_main_head != EXPECTED_MERGED_MAIN_HEAD:
        raise QualificationError("merged main head mismatch")

    source_path = _qualified_v2_source_path()
    source_sha = file_sha256(source_path)
    if source_sha != QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]:
        raise QualificationError("qualified combined_pdn_hardware.c SHA-256 mismatch")
    if QUALIFIED_V2_SOURCE["reviewed_source"] != QUALIFIED_REVIEWED_SOURCE:
        raise QualificationError("qualified V2 reviewed source mismatch")
    authority_state = qualification_authority_state()
    _check_authority(authority_state)

    schedule_digest, mock_custody_digest = _schedule_and_mock_digests()
    contract = {
        "schema_id": QUALIFICATION_SCHEMA_ID,
        "reviewed_implementation_head": reviewed_implementation_head,
        "merged_main_head": merged_main_head,
        "source_review": SOURCE_REVIEW,
        "qualified_v2_reviewed_source": QUALIFIED_REVIEWED_SOURCE,
        "qualified_v2_source_bundle_digest": QUALIFIED_V2_SOURCE["source_bundle_sha256"],
        "qualified_combined_pdn_hardware_c_path": QUALIFIED_V2_SOURCE["physical_interface_source_path"],
        "qualified_combined_pdn_hardware_c_sha256": source_sha,
        "phase6b6_imported_table_digest": TONE_CODEWORD_TABLE["tone_codeword_table_sha256"],
        "schedule_digest": schedule_digest,
        "mock_custody_digest": mock_custody_digest,
        "pre_acquisition_v2_equivalence_requirement": PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT["schema_id"],
        "authority_state": authority_state,
        "qualification_evidence_created": False,
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
    }
    contract["qualification_contract_sha256"] = digest(contract)
    validate_schema("qualification_contract.schema.json", contract)
    return contract


def compile_reference_emitter(
    output_path: Path,
    *,
    source_path: Path | None = None,
    sanitize: str | None = None,
) -> None:
    source = source_path or _qualified_v2_source_path()
    source_sha = file_sha256(source)
    if source_sha != QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]:
        raise QualificationError("qualified V2 source digest mismatch before compile")

    cc = os.environ.get("CC", "gcc")
    use_wsl_gcc = False
    if shutil.which(cc) is None and _wsl_gcc_available():
        cc = "gcc"
        use_wsl_gcc = True
    if shutil.which(cc) is None and not use_wsl_gcc:
        raise QualificationError(f"C compiler not found: {cc}")

    flags = list(STRICT_C_FLAGS)
    if sanitize == "asan":
        flags.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    elif sanitize == "ubsan":
        flags.extend(["-fsanitize=undefined", "-fno-omit-frame-pointer"])
    elif sanitize is not None:
        raise QualificationError(f"unknown sanitizer: {sanitize}")

    if use_wsl_gcc:
        include_define = f'-DQUALIFIED_V2_SOURCE_PATH="{_wsl_path(source)}"'
        cmd = [
            "wsl",
            "-e",
            cc,
            *flags,
            include_define,
            _wsl_path(QUALIFICATION_ROOT / "emit_v2_reference_table.c"),
            "-lm",
            "-lpthread",
            "-Wl,--gc-sections",
            "-o",
            _wsl_path(output_path),
        ]
    else:
        include_define = f'-DQUALIFIED_V2_SOURCE_PATH="{source.as_posix()}"'
        cmd = [
            cc,
            *flags,
            include_define,
            str(QUALIFICATION_ROOT / "emit_v2_reference_table.c"),
            "-lm",
            "-lpthread",
            "-Wl,--gc-sections",
            "-o",
            str(output_path),
        ]
    result = subprocess.run(cmd, cwd=str(QUALIFICATION_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise QualificationError((result.stderr or result.stdout or "strict C compile failed").strip())


def _run_emitter(executable: Path) -> dict[str, Any]:
    source_sha = QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]
    cmd = [str(executable), source_sha]
    if os.name == "nt":
        try:
            result = subprocess.run(
                cmd,
                cwd=str(QUALIFICATION_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            result = subprocess.run(
                ["wsl", "-e", _wsl_path(executable), source_sha],
                cwd=str(QUALIFICATION_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
    else:
        result = subprocess.run(
            cmd,
            cwd=str(QUALIFICATION_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise QualificationError((result.stderr or result.stdout or "C reference emitter failed").strip())
    payload = json.loads(result.stdout)
    if payload["qualified_source_sha256"] != source_sha:
        raise QualificationError("C reference source SHA-256 mismatch")
    validate_schema("c_reference_table.schema.json", payload)
    payload["reference_table_sha256"] = digest(payload)
    validate_schema("c_reference_table.schema.json", payload)
    return payload


def emit_reference_table(*, sanitize: str | None = None) -> dict[str, Any]:
    qualification_contract()
    with tempfile.TemporaryDirectory(prefix="phase6b6-qualification-") as tmp:
        exe = Path(tmp) / ("emit_v2_reference_table.exe" if os.name == "nt" else "emit_v2_reference_table")
        compile_reference_emitter(exe, sanitize=sanitize)
        first = _run_emitter(exe)
        second = _run_emitter(exe)
        if canonical_json(first) != canonical_json(second):
            raise QualificationError("C reference emission is not deterministic")
        return first


def compare_reference() -> dict[str, Any]:
    try:
        from .compare_v2_reference import compare_reference_tables
    except ImportError:  # pragma: no cover
        from compare_v2_reference import compare_reference_tables  # type: ignore

    c_reference = emit_reference_table()
    result = compare_reference_tables(c_reference, TONE_CODEWORD_TABLE)
    validate_schema("equivalence_result.schema.json", result)
    return result


def validate_only() -> dict[str, Any]:
    contract = qualification_contract()
    reference = emit_reference_table()
    equivalence = compare_reference()
    if equivalence["status"] != "V2_REFERENCE_EQUIVALENCE_PASS":
        raise QualificationError("V2 reference equivalence failed")
    return {
        "schema_id": "CAT_CAS_PHASE6B6_QUALIFICATION_VALIDATE_ONLY_RESULT_V1",
        "status": "PHASE6B6_NONHARDWARE_QUALIFICATION_VALIDATE_ONLY_PASS",
        "qualification_contract_digest": contract["qualification_contract_sha256"],
        "c_reference_table_digest": reference["reference_table_sha256"],
        "imported_python_table_digest": TONE_CODEWORD_TABLE["tone_codeword_table_sha256"],
        "equivalence_result_digest": equivalence["result_sha256"],
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
    }


def _write_or_print(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if out is None:
        print(text, end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    forbidden = sorted(set(args_list) & FORBIDDEN_AUTHORITY_OPTIONS)
    if forbidden:
        print(f"PHASE6B6_NONHARDWARE_AUTHORITY_ERROR: {RUNTIME_AUTHORITY_ERROR}; forbidden option(s): {', '.join(forbidden)}", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Phase 6B.6 non-hardware qualification harness")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract", action="store_true")
    mode.add_argument("--strict-compile", action="store_true")
    mode.add_argument("--emit-reference", action="store_true")
    mode.add_argument("--compare-reference", action="store_true")
    mode.add_argument("--verify-snapshot", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--snapshot-identity", type=Path)
    parser.add_argument("--sanitize", choices=("asan", "ubsan"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(args_list)

    try:
        if args.contract:
            _write_or_print(qualification_contract(), args.out)
        elif args.strict_compile:
            with tempfile.TemporaryDirectory(prefix="phase6b6-qualification-") as tmp:
                exe = Path(tmp) / ("emit_v2_reference_table.exe" if os.name == "nt" else "emit_v2_reference_table")
                compile_reference_emitter(exe, sanitize=args.sanitize)
            print("PHASE6B6_REFERENCE_STRICT_C_COMPILE_OK")
        elif args.emit_reference:
            _write_or_print(emit_reference_table(sanitize=args.sanitize), args.out)
        elif args.compare_reference:
            result = compare_reference()
            _write_or_print(result, args.out)
        elif args.verify_snapshot:
            if args.snapshot_dir is None or args.snapshot_identity is None:
                raise QualificationError("--verify-snapshot requires --snapshot-dir and --snapshot-identity")
            try:
                from .verify_sealed_snapshot import verify_snapshot_identity
            except ImportError:  # pragma: no cover
                from verify_sealed_snapshot import verify_snapshot_identity  # type: ignore

            _write_or_print(verify_snapshot_identity(args.snapshot_dir, _load_json(args.snapshot_identity)), args.out)
        elif args.validate_only:
            _write_or_print(validate_only(), args.out)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

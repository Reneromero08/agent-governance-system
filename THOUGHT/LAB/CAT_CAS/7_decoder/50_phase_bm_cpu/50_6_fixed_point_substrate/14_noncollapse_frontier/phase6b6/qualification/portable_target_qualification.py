"""Copy-only Phase 6B.6 target qualification runner.

This entry point is intentionally standalone. It uses only the Python standard
library, GCC/libc through subprocess, and ordinary filesystem operations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

MANIFEST_SCHEMA_ID = "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_MANIFEST_V1"
BINDING_SCHEMA_ID = "CAT_CAS_PHASE6B6_TRUSTED_SNAPSHOT_BINDING_V1"
CONTRACT_SCHEMA_ID = "CAT_CAS_PHASE6B6_NONHARDWARE_QUALIFICATION_CONTRACT_V1"
C_REFERENCE_SCHEMA_ID = "CAT_CAS_PHASE6B6_C_REFERENCE_TABLE_V1"
RESULT_SCHEMA_ID = "CAT_CAS_PHASE6B6_PORTABLE_TARGET_QUALIFICATION_RESULT_V1"

BASE_QUALIFICATION_REVIEWED_HEAD = "5ad5b5f07bd31e368de56ab3c721f20498fb7aa1"
BASE_QUALIFICATION_MERGE = "3c6a5dd344a58d729ea84d23cc90e9e34d6f8336"
SNAPSHOT_SUBJECT_COMMIT = "d351a62f4f211589d57359d872734757b6e280d9"
SNAPSHOT_SUBJECT_TREE = "1a927b20cb2d712a7220a823621c8fc83cbc984d"
EXPECTED_INVENTORY_SHA256 = "e47dea4c3467835a425d9d553803da48f672a8799970db4fc9b83e98596f50d8"
EXPECTED_SCOPED_TREE = "408ee35257417898a992510b0f260602117a15af"
EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256 = "24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5"
EXPECTED_V2_SOURCE_SHA256 = "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976"
QUALIFICATION_CONTRACT_DIGEST = "986d1eb27e6e715da0ed8765f58566b0608e464b94dbd0d58ab3d130d80fd0d2"
SCHEDULE_DIGEST = "c632d59934c2610541e279cac3a48202f2c0a79bb734e995f2cc4f28d19e87d3"
MOCK_CUSTODY_DIGEST = "4c0a58772fd25fe77759d6d09089ad532a09b3a5adcdce01dc099b6b7b00dba1"
PHASE6B6_IMPORTED_TABLE_DIGEST = "b8af91cfa7bb6ae5dbd6bb2283d4612a71ba432d28a016cb3d662bf514d503cf"
QUALIFIED_V2_SOURCE_BUNDLE_DIGEST = "bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f"
REVIEWED_IMPLEMENTATION_HEAD = "e33cb2d4b895746d7ca45e1aa2e6fde673fac20f"
QUALIFIED_V2_REVIEWED_SOURCE = "ba48125d15009a044bb869b5716c412b1a8baa1b"
SOURCE_REVIEW = 4596915321
SNAPSHOT_SUBJECT_KIND = "PHASE6B6_SOFTWARE_IMPLEMENTATION_ONLY"
PRE_ACQUISITION_REQUIREMENT = "CAT_CAS_PHASE6B6_PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT_V1"
PACKAGE_ROOT_DIR = "phase6b6_portable_target_package"

PHASE6B6_RELATIVE_ROOT = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6"
)
V2_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.c"
)
V2_SOURCE_CONTRACT_PATH = "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.c"
V2_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.h"
)
CAPTURED_FILE_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/captured_file.h"
)
CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/capture_quality_contract.h"
)
PORTABLE_TARGET_RUNNER_SOURCE = f"{PHASE6B6_RELATIVE_ROOT}/qualification/portable_target_qualification.py"
C_REFERENCE_EMITTER_SOURCE = f"{PHASE6B6_RELATIVE_ROOT}/qualification/emit_v2_reference_table.c"
SNAPSHOT_SCOPE = [PHASE6B6_RELATIVE_ROOT, V2_RELATIVE_SOURCE]
REQUIRED_SNAPSHOT_PATHS = {
    V2_RELATIVE_SOURCE,
    f"{PHASE6B6_RELATIVE_ROOT}/contracts/v2_interface.py",
    f"{PHASE6B6_RELATIVE_ROOT}/contracts/contract.py",
    f"{PHASE6B6_RELATIVE_ROOT}/contracts/schedule.py",
    f"{PHASE6B6_RELATIVE_ROOT}/PHASE6B6_SOFTWARE_ENTRY_APPROVAL.json",
}
QUALIFIED_V2_SUPPORT_HEADERS = {
    "combined_pdn_hardware.h": V2_HEADER_RELATIVE_SOURCE,
    "captured_file.h": CAPTURED_FILE_HEADER_RELATIVE_SOURCE,
    "capture_quality_contract.h": CAPTURE_QUALITY_HEADER_RELATIVE_SOURCE,
}
ROLE_COUNTS = {
    "phase6b6_scoped_source": 29,
    "qualified_v2_source": 1,
    "qualified_v2_support_header": 3,
    "portable_target_runner": 1,
    "c_reference_emitter": 1,
}
COPIED_FILE_COUNT = sum(ROLE_COUNTS.values())
SNAPSHOT_FILE_COUNT = 30
ALLOWED_ROLES = set(ROLE_COUNTS)
PORTABLE_QUALIFICATION_SCOPE = [
    "portable package manifest verification",
    "copied-file inventory verification",
    "pure-Python Git blob and scoped-tree reconstruction",
    "qualification contract verification",
    "strict C reference-emitter compile",
    "deterministic raw C reference emission A and B",
    "byte comparison",
    "C versus Python tone-codeword equivalence",
    "ASan reference-emitter execution",
    "UBSan reference-emitter execution",
    "runtime validate-only",
    "hardware-option rejection",
    "sender-process absence check",
    "final result validation",
]

AUTHORITY_KEYS = {
    "schema_id",
    "architecture_review",
    "software_entry_review",
    "project_owner_decision",
    "implementation_authorized",
    "software_qualification_authorized",
    "non_hardware_target_qualification_authorized",
    "hardware_ran",
    "authorization_artifact_created",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "automatic_retry",
    "phase6b6_entry_approved",
    "phase6b6_entered",
}
AUTHORITY_FALSE_FIELDS = (
    "hardware_ran",
    "authorization_artifact_created",
    "calibration_authorized",
    "scientific_acquisition_authorized",
    "restoration_authorized",
    "target_coupling_authorized",
    "small_wall_authorized",
    "automatic_retry",
)
AUTHORITY_TRUE_FIELDS = (
    "phase6b6_entry_approved",
    "phase6b6_entered",
    "implementation_authorized",
    "software_qualification_authorized",
    "non_hardware_target_qualification_authorized",
)
STRICT_C_FLAGS = (
    "-std=gnu11",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-ffunction-sections",
    "-fdata-sections",
)
MODE_NAMES = ("basis", "rotation", "residual", "mini")
TONE_ABS_TOLERANCE_HZ = 1e-9
HARDWARE_OPTIONS = ("--hardware", "--acquire", "--calibrate", "--run-campaign")
FORBIDDEN_PROCESS_PATTERNS = (
    "combined_pdn_runner",
    "run_combined_campaign",
    "explicit_slot_runtime --hardware",
    "wrmsr",
    "rdmsr",
    "cpupower",
    "turbostat",
    "portable_target_qualification_forbidden_probe",
)


class PortableQualificationError(ValueError):
    """Raised when copied-file target qualification fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PortableQualificationError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise PortableQualificationError(f"non-finite JSON constant rejected: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise PortableQualificationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PortableQualificationError(f"JSON object required: {path}")
    return payload


def parse_json_bytes(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableQualificationError("invalid C reference JSON bytes") from exc
    if not isinstance(payload, dict):
        raise PortableQualificationError("C reference JSON object required")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(ch in "0123456789abcdef" for ch in value)


def _expect_keys(name: str, obj: dict[str, Any], keys: set[str]) -> None:
    observed = set(obj)
    if observed != keys:
        raise PortableQualificationError(f"{name} keys mismatch missing={sorted(keys - observed)} extra={sorted(observed - keys)}")


def _reject_bad_path(path: str) -> None:
    if not isinstance(path, str) or path == "" or path.startswith("/") or "\\" in path:
        raise PortableQualificationError(f"unsafe path rejected: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise PortableQualificationError(f"unsafe path rejected: {path!r}")


def _entry_mode(path: Path) -> str:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise PortableQualificationError(f"regular file required: {path}")
    return "100755" if (st.st_mode & stat.S_IXUSR) else "100644"


def _scan_files(root: Path) -> list[str]:
    paths: list[str] = []
    for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        rel_base = "" if base_path == root else base_path.relative_to(root).as_posix()
        for dirname in list(dirs):
            rel = f"{rel_base}/{dirname}" if rel_base else dirname
            if dirname == ".git" or rel.endswith(".git") or ".git/" in rel:
                raise PortableQualificationError(f"forbidden .git path present: {rel}")
            st = (base_path / dirname).lstat()
            if not stat.S_ISDIR(st.st_mode):
                raise PortableQualificationError(f"non-directory traversal entry rejected: {rel}")
            _reject_bad_path(rel)
            paths.append(rel)
        for filename in files:
            rel = f"{rel_base}/{filename}" if rel_base else filename
            _reject_bad_path(rel)
            if rel.endswith(".bundle") or rel == ".git" or "/.git/" in f"/{rel}/":
                raise PortableQualificationError(f"forbidden Git content present: {rel}")
            st = (base_path / filename).lstat()
            if not stat.S_ISREG(st.st_mode):
                raise PortableQualificationError(f"non-regular file rejected: {rel}")
            paths.append(rel)
    seen: dict[str, str] = {}
    for path in paths:
        key = path.casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise PortableQualificationError(f"case-colliding paths: {previous} and {path}")
        seen[key] = path
    return sorted(path for path in paths if (root / path).is_file())


def _read_manifest_sha(root: Path) -> str:
    text = (root / "PORTABLE_PACKAGE_MANIFEST.sha256").read_text(encoding="ascii")
    parts = text.strip().split()
    if len(parts) != 2 or parts[1] != "PORTABLE_PACKAGE_MANIFEST.json" or not _hex(parts[0], 64):
        raise PortableQualificationError("malformed PORTABLE_PACKAGE_MANIFEST.sha256")
    observed = file_sha256(root / "PORTABLE_PACKAGE_MANIFEST.json")
    if observed != parts[0]:
        raise PortableQualificationError("portable manifest SHA-256 mismatch")
    return observed


def _validate_copied_record(item: dict[str, Any], manifest: dict[str, Any]) -> None:
    _expect_keys(
        "copied file",
        item,
        {"path", "source_path", "source_commit", "source_tree", "source_blob_sha1", "content_sha256", "mode", "size", "role"},
    )
    path = item["path"]
    source_path = item["source_path"]
    _reject_bad_path(path)
    _reject_bad_path(source_path)
    if item["role"] not in ALLOWED_ROLES:
        raise PortableQualificationError(f"unknown copied file role: {item['role']}")
    if item["mode"] not in ("100644", "100755"):
        raise PortableQualificationError(f"bad copied file mode: {path}")
    if not isinstance(item["size"], int) or item["size"] < 0:
        raise PortableQualificationError(f"bad copied file size: {path}")
    if not _hex(item["content_sha256"], 64) or not _hex(item["source_blob_sha1"], 40):
        raise PortableQualificationError(f"bad copied file digest: {path}")
    if not _hex(item["source_commit"], 40) or not _hex(item["source_tree"], 40):
        raise PortableQualificationError(f"bad copied file source identity: {path}")

    role = item["role"]
    if role == "portable_target_runner":
        if path != "portable_target_qualification.py" or source_path != PORTABLE_TARGET_RUNNER_SOURCE:
            raise PortableQualificationError("portable target runner path/source mismatch")
        if item["source_commit"] != manifest["portable_export_commit"] or item["source_tree"] != manifest["portable_export_tree"]:
            raise PortableQualificationError("portable target runner export identity mismatch")
    elif role == "c_reference_emitter":
        if path != "emit_v2_reference_table.c" or source_path != C_REFERENCE_EMITTER_SOURCE:
            raise PortableQualificationError("C reference emitter path/source mismatch")
        if item["source_commit"] != manifest["portable_export_commit"] or item["source_tree"] != manifest["portable_export_tree"]:
            raise PortableQualificationError("C reference emitter export identity mismatch")
    elif role == "qualified_v2_support_header":
        if QUALIFIED_V2_SUPPORT_HEADERS.get(path) != source_path:
            raise PortableQualificationError("qualified V2 support header path/source mismatch")
        if item["source_commit"] != SNAPSHOT_SUBJECT_COMMIT or item["source_tree"] != manifest["snapshot_subject_tree"]:
            raise PortableQualificationError("qualified V2 support header snapshot identity mismatch")
    elif role == "qualified_v2_source":
        if path != f"snapshot/{V2_RELATIVE_SOURCE}" or source_path != V2_RELATIVE_SOURCE:
            raise PortableQualificationError("qualified V2 source path mismatch")
        if item["source_commit"] != SNAPSHOT_SUBJECT_COMMIT or item["source_tree"] != manifest["snapshot_subject_tree"]:
            raise PortableQualificationError("qualified V2 source snapshot identity mismatch")
    elif role == "phase6b6_scoped_source":
        if not path.startswith("snapshot/") or not source_path.startswith(f"{PHASE6B6_RELATIVE_ROOT}/"):
            raise PortableQualificationError("Phase 6B.6 scoped source path mismatch")
        if item["source_commit"] != SNAPSHOT_SUBJECT_COMMIT or item["source_tree"] != manifest["snapshot_subject_tree"]:
            raise PortableQualificationError("Phase 6B.6 scoped source snapshot identity mismatch")


def validate_manifest(manifest: dict[str, Any]) -> None:
    _expect_keys(
        "portable manifest",
        manifest,
        {
            "schema_id",
            "format_version",
            "package_root",
            "base_qualification_reviewed_head",
            "base_qualification_merge",
            "portable_export_commit",
            "portable_export_tree",
            "portable_support_blob_bindings",
            "snapshot_subject_commit",
            "snapshot_subject_tree",
            "expected_scoped_tree",
            "expected_inventory_sha256",
            "expected_phase6b6_subtree_inventory_sha256",
            "qualified_v2_source_sha256",
            "target_executes_git",
            "target_requires_jsonschema",
            "target_requires_repository_history",
            "copied_files",
            "copied_file_count",
            "snapshot_file_count",
            "portable_qualification_scope",
        },
    )
    exact_values = {
        "schema_id": MANIFEST_SCHEMA_ID,
        "format_version": 1,
        "package_root": PACKAGE_ROOT_DIR,
        "base_qualification_reviewed_head": BASE_QUALIFICATION_REVIEWED_HEAD,
        "base_qualification_merge": BASE_QUALIFICATION_MERGE,
        "snapshot_subject_commit": SNAPSHOT_SUBJECT_COMMIT,
        "snapshot_subject_tree": SNAPSHOT_SUBJECT_TREE,
        "expected_scoped_tree": EXPECTED_SCOPED_TREE,
        "expected_inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "expected_phase6b6_subtree_inventory_sha256": EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256,
        "qualified_v2_source_sha256": EXPECTED_V2_SOURCE_SHA256,
        "target_executes_git": False,
        "target_requires_jsonschema": False,
        "target_requires_repository_history": False,
        "copied_file_count": COPIED_FILE_COUNT,
        "snapshot_file_count": SNAPSHOT_FILE_COUNT,
        "portable_qualification_scope": PORTABLE_QUALIFICATION_SCOPE,
    }
    for key, expected in exact_values.items():
        if manifest[key] != expected:
            raise PortableQualificationError(f"portable manifest value mismatch: {key}")
    if not _hex(manifest["portable_export_commit"], 40) or not _hex(manifest["portable_export_tree"], 40):
        raise PortableQualificationError("portable export identity is not lowercase hex")
    if not isinstance(manifest["copied_files"], list) or len(manifest["copied_files"]) != manifest["copied_file_count"]:
        raise PortableQualificationError("portable manifest copied file count mismatch")
    if not isinstance(manifest["portable_support_blob_bindings"], list):
        raise PortableQualificationError("portable support blob bindings must be a list")

    paths: set[str] = set()
    source_paths: set[str] = set()
    counts = {role: 0 for role in ROLE_COUNTS}
    for item in manifest["copied_files"]:
        if not isinstance(item, dict):
            raise PortableQualificationError("copied file entry must be an object")
        _validate_copied_record(item, manifest)
        if item["path"] in paths:
            raise PortableQualificationError(f"duplicate copied file path: {item['path']}")
        if item["source_path"] in source_paths:
            raise PortableQualificationError(f"duplicate copied source path: {item['source_path']}")
        paths.add(item["path"])
        source_paths.add(item["source_path"])
        counts[item["role"]] += 1
    if counts != ROLE_COUNTS:
        raise PortableQualificationError(f"copied file role counts mismatch: {counts}")
    expected_support = [
        item for item in sorted(manifest["copied_files"], key=lambda entry: entry["path"])
        if item["role"] in ("c_reference_emitter", "portable_target_runner")
    ]
    if manifest["portable_support_blob_bindings"] != expected_support:
        raise PortableQualificationError("portable support blob bindings mismatch")


def validate_inventory_entry(entry: dict[str, Any]) -> None:
    _expect_keys("trusted inventory entry", entry, {"path", "mode", "git_object_type", "git_object_sha", "sha256", "size"})
    _reject_bad_path(entry["path"])
    if entry["mode"] not in ("100644", "100755") or entry["git_object_type"] != "blob":
        raise PortableQualificationError("trusted inventory mode/object mismatch")
    if not _hex(entry["git_object_sha"], 40) or not _hex(entry["sha256"], 64):
        raise PortableQualificationError("trusted inventory digest mismatch")
    if not isinstance(entry["size"], int) or entry["size"] < 0:
        raise PortableQualificationError("trusted inventory size mismatch")


def validate_binding(binding: dict[str, Any]) -> None:
    _expect_keys(
        "trusted binding",
        binding,
        {
            "schema_id",
            "snapshot_subject_kind",
            "snapshot_subject_commit",
            "snapshot_subject_tree",
            "snapshot_scope",
            "qualification_harness_source_equals_snapshot_subject",
            "tracked_file_count",
            "phase6b6_tracked_file_count",
            "path_mode_blob_inventory",
            "expected_inventory_sha256",
            "expected_phase6b6_subtree_inventory_sha256",
            "expected_scoped_tree",
        },
    )
    exact = {
        "schema_id": BINDING_SCHEMA_ID,
        "snapshot_subject_kind": SNAPSHOT_SUBJECT_KIND,
        "snapshot_subject_commit": SNAPSHOT_SUBJECT_COMMIT,
        "snapshot_subject_tree": SNAPSHOT_SUBJECT_TREE,
        "snapshot_scope": SNAPSHOT_SCOPE,
        "qualification_harness_source_equals_snapshot_subject": False,
        "tracked_file_count": SNAPSHOT_FILE_COUNT,
        "phase6b6_tracked_file_count": 29,
        "expected_inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "expected_phase6b6_subtree_inventory_sha256": EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256,
        "expected_scoped_tree": EXPECTED_SCOPED_TREE,
    }
    for key, expected in exact.items():
        if binding[key] != expected:
            raise PortableQualificationError(f"trusted binding value mismatch: {key}")
    inventory = binding["path_mode_blob_inventory"]
    if not isinstance(inventory, list) or len(inventory) != SNAPSHOT_FILE_COUNT:
        raise PortableQualificationError("trusted binding inventory count mismatch")
    paths: set[str] = set()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise PortableQualificationError("trusted inventory entry must be an object")
        validate_inventory_entry(entry)
        if entry["path"] in paths:
            raise PortableQualificationError(f"duplicate trusted inventory path: {entry['path']}")
        paths.add(entry["path"])
    if not REQUIRED_SNAPSHOT_PATHS.issubset(paths):
        raise PortableQualificationError("trusted binding required path set mismatch")
    sorted_inventory = sorted(inventory, key=lambda item: item["path"])
    if inventory != sorted_inventory:
        raise PortableQualificationError("trusted inventory must be path sorted")
    phase_entries = [entry for entry in inventory if entry["path"].startswith(f"{PHASE6B6_RELATIVE_ROOT}/")]
    if digest(inventory) != EXPECTED_INVENTORY_SHA256:
        raise PortableQualificationError("trusted binding inventory digest recomputation mismatch")
    if digest(phase_entries) != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortableQualificationError("trusted binding Phase 6B.6 digest recomputation mismatch")


def validate_authority(authority: dict[str, Any]) -> None:
    _expect_keys("authority state", authority, AUTHORITY_KEYS)
    exact = {
        "schema_id": "CAT_CAS_PHASE6B6_SOFTWARE_ENTRY_RUNTIME_AUTHORITY_V1",
        "architecture_review": 4588082595,
        "software_entry_review": 4588098104,
        "project_owner_decision": "APPROVE_PHASE6B6_SOFTWARE_ENTRY_ONLY",
    }
    for key, expected in exact.items():
        if authority[key] != expected:
            raise PortableQualificationError(f"authority exact value mismatch: {key}")
    for field in AUTHORITY_TRUE_FIELDS:
        if authority[field] is not True:
            raise PortableQualificationError(f"authority true field mismatch: {field}")
    for field in AUTHORITY_FALSE_FIELDS:
        if authority[field] is not False:
            raise PortableQualificationError(f"authority false field mismatch: {field}")


def validate_contract(contract: dict[str, Any]) -> None:
    _expect_keys(
        "qualification contract",
        contract,
        {
            "schema_id",
            "reviewed_implementation_head",
            "merged_main_head",
            "source_review",
            "qualified_v2_reviewed_source",
            "qualified_v2_source_bundle_digest",
            "qualified_combined_pdn_hardware_c_path",
            "qualified_combined_pdn_hardware_c_sha256",
            "phase6b6_imported_table_digest",
            "schedule_digest",
            "mock_custody_digest",
            "snapshot_subject_kind",
            "snapshot_subject_commit",
            "snapshot_subject_tree",
            "snapshot_scope",
            "qualification_harness_source_equals_snapshot_subject",
            "expected_inventory_sha256",
            "expected_phase6b6_subtree_inventory_sha256",
            "pre_acquisition_v2_equivalence_requirement",
            "authority_state",
            "qualification_evidence_created",
            "hardware_ran",
            "scientific_acquisition_authorized",
            "qualification_contract_sha256",
        },
    )
    exact = {
        "schema_id": CONTRACT_SCHEMA_ID,
        "reviewed_implementation_head": REVIEWED_IMPLEMENTATION_HEAD,
        "merged_main_head": SNAPSHOT_SUBJECT_COMMIT,
        "source_review": SOURCE_REVIEW,
        "qualified_v2_reviewed_source": QUALIFIED_V2_REVIEWED_SOURCE,
        "qualified_v2_source_bundle_digest": QUALIFIED_V2_SOURCE_BUNDLE_DIGEST,
        "qualified_combined_pdn_hardware_c_path": V2_SOURCE_CONTRACT_PATH,
        "qualified_combined_pdn_hardware_c_sha256": EXPECTED_V2_SOURCE_SHA256,
        "phase6b6_imported_table_digest": PHASE6B6_IMPORTED_TABLE_DIGEST,
        "schedule_digest": SCHEDULE_DIGEST,
        "mock_custody_digest": MOCK_CUSTODY_DIGEST,
        "snapshot_subject_kind": SNAPSHOT_SUBJECT_KIND,
        "snapshot_subject_commit": SNAPSHOT_SUBJECT_COMMIT,
        "snapshot_subject_tree": SNAPSHOT_SUBJECT_TREE,
        "snapshot_scope": SNAPSHOT_SCOPE,
        "qualification_harness_source_equals_snapshot_subject": False,
        "expected_inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "expected_phase6b6_subtree_inventory_sha256": EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256,
        "pre_acquisition_v2_equivalence_requirement": PRE_ACQUISITION_REQUIREMENT,
        "qualification_evidence_created": False,
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
        "qualification_contract_sha256": QUALIFICATION_CONTRACT_DIGEST,
    }
    for key, expected in exact.items():
        if contract[key] != expected:
            raise PortableQualificationError(f"qualification contract exact value mismatch: {key}")
    if not isinstance(contract["authority_state"], dict):
        raise PortableQualificationError("qualification contract authority_state must be object")
    validate_authority(contract["authority_state"])
    unsigned = dict(contract)
    observed_digest = unsigned.pop("qualification_contract_sha256")
    if digest(unsigned) != observed_digest:
        raise PortableQualificationError("qualification contract digest recomputation mismatch")


def _manifest_files(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in manifest["copied_files"]:
        path = item["path"]
        if path in records:
            raise PortableQualificationError(f"duplicate copied file path: {path}")
        records[path] = item
    return records


def verify_copied_files(root: Path, manifest: dict[str, Any]) -> None:
    records = _manifest_files(manifest)
    actual_files = set(_scan_files(root))
    expected_files = set(records) | {
        "PORTABLE_PACKAGE_MANIFEST.json",
        "PORTABLE_PACKAGE_MANIFEST.sha256",
        "TRUSTED_SNAPSHOT_BINDING.json",
        "QUALIFICATION_CONTRACT.json",
    }
    if actual_files != expected_files:
        raise PortableQualificationError(
            f"portable package file set mismatch missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}"
        )
    for rel, record in records.items():
        path = root / rel
        data = path.read_bytes()
        if len(data) != record["size"]:
            raise PortableQualificationError(f"copied file size mismatch: {rel}")
        if file_sha256(path) != record["content_sha256"]:
            raise PortableQualificationError(f"copied file SHA-256 mismatch: {rel}")
        if blob_sha1(data) != record["source_blob_sha1"]:
            raise PortableQualificationError(f"copied file Git blob SHA-1 mismatch: {rel}")
        if _entry_mode(path) != record["mode"]:
            raise PortableQualificationError(f"copied file mode mismatch: {rel}")


def _tree_sha(entries: list[dict[str, Any]], prefix: str = "") -> str:
    children: dict[str, list[dict[str, Any]]] = {}
    files: list[dict[str, Any]] = []
    for entry in entries:
        path = entry["path"][len(prefix) :] if prefix else entry["path"]
        first, _, rest = path.partition("/")
        if rest:
            children.setdefault(first, []).append(entry)
        else:
            files.append(entry)
    records: list[tuple[bytes, bytes, bytes]] = []
    for item in files:
        basename = item["path"].split("/")[-1].encode("utf-8")
        records.append((basename, item["mode"].encode("ascii") + b" " + basename + b"\0", bytes.fromhex(item["git_object_sha"])))
    for dirname, nested in children.items():
        child_prefix = f"{prefix}{dirname}/"
        child_sha = _tree_sha(nested, child_prefix)
        name = dirname.encode("utf-8")
        records.append((name, b"40000 " + name + b"\0", bytes.fromhex(child_sha)))
    payload = b"".join(header + raw for _, header, raw in sorted(records, key=lambda record: record[0]))
    return hashlib.sha1(b"tree " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def verify_snapshot(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    for entry in binding["path_mode_blob_inventory"]:
        rel = entry["path"]
        path = root / "snapshot" / rel
        if not path.is_file():
            raise PortableQualificationError(f"missing copied snapshot file: {rel}")
        data = path.read_bytes()
        observed_entry = {
            "path": rel,
            "mode": _entry_mode(path),
            "git_object_type": "blob",
            "git_object_sha": blob_sha1(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        validate_inventory_entry(observed_entry)
        observed.append(observed_entry)
    observed.sort(key=lambda item: item["path"])
    expected = sorted(binding["path_mode_blob_inventory"], key=lambda item: item["path"])
    if observed != expected:
        raise PortableQualificationError("copied snapshot inventory mismatch")
    inventory_sha = digest(observed)
    phase_entries = [entry for entry in observed if entry["path"].startswith(f"{PHASE6B6_RELATIVE_ROOT}/")]
    phase_digest = digest(phase_entries)
    scoped_tree = _tree_sha(observed)
    v2_entry = next(item for item in observed if item["path"] == V2_RELATIVE_SOURCE)
    if inventory_sha != EXPECTED_INVENTORY_SHA256 or inventory_sha != binding["expected_inventory_sha256"]:
        raise PortableQualificationError("observed inventory SHA-256 mismatch")
    if scoped_tree != EXPECTED_SCOPED_TREE or scoped_tree != binding["expected_scoped_tree"]:
        raise PortableQualificationError("observed scoped tree SHA-1 mismatch")
    if phase_digest != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortableQualificationError("observed Phase 6B.6 subtree inventory mismatch")
    if v2_entry["sha256"] != EXPECTED_V2_SOURCE_SHA256:
        raise PortableQualificationError("observed V2 source SHA-256 mismatch")
    return {
        "observed_inventory_sha256": inventory_sha,
        "calculated_scoped_tree": scoped_tree,
        "calculated_phase6b6_subtree_inventory_sha256": phase_digest,
        "calculated_v2_source_sha256": v2_entry["sha256"],
    }


def _u64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _code_rand(state: int) -> tuple[int, int]:
    x = state
    x = _u64(x ^ _u64(x << 13))
    x = _u64(x ^ (x >> 7))
    x = _u64(x ^ _u64(x << 17))
    return x, x


def codebook() -> dict[str, list[int]]:
    weights = (4, 5, 6, 7)
    state = 0x243F6A8885A308D3 ^ 7
    best: list[list[int]] | None = None
    best_distance = -1
    for _ in range(4000):
        candidate: list[list[int]] = []
        for weight in weights:
            row = [1] * 12
            pool = list(range(12))
            for i in range(weight):
                state, rnd = _code_rand(state)
                j = i + int(rnd % (12 - i))
                pool[i], pool[j] = pool[j], pool[i]
                row[pool[i]] = -1
            candidate.append(row)
        distance = 99
        for i in range(4):
            for j in range(i + 1, 4):
                hamming = 0
                for k in range(12):
                    if candidate[i][k] != candidate[j][k]:
                        hamming += 1
                distance = min(distance, hamming)
        if distance > best_distance:
            best_distance = distance
            best = [row[:] for row in candidate]
    if best is None:
        raise PortableQualificationError("portable codebook generation failed")
    return {name: list(best[index]) for index, name in enumerate(MODE_NAMES)}


def tone_hz(index: int) -> float:
    low = math.log(20.0)
    high = math.log(1500.0)
    x = index / 11.0
    return math.exp(low + (high - low) * x) * (1.0 + 0.013 * math.sin(2.399963 * (index + 1)))


def python_reference_table() -> dict[str, Any]:
    book = codebook()
    return {
        "schema_id": C_REFERENCE_SCHEMA_ID,
        "format_version": 1,
        "qualified_source_sha256": EXPECTED_V2_SOURCE_SHA256,
        "tone_count": 12,
        "mode_count": 4,
        "mode_names": list(MODE_NAMES),
        "mode_to_codeword_mapping": {mode: index for index, mode in enumerate(MODE_NAMES)},
        "tones": [
            {"physical_tone_index": index, "frequency_hz": tone_hz(index), "codeword_source_index": index}
            for index in range(12)
        ],
        "codebook": book,
        "codebook_rows": [{"mode": mode, "row": list(book[mode])} for mode in MODE_NAMES],
    }


def validate_c_reference(payload: dict[str, Any]) -> None:
    _expect_keys(
        "C reference",
        payload,
        {
            "schema_id",
            "format_version",
            "qualified_source_sha256",
            "tone_count",
            "mode_count",
            "mode_names",
            "mode_to_codeword_mapping",
            "tones",
            "codebook",
            "codebook_rows",
        },
    )
    if payload["schema_id"] != C_REFERENCE_SCHEMA_ID or payload["format_version"] != 1:
        raise PortableQualificationError("C reference identity mismatch")
    if payload["qualified_source_sha256"] != EXPECTED_V2_SOURCE_SHA256:
        raise PortableQualificationError("C reference source SHA mismatch")
    if payload["tone_count"] != 12 or payload["mode_count"] != 4:
        raise PortableQualificationError("C reference geometry mismatch")
    if payload["mode_names"] != list(MODE_NAMES):
        raise PortableQualificationError("C reference mode_names mismatch")
    if payload["mode_to_codeword_mapping"] != {mode: index for index, mode in enumerate(MODE_NAMES)}:
        raise PortableQualificationError("C reference mode mapping mismatch")
    tones = payload["tones"]
    if not isinstance(tones, list) or len(tones) != 12:
        raise PortableQualificationError("C reference tone count mismatch")
    for index in range(12):
        tone = tones[index]
        if not isinstance(tone, dict):
            raise PortableQualificationError("C reference tone must be object")
        _expect_keys("C reference tone", tone, {"physical_tone_index", "frequency_hz", "codeword_source_index"})
        if tone["physical_tone_index"] != index or tone["codeword_source_index"] != index:
            raise PortableQualificationError("C reference tone index mismatch")
        if not isinstance(tone["frequency_hz"], (int, float)) or not math.isfinite(float(tone["frequency_hz"])):
            raise PortableQualificationError("C reference tone frequency must be finite")
    code = payload["codebook"]
    if not isinstance(code, dict) or set(code) != set(MODE_NAMES):
        raise PortableQualificationError("C reference codebook keys mismatch")
    for mode in MODE_NAMES:
        row = code[mode]
        if not isinstance(row, list) or len(row) != 12:
            raise PortableQualificationError("C reference codebook row length mismatch")
        for sign in row:
            if sign not in (-1, 1):
                raise PortableQualificationError("C reference codebook sign mismatch")
    rows = payload["codebook_rows"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise PortableQualificationError("C reference codebook_rows length mismatch")
    for index in range(4):
        row = rows[index]
        if not isinstance(row, dict):
            raise PortableQualificationError("C reference codebook row must be object")
        _expect_keys("C reference codebook row", row, {"mode", "row"})
        mode = MODE_NAMES[index]
        if row["mode"] != mode or row["row"] != code[mode]:
            raise PortableQualificationError("C reference codebook_rows structure mismatch")


def _temp_parent() -> Path | None:
    value = os.environ.get("PHASE6B6_PORTABLE_C_TEMP_ROOT")
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compile_and_emit(root: Path, sanitize: str | None = None) -> dict[str, Any]:
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        raise PortableQualificationError(f"C compiler not found: {cc}")
    source = root / "emit_v2_reference_table.c"
    v2_source = root / "snapshot" / V2_RELATIVE_SOURCE
    flags = list(STRICT_C_FLAGS)
    label = sanitize or "strict"
    if sanitize == "asan":
        flags.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    elif sanitize == "ubsan":
        flags.extend(["-fsanitize=undefined", "-fno-omit-frame-pointer"])
    elif sanitize is not None:
        raise PortableQualificationError(f"unknown sanitizer: {sanitize}")
    parent = _temp_parent()
    before = set(parent.iterdir()) if parent is not None else None
    with tempfile.TemporaryDirectory(prefix=f"phase6b6-portable-c-{label}-", dir=str(parent) if parent else None) as tmp:
        temp_dir = Path(tmp)
        exe = temp_dir / ("emit_v2_reference_table.exe" if os.name == "nt" else "emit_v2_reference_table")
        cmd = [
            cc,
            *flags,
            f"-I{root.as_posix()}",
            f'-DQUALIFIED_V2_SOURCE_PATH="{v2_source.as_posix()}"',
            str(source),
            "-lm",
            "-lpthread",
            "-Wl,--gc-sections",
            "-o",
            str(exe),
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            raise PortableQualificationError((result.stderr or result.stdout or b"portable strict C compile failed").decode("utf-8", errors="replace").strip())
        run = subprocess.run([str(exe), EXPECTED_V2_SOURCE_SHA256], capture_output=True, check=False)
        if run.returncode != 0:
            raise PortableQualificationError((run.stderr or run.stdout or b"portable C reference execution failed").decode("utf-8", errors="replace").strip())
        if run.stderr:
            raise PortableQualificationError(f"{label} C reference emitted stderr")
        raw = run.stdout
        payload = parse_json_bytes(raw)
        validate_c_reference(payload)
        output = {
            "status": "PHASE6B6_PORTABLE_C_REFERENCE_EMIT_OK",
            "mode": label,
            "raw_stdout_sha256": sha256_bytes(raw),
            "stderr_sha256": sha256_bytes(run.stderr),
            "payload": payload,
            "raw_stdout": raw,
        }
    if before is not None and set(parent.iterdir()) != before:
        raise PortableQualificationError("temporary compiler artifact leakage detected")
    return output


def compare_reference(c_reference: dict[str, Any]) -> dict[str, Any]:
    validate_c_reference(c_reference)
    expected = python_reference_table()
    max_error = 0.0
    for index in range(12):
        observed = c_reference["tones"][index]
        wanted = expected["tones"][index]
        if observed["physical_tone_index"] != wanted["physical_tone_index"]:
            raise PortableQualificationError("C reference tone index mismatch")
        error = abs(float(observed["frequency_hz"]) - float(wanted["frequency_hz"]))
        max_error = max(max_error, error)
        if error > TONE_ABS_TOLERANCE_HZ:
            raise PortableQualificationError("C reference tone frequency mismatch")
    for mode in MODE_NAMES:
        if c_reference["codebook"][mode] != expected["codebook"][mode]:
            raise PortableQualificationError("C reference codebook mismatch")
    for index in range(4):
        if c_reference["codebook_rows"][index] != expected["codebook_rows"][index]:
            raise PortableQualificationError("C reference codebook_rows mismatch")
    return {
        "status": "V2_REFERENCE_EQUIVALENCE_PASS",
        "tone_count": 12,
        "mode_count": 4,
        "max_abs_error_hz": max_error,
    }


def runtime_validate_only(root: Path, manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if "runtime validate-only" not in manifest["portable_qualification_scope"]:
        raise PortableQualificationError("runtime validate-only scope missing")
    phase_root = root / "snapshot" / PHASE6B6_RELATIVE_ROOT
    module_names = [name for name in sys.modules if name == "contracts" or name.startswith("contracts.") or name == "runtime" or name.startswith("runtime.")]
    saved_modules = {name: sys.modules[name] for name in module_names}
    for name in module_names:
        del sys.modules[name]
    sys.path.insert(0, str(phase_root))
    try:
        schedule_module = importlib.import_module("contracts.schedule")
        runtime_module = importlib.import_module("runtime.explicit_slot_runtime")
        schedule = schedule_module.campaign_schedule()
        schedule_module.validate_schedule(schedule)
        custody = runtime_module.run_mock(schedule)
        runtime_module.validate_authority(True, False, False)
        try:
            runtime_module.validate_authority(False, False, True)
        except PermissionError:
            hardware_rejected = True
        else:
            hardware_rejected = False
    finally:
        try:
            sys.path.remove(str(phase_root))
        except ValueError:
            pass
        for name in [name for name in sys.modules if name == "contracts" or name.startswith("contracts.") or name == "runtime" or name.startswith("runtime.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
    if schedule["schedule_sha256"] != SCHEDULE_DIGEST or schedule["schedule_sha256"] != contract["schedule_digest"]:
        raise PortableQualificationError("portable runtime schedule digest mismatch")
    if custody["custody_sha256"] != MOCK_CUSTODY_DIGEST or custody["custody_sha256"] != contract["mock_custody_digest"]:
        raise PortableQualificationError("portable runtime custody digest mismatch")
    if schedule["session_count"] != 12 or schedule["total_slots"] != 10368:
        raise PortableQualificationError("portable runtime schedule geometry mismatch")
    sham_rows = 0
    ordinary_rows = 0
    drive_on_count = 0
    for session in schedule["sessions"]:
        for slot in session["slots"]:
            declared = slot["declared"]
            executed = slot["executed"]
            if not isinstance(executed["drive_on"], bool):
                raise PortableQualificationError("portable runtime drive_on is not explicit bool")
            if executed["drive_on"]:
                drive_on_count += 1
            family = declared.get("order_control_family")
            if family == "ORDER_LABEL_SHAM":
                sham_rows += 1
                declared_family = declared.get("declared_order_family")
                executed_family = executed.get("executed_order_family")
                if {declared_family, executed_family} != {"RND1", "RND2"}:
                    raise PortableQualificationError("portable runtime sham declared/executed controls mismatch")
            elif family in ("FWD", "REV", "RND1", "RND2"):
                ordinary_rows += 1
                if declared.get("declared_order_family") != executed.get("executed_order_family"):
                    raise PortableQualificationError("portable runtime ordinary declared/executed controls mismatch")
    if sham_rows != 864 or ordinary_rows != 3456 or drive_on_count <= 0 or not hardware_rejected:
        raise PortableQualificationError("portable runtime semantic coverage mismatch")
    return {
        "status": "PHASE6B6_PORTABLE_RUNTIME_VALIDATE_ONLY_OK",
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
        "schedule_digest": schedule["schedule_sha256"],
        "mock_custody_digest": custody["custody_sha256"],
        "session_count": schedule["session_count"],
        "total_slots": schedule["total_slots"],
        "order_label_sham_rows": sham_rows,
        "ordinary_order_rows": ordinary_rows,
        "drive_on_slot_count": drive_on_count,
        "hardware_backend_rejected": hardware_rejected,
    }


def hardware_rejection(args: list[str]) -> dict[str, Any]:
    for arg in args:
        name = arg.split("=", 1)[0]
        if name in HARDWARE_OPTIONS:
            raise PortableQualificationError(f"PHASE6B6_PORTABLE_HARDWARE_AUTHORITY_ERROR: forbidden option: {arg}")
        raise PortableQualificationError(f"unknown portable target option: {arg}")
    return {"status": "PHASE6B6_PORTABLE_HARDWARE_OPTIONS_ABSENT"}


def sender_absence_probe() -> dict[str, Any]:
    proc = Path("/proc")
    matches: list[dict[str, Any]] = []
    scanned = 0
    self_pid = os.getpid()
    if proc.is_dir():
        for item in proc.iterdir():
            if not item.name.isdigit():
                continue
            pid = int(item.name)
            if pid == self_pid:
                continue
            try:
                comm = (item / "comm").read_text(encoding="utf-8", errors="replace").strip()
                cmdline = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
                try:
                    exe = os.readlink(item / "exe")
                except OSError:
                    exe = ""
            except OSError:
                continue
            scanned += 1
            haystack = " ".join((comm, cmdline, exe))
            for pattern in FORBIDDEN_PROCESS_PATTERNS:
                if pattern in haystack:
                    matches.append({"pid": pid, "comm": comm, "cmdline": cmdline, "exe": exe, "pattern": pattern})
                    break
    if matches:
        raise PortableQualificationError(f"forbidden process match: {matches}")
    return {"status": "PHASE6B6_PORTABLE_SENDER_PROCESS_ABSENT", "scanned_pid_count": scanned, "matches": []}


def validate_final_result(result: dict[str, Any]) -> None:
    _expect_keys(
        "portable final result",
        result,
        {
            "schema_id",
            "status",
            "portable_manifest_sha256",
            "portable_export_commit",
            "portable_export_tree",
            "snapshot_subject_commit",
            "snapshot_subject_tree",
            "observed_inventory_sha256",
            "calculated_scoped_tree",
            "calculated_phase6b6_subtree_inventory_sha256",
            "calculated_v2_source_sha256",
            "raw_c_emission_sha256",
            "c_reference_equivalence",
            "runtime_validate_only",
            "asan_result",
            "ubsan_result",
            "sender_process_absence",
            "target_executed_git",
            "jsonschema_required",
            "hardware_ran",
            "scientific_acquisition_authorized",
            "final_result_sha256",
        },
    )
    exact = {
        "schema_id": RESULT_SCHEMA_ID,
        "status": "PHASE6B6_PORTABLE_TARGET_QUALIFICATION_PASS",
        "snapshot_subject_commit": SNAPSHOT_SUBJECT_COMMIT,
        "snapshot_subject_tree": SNAPSHOT_SUBJECT_TREE,
        "observed_inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "calculated_scoped_tree": EXPECTED_SCOPED_TREE,
        "calculated_phase6b6_subtree_inventory_sha256": EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256,
        "calculated_v2_source_sha256": EXPECTED_V2_SOURCE_SHA256,
        "target_executed_git": False,
        "jsonschema_required": False,
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
    }
    for key, expected in exact.items():
        if result[key] != expected:
            raise PortableQualificationError(f"portable final result value mismatch: {key}")
    for key in ("portable_manifest_sha256", "raw_c_emission_sha256", "final_result_sha256"):
        if not _hex(result[key], 64):
            raise PortableQualificationError(f"portable final result digest mismatch: {key}")
    for key in ("portable_export_commit", "portable_export_tree"):
        if not _hex(result[key], 40):
            raise PortableQualificationError(f"portable final result identity mismatch: {key}")
    if result["c_reference_equivalence"].get("status") != "V2_REFERENCE_EQUIVALENCE_PASS":
        raise PortableQualificationError("portable final result equivalence status mismatch")
    if result["runtime_validate_only"].get("status") != "PHASE6B6_PORTABLE_RUNTIME_VALIDATE_ONLY_OK":
        raise PortableQualificationError("portable final result runtime status mismatch")
    for key, mode in (("asan_result", "asan"), ("ubsan_result", "ubsan")):
        payload = result[key]
        _expect_keys(f"{mode} result", payload, {"status", "mode", "raw_stdout_sha256", "stderr_sha256"})
        if payload["status"] != "PHASE6B6_PORTABLE_C_REFERENCE_EMIT_OK" or payload["mode"] != mode:
            raise PortableQualificationError(f"{mode} result status mismatch")
        if not _hex(payload["raw_stdout_sha256"], 64) or not _hex(payload["stderr_sha256"], 64):
            raise PortableQualificationError(f"{mode} result digest mismatch")
    sender = result["sender_process_absence"]
    _expect_keys("sender absence result", sender, {"status", "scanned_pid_count", "matches"})
    if sender["status"] != "PHASE6B6_PORTABLE_SENDER_PROCESS_ABSENT" or sender["matches"] != []:
        raise PortableQualificationError("sender absence result mismatch")
    unsigned = copy.deepcopy(result)
    observed = unsigned.pop("final_result_sha256")
    if digest(unsigned) != observed:
        raise PortableQualificationError("portable final result digest recomputation mismatch")


def _public_emit_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "mode": result["mode"],
        "raw_stdout_sha256": result["raw_stdout_sha256"],
        "stderr_sha256": result["stderr_sha256"],
    }


def run(package_root: Path, args: list[str]) -> dict[str, Any]:
    root = package_root.resolve()
    if not root.is_dir():
        raise PortableQualificationError("package root does not exist")
    hardware_rejection(args)
    manifest_sha = _read_manifest_sha(root)
    manifest = load_json(root / "PORTABLE_PACKAGE_MANIFEST.json")
    binding = load_json(root / "TRUSTED_SNAPSHOT_BINDING.json")
    contract = load_json(root / "QUALIFICATION_CONTRACT.json")
    validate_manifest(manifest)
    validate_binding(binding)
    validate_contract(contract)
    verify_copied_files(root, manifest)
    snapshot = verify_snapshot(root, binding)
    first = compile_and_emit(root)
    second = compile_and_emit(root)
    if first["raw_stdout"] != second["raw_stdout"]:
        raise PortableQualificationError("portable raw C reference emission is not deterministic")
    equivalence = compare_reference(first["payload"])
    asan = compile_and_emit(root, sanitize="asan")
    ubsan = compile_and_emit(root, sanitize="ubsan")
    if asan["raw_stdout"] != first["raw_stdout"]:
        raise PortableQualificationError("ASan C reference output mismatch")
    if ubsan["raw_stdout"] != first["raw_stdout"]:
        raise PortableQualificationError("UBSan C reference output mismatch")
    runtime = runtime_validate_only(root, manifest, contract)
    sender = sender_absence_probe()
    result = {
        "schema_id": RESULT_SCHEMA_ID,
        "status": "PHASE6B6_PORTABLE_TARGET_QUALIFICATION_PASS",
        "portable_manifest_sha256": manifest_sha,
        "portable_export_commit": manifest["portable_export_commit"],
        "portable_export_tree": manifest["portable_export_tree"],
        "snapshot_subject_commit": manifest["snapshot_subject_commit"],
        "snapshot_subject_tree": manifest["snapshot_subject_tree"],
        "observed_inventory_sha256": snapshot["observed_inventory_sha256"],
        "calculated_scoped_tree": snapshot["calculated_scoped_tree"],
        "calculated_phase6b6_subtree_inventory_sha256": snapshot["calculated_phase6b6_subtree_inventory_sha256"],
        "calculated_v2_source_sha256": snapshot["calculated_v2_source_sha256"],
        "raw_c_emission_sha256": first["raw_stdout_sha256"],
        "c_reference_equivalence": equivalence,
        "runtime_validate_only": runtime,
        "asan_result": _public_emit_result(asan),
        "ubsan_result": _public_emit_result(ubsan),
        "sender_process_absence": sender,
        "target_executed_git": False,
        "jsonschema_required": False,
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
    }
    result["final_result_sha256"] = digest(result)
    validate_final_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--package-root", type=Path, required=True)
    args, unknown = parser.parse_known_args(argv)
    try:
        result = run(args.package_root, unknown)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

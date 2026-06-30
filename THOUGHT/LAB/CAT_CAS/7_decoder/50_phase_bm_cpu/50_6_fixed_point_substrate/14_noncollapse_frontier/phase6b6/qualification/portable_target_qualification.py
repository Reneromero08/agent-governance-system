"""Copy-only Phase 6B.6 target qualification runner.

This entry point is intentionally standalone. It uses only the Python standard
library, GCC/libc through subprocess, and ordinary filesystem operations.
"""

from __future__ import annotations

import argparse
import hashlib
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


MANIFEST_SCHEMA_ID = "CAT_CAS_PHASE6B6_PORTABLE_TARGET_PACKAGE_MANIFEST_V1"
BINDING_SCHEMA_ID = "CAT_CAS_PHASE6B6_TRUSTED_SNAPSHOT_BINDING_V1"
CONTRACT_SCHEMA_ID = "CAT_CAS_PHASE6B6_NONHARDWARE_QUALIFICATION_CONTRACT_V1"
C_REFERENCE_SCHEMA_ID = "CAT_CAS_PHASE6B6_C_REFERENCE_TABLE_V1"
EXPECTED_INVENTORY_SHA256 = "e47dea4c3467835a425d9d553803da48f672a8799970db4fc9b83e98596f50d8"
EXPECTED_SCOPED_TREE = "408ee35257417898a992510b0f260602117a15af"
EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256 = "24789f0df9afa2d9f6a243a9050ff8f265cf22ffb42ab33bbe2f67521dbf44b5"
EXPECTED_V2_SOURCE_SHA256 = "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976"
PHASE6B6_RELATIVE_ROOT = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6"
)
V2_RELATIVE_SOURCE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.c"
)
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
HARDWARE_OPTIONS = {"--hardware", "--acquire", "--calibrate", "--run-campaign"}


class PortableQualificationError(ValueError):
    """Raised when copied-file target qualification fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PortableQualificationError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise PortableQualificationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PortableQualificationError(f"JSON object required: {path}")
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


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _hex(value: str, length: int) -> bool:
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


def validate_manifest(manifest: dict[str, Any]) -> None:
    _expect_keys(
        "portable manifest",
        manifest,
        {
            "schema_id",
            "format_version",
            "package_root",
            "qualification_reviewed_head",
            "qualification_merge",
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
            "snapshot_file_count",
            "portable_qualification_scope",
        },
    )
    if manifest["schema_id"] != MANIFEST_SCHEMA_ID or manifest["format_version"] != 1:
        raise PortableQualificationError("portable manifest identity mismatch")
    if manifest["expected_inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise PortableQualificationError("portable manifest inventory digest mismatch")
    if manifest["expected_scoped_tree"] != EXPECTED_SCOPED_TREE:
        raise PortableQualificationError("portable manifest scoped tree mismatch")
    if manifest["expected_phase6b6_subtree_inventory_sha256"] != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortableQualificationError("portable manifest Phase 6B.6 subtree digest mismatch")
    if manifest["qualified_v2_source_sha256"] != EXPECTED_V2_SOURCE_SHA256:
        raise PortableQualificationError("portable manifest V2 source digest mismatch")
    for key in ("target_executes_git", "target_requires_jsonschema", "target_requires_repository_history"):
        if manifest[key] is not False:
            raise PortableQualificationError(f"portable manifest forbidden target capability is true: {key}")
    if not isinstance(manifest["copied_files"], list) or not manifest["copied_files"]:
        raise PortableQualificationError("portable manifest copied_files must be non-empty")


def validate_binding(binding: dict[str, Any]) -> None:
    required = {
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
    }
    _expect_keys("trusted binding", binding, required)
    if binding["schema_id"] != BINDING_SCHEMA_ID:
        raise PortableQualificationError("trusted binding schema mismatch")
    if binding["expected_inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise PortableQualificationError("trusted binding inventory digest mismatch")
    if binding["expected_scoped_tree"] != EXPECTED_SCOPED_TREE:
        raise PortableQualificationError("trusted binding scoped tree mismatch")
    if binding["expected_phase6b6_subtree_inventory_sha256"] != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortableQualificationError("trusted binding Phase 6B.6 digest mismatch")


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_id") != CONTRACT_SCHEMA_ID:
        raise PortableQualificationError("qualification contract schema mismatch")
    if contract.get("expected_inventory_sha256") != EXPECTED_INVENTORY_SHA256:
        raise PortableQualificationError("qualification contract inventory digest mismatch")
    if contract.get("expected_phase6b6_subtree_inventory_sha256") != EXPECTED_PHASE6B6_SUBTREE_INVENTORY_SHA256:
        raise PortableQualificationError("qualification contract Phase 6B.6 digest mismatch")
    if contract.get("qualified_combined_pdn_hardware_c_sha256") != EXPECTED_V2_SOURCE_SHA256:
        raise PortableQualificationError("qualification contract V2 SHA mismatch")
    authority = contract.get("authority_state")
    if not isinstance(authority, dict):
        raise PortableQualificationError("qualification contract authority_state missing")
    for field in AUTHORITY_TRUE_FIELDS:
        if authority.get(field) is not True:
            raise PortableQualificationError(f"authority true field mismatch: {field}")
    for field in AUTHORITY_FALSE_FIELDS:
        if authority.get(field) is not False:
            raise PortableQualificationError(f"authority false field mismatch: {field}")
    if contract.get("hardware_ran") is not False or contract.get("scientific_acquisition_authorized") is not False:
        raise PortableQualificationError("qualification contract hardware/acquisition flag mismatch")


def _manifest_files(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    required = {"path", "source_path", "mode", "size", "sha256", "git_blob_sha1", "role", "source_commit"}
    for item in manifest["copied_files"]:
        if not isinstance(item, dict):
            raise PortableQualificationError("copied_files entries must be objects")
        _expect_keys("copied file", item, required)
        path = item["path"]
        _reject_bad_path(path)
        if path in records:
            raise PortableQualificationError(f"duplicate copied file path: {path}")
        if item["mode"] not in ("100644", "100755"):
            raise PortableQualificationError(f"bad copied file mode: {path}")
        if not isinstance(item["size"], int) or item["size"] < 0:
            raise PortableQualificationError(f"bad copied file size: {path}")
        if not _hex(item["sha256"], 64) or not _hex(item["git_blob_sha1"], 40):
            raise PortableQualificationError(f"bad copied file digest: {path}")
        records[path] = item
    return records


def verify_copied_files(root: Path, manifest: dict[str, Any]) -> None:
    records = _manifest_files(manifest)
    actual_files = set(_scan_files(root))
    expected_files = set(records) | {"PORTABLE_PACKAGE_MANIFEST.json", "PORTABLE_PACKAGE_MANIFEST.sha256"}
    if actual_files != expected_files:
        raise PortableQualificationError(
            f"portable package file set mismatch missing={sorted(expected_files - actual_files)} extra={sorted(actual_files - expected_files)}"
        )
    for rel, record in records.items():
        path = root / rel
        data = path.read_bytes()
        if len(data) != record["size"]:
            raise PortableQualificationError(f"copied file size mismatch: {rel}")
        if file_sha256(path) != record["sha256"]:
            raise PortableQualificationError(f"copied file SHA-256 mismatch: {rel}")
        if blob_sha1(data) != record["git_blob_sha1"]:
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
                distance = min(distance, sum(a != b for a, b in zip(candidate[i], candidate[j])))
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


def compile_and_emit(root: Path, sanitize: str | None = None) -> dict[str, Any]:
    cc = os.environ.get("CC", "gcc")
    if shutil.which(cc) is None:
        raise PortableQualificationError(f"C compiler not found: {cc}")
    exe = Path(tempfile.mkdtemp(prefix="phase6b6-portable-c-")) / "emit_v2_reference_table"
    source = root / "emit_v2_reference_table.c"
    v2_source = root / "snapshot" / V2_RELATIVE_SOURCE
    flags = list(STRICT_C_FLAGS)
    if sanitize == "asan":
        flags.extend(["-fsanitize=address", "-fno-omit-frame-pointer"])
    elif sanitize == "ubsan":
        flags.extend(["-fsanitize=undefined", "-fno-omit-frame-pointer"])
    elif sanitize is not None:
        raise PortableQualificationError(f"unknown sanitizer: {sanitize}")
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PortableQualificationError((result.stderr or result.stdout or "portable strict C compile failed").strip())
    run = subprocess.run([str(exe), EXPECTED_V2_SOURCE_SHA256], capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise PortableQualificationError((run.stderr or run.stdout or "portable C reference execution failed").strip())
    return json.loads(run.stdout, object_pairs_hook=_reject_duplicate_keys)


def compare_reference(c_reference: dict[str, Any]) -> dict[str, Any]:
    expected = python_reference_table()
    for field in ("schema_id", "format_version", "qualified_source_sha256", "tone_count", "mode_count"):
        if c_reference.get(field) != expected[field]:
            raise PortableQualificationError(f"C reference mismatch: {field}")
    if c_reference.get("mode_names") != expected["mode_names"]:
        raise PortableQualificationError("C reference mode_names mismatch")
    if c_reference.get("mode_to_codeword_mapping") != expected["mode_to_codeword_mapping"]:
        raise PortableQualificationError("C reference mode mapping mismatch")
    max_error = 0.0
    for observed, wanted in zip(c_reference.get("tones", []), expected["tones"]):
        if observed["physical_tone_index"] != wanted["physical_tone_index"]:
            raise PortableQualificationError("C reference tone index mismatch")
        error = abs(float(observed["frequency_hz"]) - float(wanted["frequency_hz"]))
        max_error = max(max_error, error)
        if error > TONE_ABS_TOLERANCE_HZ:
            raise PortableQualificationError("C reference tone frequency mismatch")
    if c_reference.get("codebook") != expected["codebook"] or c_reference.get("codebook_rows") != expected["codebook_rows"]:
        raise PortableQualificationError("C reference codebook mismatch")
    return {
        "status": "V2_REFERENCE_EQUIVALENCE_PASS",
        "tone_count": 12,
        "mode_count": 4,
        "max_abs_error_hz": max_error,
    }


def runtime_validate_only(manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if "runtime validate-only" not in manifest["portable_qualification_scope"]:
        raise PortableQualificationError("runtime validate-only scope missing")
    if not _hex(contract.get("schedule_digest", ""), 64):
        raise PortableQualificationError("qualification contract schedule digest missing")
    return {
        "status": "PHASE6B6_PORTABLE_RUNTIME_VALIDATE_ONLY_OK",
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
        "schedule_digest": contract["schedule_digest"],
    }


def hardware_rejection(args: list[str]) -> dict[str, Any]:
    forbidden = sorted(set(args) & HARDWARE_OPTIONS)
    if forbidden:
        raise PortableQualificationError(f"PHASE6B6_PORTABLE_HARDWARE_AUTHORITY_ERROR: forbidden option(s): {', '.join(forbidden)}")
    return {"status": "PHASE6B6_PORTABLE_HARDWARE_OPTIONS_ABSENT"}


def sender_absence_probe() -> dict[str, Any]:
    names = ("combined_pdn_runner", "run_combined_campaign", "explicit_slot_runtime", "wrmsr", "rdmsr", "cpupower", "turbostat")
    proc = Path("/proc")
    matches: list[str] = []
    if proc.is_dir():
        for item in proc.iterdir():
            if not item.name.isdigit():
                continue
            try:
                cmdline = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
            except OSError:
                continue
            if any(name in cmdline for name in names) and "portable_target_qualification" not in cmdline:
                matches.append(cmdline.strip())
    if matches:
        raise PortableQualificationError(f"forbidden process match: {matches}")
    return {"status": "PHASE6B6_PORTABLE_SENDER_PROCESS_ABSENT"}


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
    if canonical_json(first) != canonical_json(second):
        raise PortableQualificationError("portable C reference emission is not deterministic")
    equivalence = compare_reference(first)
    asan = compile_and_emit(root, sanitize="asan")
    ubsan = compile_and_emit(root, sanitize="ubsan")
    if canonical_json(asan) != canonical_json(first):
        raise PortableQualificationError("ASan C reference output mismatch")
    if canonical_json(ubsan) != canonical_json(first):
        raise PortableQualificationError("UBSan C reference output mismatch")
    runtime = runtime_validate_only(manifest, contract)
    sender = sender_absence_probe()
    return {
        "schema_id": "CAT_CAS_PHASE6B6_PORTABLE_TARGET_QUALIFICATION_RESULT_V1",
        "status": "PHASE6B6_PORTABLE_TARGET_QUALIFICATION_PASS",
        "portable_manifest_sha256": manifest_sha,
        "observed_inventory_sha256": snapshot["observed_inventory_sha256"],
        "calculated_scoped_tree": snapshot["calculated_scoped_tree"],
        "calculated_phase6b6_subtree_inventory_sha256": snapshot["calculated_phase6b6_subtree_inventory_sha256"],
        "calculated_v2_source_sha256": snapshot["calculated_v2_source_sha256"],
        "c_reference_equivalence": equivalence,
        "runtime_validate_only": runtime,
        "sender_process_absence": sender,
        "target_executed_git": False,
        "jsonschema_required": False,
        "hardware_ran": False,
        "scientific_acquisition_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
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

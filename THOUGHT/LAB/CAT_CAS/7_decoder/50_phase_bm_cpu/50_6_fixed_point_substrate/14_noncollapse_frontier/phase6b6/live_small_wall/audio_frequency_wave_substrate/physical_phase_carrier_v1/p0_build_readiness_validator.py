#!/usr/bin/env python3
"""Snapshot-validate and close the non-executing P0 build-readiness packet.

The validator performs local file reads only.  It has no network, instrument,
audio, purchasing, or hardware capability.  Every validation run reads the
candidate bytes once, validates that immutable snapshot, and binds independent
reviews to a domain-separated candidate root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent
LANE = ROOT.parent / "AUDIO_SIDE_QUEST_ROADMAP.md"
AUTHORITY = "AUTHORIZE P0 BUILD-READINESS ONLY"
CEILING = "NON_EXECUTING_P0_BUILD_READINESS_ONLY"
NEXT = "USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD"
MANIFEST = "P0_BUILD_READINESS_MANIFEST.json"
RESULT = "P0_BUILD_READINESS_RESULTS.json"
REVIEWS = "P0_BUILD_READINESS_REVIEWS.json"
MUTATION_RESULT = "P0_BUILD_READINESS_MUTATION_RESULTS.json"
RESEARCH_PREFIX = "research/P0_research_bundle_2026-07-18"
RESEARCH_SOURCE_COMMIT = "cb53976612cbe83bec82df826a9889418f7e0b89"
RESEARCH_NAMES = (
    ".gitignore",
    "DOWNLOAD_LINKS.md",
    "MANIFEST.json",
    "MANIFEST.schema.json",
    "OPEN_GAPS.md",
    "README.md",
    "REGISTRY_RECONCILIATION.md",
    "RESEARCH_SYNTHESIS.md",
    "SOURCE_CUSTODY.json",
    "THIRD_PARTY_RIGHTS.md",
    "notes/INSTALL_IN_REPO.md",
    "notes/REVISION_ALERTS.md",
    "notes/SCIENCE_RESEARCH_NOTES.md",
    "repo_context/README.md",
    "scripts/bootstrap_windows.ps1",
    "scripts/build_custody_snapshot.py",
    "scripts/download_sources.py",
    "scripts/import_repo_context.py",
    "scripts/make_complete_archive.py",
    "scripts/verify_downloads.py",
    "sources/README.md",
    "sources/official/README.md",
    "sources/supplemental/README.md",
)

CANDIDATE_NAMES = (
    "P0_CARRIER_AND_ACCESS_SELECTION.md",
    "P0_MEASUREMENT_AND_SOURCE_OFF_PLAN.md",
    "P0_CONTROL_KILL_AND_ADJUDICATION.md",
    "P0_BOM_SAFETY_AND_SILICON_TRANSLATION.md",
    "P0_EVIDENCE_SCHEMAS.json",
    "P0_ANALYSIS_CONFORMANCE_VECTORS.json",
    "p0_analysis_conformance.py",
    "p0_packet_validator.py",
    "P0_REVIEW_REPORTS.md",
    "P0_FINDINGS_NORMALIZED.json",
    "P0_BUILD_READINESS_AUTHORITY.md",
    "P0_BUILD_READINESS_COMPONENT_SEED.md",
    "PHYSICAL_PHASE_CARRIER_P0_CONTRACT.md",
    "P0_BUILD_READINESS_PACKET.md",
    "P0_FINAL_NETLIST.json",
    "P0_NONPURCHASING_BOM.json",
    "P0_PCB_FABRICATION_RELEASE.json",
    "P0_COMPONENT_DOCUMENTS.json",
    "P0_AUTHORED_ASSEMBLY_DRAWINGS.md",
    "P0_UNPOWERED_ASSEMBLY_PACKET.md",
    "P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md",
    "P0_BUILD_READINESS_SCHEMAS.json",
    "P0_SCIENTIFIC_FIXTURES.json",
    "P0_ANALYZER_REFERENCE_RESULTS.json",
    "p0_scientific_analyzer.py",
    "p0_build_readiness_design.py",
    "p0_build_readiness_validator.py",
    "p0_build_readiness_mutation_test.py",
    "P0_BUILD_READINESS_FINDINGS.json",
    "../AUDIO_SIDE_QUEST_ROADMAP.md",
) + tuple(f"{RESEARCH_PREFIX}/{name}" for name in RESEARCH_NAMES)
FINAL_ONLY = (REVIEWS, "P0_BUILD_READINESS_REVIEW_REPORTS.md", MUTATION_RESULT)
PRETTY_JSON = {
    "P0_FINAL_NETLIST.json", "P0_NONPURCHASING_BOM.json", "P0_PCB_FABRICATION_RELEASE.json",
    "P0_COMPONENT_DOCUMENTS.json", "P0_BUILD_READINESS_SCHEMAS.json", "P0_SCIENTIFIC_FIXTURES.json",
    "P0_ANALYZER_REFERENCE_RESULTS.json", "P0_BUILD_READINESS_FINDINGS.json",
    f"{RESEARCH_PREFIX}/MANIFEST.json", f"{RESEARCH_PREFIX}/SOURCE_CUSTODY.json",
}
HEX = set("0123456789abcdef")


class Failure(ValueError):
    pass


def duplicate_safe(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise Failure(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def parse_json(name: str, snapshot: Mapping[str, bytes], canonical_required: bool = True) -> dict[str, Any]:
    if name not in snapshot:
        raise Failure(f"snapshot missing {name}")
    try:
        value = json.loads(
            snapshot[name].decode("utf-8"), object_pairs_hook=duplicate_safe,
            parse_constant=lambda token: (_ for _ in ()).throw(Failure(f"non-finite JSON token: {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Failure(f"invalid JSON {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise Failure(f"JSON root must be an object: {name}")
    if canonical_required and snapshot[name] != canonical(value):
        raise Failure(f"noncanonical JSON bytes: {name}")
    return value


def read_snapshot(include_final: bool = False) -> dict[str, bytes]:
    names = CANDIDATE_NAMES + (FINAL_ONLY if include_final else ())
    snapshot: dict[str, bytes] = {}
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_file():
            raise Failure(f"missing required file: {name}")
        snapshot[name] = path.read_bytes()
    return snapshot


def candidate_root(snapshot: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256(b"P0-BUILD-READINESS-CANDIDATE-ROOT-V2\0")
    for name in sorted(CANDIDATE_NAMES):
        if name not in snapshot:
            raise Failure(f"candidate root missing {name}")
        data = snapshot[name]
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def text(name: str, snapshot: Mapping[str, bytes]) -> str:
    try:
        return snapshot[name].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Failure(f"UTF-8 required: {name}") from exc


def check_scripts(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    forbidden = {"socket", "requests", "urllib", "http", "serial", "sounddevice", "pyaudio", "subprocess", "paramiko"}
    report: dict[str, Any] = {}
    for name in ("p0_scientific_analyzer.py", "p0_build_readiness_design.py", "p0_build_readiness_validator.py", "p0_build_readiness_mutation_test.py"):
        source = text(name, snapshot)
        tree = ast.parse(source, filename=name)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        overlap = sorted(imports & forbidden)
        if overlap:
            raise Failure(f"forbidden capability import in {name}: {overlap}")
        report[name] = {"bytes": len(snapshot[name]), "sha256": sha256(snapshot[name]), "git_blob_sha1": git_blob(snapshot[name]), "forbidden_capability_imports": []}
    analyzer = text("p0_scientific_analyzer.py", snapshot)
    for token in ("build", "self-test", "verify", "analyze", "SYNTHETIC_ANALYZER_PASS", "physical_claim_authorized", "source_muted_at_gate_raw", "reference_tone_missing_raw", "ENVIRONMENT_CRC"):
        if token not in analyzer:
            raise Failure(f"analyzer missing required mechanism: {token}")
    return report


def valid_https_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("https://") and value.strip() == value and not any(character.isspace() for character in value)


def check_research(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    manifest_name = f"{RESEARCH_PREFIX}/MANIFEST.json"
    schema_name = f"{RESEARCH_PREFIX}/MANIFEST.schema.json"
    custody_name = f"{RESEARCH_PREFIX}/SOURCE_CUSTODY.json"
    manifest = parse_json(manifest_name, snapshot)
    manifest_schema = parse_json(schema_name, snapshot, canonical_required=False)
    custody = parse_json(custody_name, snapshot)
    if manifest.get("schema") != "p0.research-bundle-manifest.v1" or manifest_schema.get("$id") != manifest["schema"]:
        raise Failure("research manifest/schema identity")
    records = manifest.get("records")
    if manifest.get("record_count") != 35 or not isinstance(records, list) or len(records) != 35:
        raise Failure("research manifest exact 35-record count")
    manifest_fields = {"collection", "direct_download_url", "download_mode", "id", "legacy_expected_bytes", "legacy_expected_sha256", "local_filename", "official_product_page"}
    ids = [record.get("id") for record in records]
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != 35:
        raise Failure("research source IDs must be 35 unique nonempty strings")
    for record in records:
        if set(record) != manifest_fields or not valid_https_url(record["official_product_page"]):
            raise Failure(f"research manifest record/official URL: {record.get('id')}")
        direct = record["direct_download_url"]
        if direct is not None and not valid_https_url(direct):
            raise Failure(f"research direct URL: {record['id']}")
        legacy_bytes = record["legacy_expected_bytes"]
        legacy_hash = record["legacy_expected_sha256"]
        if (legacy_bytes is None) != (legacy_hash is None):
            raise Failure(f"research legacy custody pair: {record['id']}")
        if legacy_hash is not None and (not isinstance(legacy_bytes, int) or isinstance(legacy_bytes, bool) or legacy_bytes < 1 or len(legacy_hash) != 64 or any(character not in HEX for character in legacy_hash)):
            raise Failure(f"research legacy custody value: {record['id']}")
    if custody.get("schema") != "p0.research-source-custody.v1" or custody.get("source_commit") != RESEARCH_SOURCE_COMMIT or custody.get("source_record_count") != 35:
        raise Failure("research custody root/source commit/count")
    if custody.get("manifest_sha256") != sha256(snapshot[manifest_name]):
        raise Failure("research custody manifest-hash binding")
    custody_records = custody.get("records")
    if not isinstance(custody_records, list) or len(custody_records) != 35:
        raise Failure("research custody records")
    custody_fields = {
        "collection", "current_bytes", "current_sha256", "custody_state", "direct_download_url",
        "download_result", "download_result_detail", "final_redirect_url", "legacy_expected_bytes",
        "legacy_expected_sha256", "license_or_redistribution_note", "local_filename",
        "official_product_page", "publisher", "relevance_to_p0", "revision_and_date", "role",
        "source_id", "title",
    }
    allowed_states = {
        "LOCAL_BYTES_CAPTURED_AND_HASH_VERIFIED",
        "LOCAL_CURRENT_BYTES_CAPTURED__LEGACY_DIFFERS",
        "URL_AND_LEGACY_HASH_RECORDED__BYTES_NOT_LOCAL",
        "MANUAL_CAPTURE_REQUIRED",
        "PROSPECTIVE_IDENTITY_ONLY",
    }
    manifest_by_id = {record["id"]: record for record in records}
    custody_ids: list[str] = []
    counts: dict[str, int] = {}
    for record in custody_records:
        if set(record) != custody_fields:
            raise Failure(f"research custody exact fields: {record.get('source_id')}")
        identity = record["source_id"]
        custody_ids.append(identity)
        if identity not in manifest_by_id:
            raise Failure(f"research custody unknown ID: {identity}")
        source = manifest_by_id[identity]
        for key in ("collection", "direct_download_url", "legacy_expected_bytes", "legacy_expected_sha256", "local_filename", "official_product_page"):
            if record[key] != source[key]:
                raise Failure(f"research manifest/custody mismatch: {identity}:{key}")
        if not all(isinstance(record[key], str) and record[key] for key in ("download_result", "download_result_detail", "license_or_redistribution_note", "publisher", "relevance_to_p0", "revision_and_date", "role", "title")):
            raise Failure(f"research source descriptive metadata: {identity}")
        state = record["custody_state"]
        if state not in allowed_states:
            raise Failure(f"research custody state: {identity}")
        counts[state] = counts.get(state, 0) + 1
        current_bytes = record["current_bytes"]
        current_hash = record["current_sha256"]
        has_current = isinstance(current_bytes, int) and not isinstance(current_bytes, bool) and current_bytes > 0 and isinstance(current_hash, str) and len(current_hash) == 64 and not any(character not in HEX for character in current_hash)
        if (current_bytes is None) != (current_hash is None):
            raise Failure(f"research current custody pair: {identity}")
        if state in {"LOCAL_BYTES_CAPTURED_AND_HASH_VERIFIED", "LOCAL_CURRENT_BYTES_CAPTURED__LEGACY_DIFFERS"}:
            if not has_current or record["download_result"] not in {"DOWNLOADED", "ALREADY_PRESENT"}:
                raise Failure(f"research local-byte custody claim: {identity}")
            legacy_hash = record["legacy_expected_sha256"]
            if state == "LOCAL_CURRENT_BYTES_CAPTURED__LEGACY_DIFFERS" and (legacy_hash is None or legacy_hash == current_hash):
                raise Failure(f"research legacy/current difference claim: {identity}")
            if state == "LOCAL_BYTES_CAPTURED_AND_HASH_VERIFIED" and legacy_hash is not None and legacy_hash != current_hash:
                raise Failure(f"research unreported legacy/current difference: {identity}")
        elif has_current:
            raise Failure(f"research nonlocal state contains current bytes: {identity}")
        if state == "URL_AND_LEGACY_HASH_RECORDED__BYTES_NOT_LOCAL" and (record["legacy_expected_bytes"] is None or record["legacy_expected_sha256"] is None):
            raise Failure(f"research URL/legacy state lacks history: {identity}")
        if state in {"MANUAL_CAPTURE_REQUIRED", "PROSPECTIVE_IDENTITY_ONLY"} and record["legacy_expected_sha256"] is not None:
            raise Failure(f"research historical hash was not retained in custody state: {identity}")
    if len(set(custody_ids)) != 35 or set(custody_ids) != set(ids) or custody.get("custody_state_counts") != dict(sorted(counts.items())):
        raise Failure("research custody IDs/state counts")
    if "private and uncommitted" not in custody.get("third_party_byte_policy", ""):
        raise Failure("research private third-party byte policy")
    required_authored = {
        "RESEARCH_SYNTHESIS.md", "OPEN_GAPS.md", "notes/REVISION_ALERTS.md", "THIRD_PARTY_RIGHTS.md",
        "scripts/download_sources.py", "scripts/verify_downloads.py", "scripts/build_custody_snapshot.py",
    }
    if not required_authored.issubset(set(RESEARCH_NAMES)):
        raise Failure("research authored dependency coverage")
    forbidden_suffixes = {".pdf", ".html", ".htm", ".zip", ".exe", ".dll", ".bin"}
    for name in (f"{RESEARCH_PREFIX}/{member}" for member in RESEARCH_NAMES):
        if Path(name).suffix.lower() in forbidden_suffixes or snapshot[name].startswith((b"%PDF-", b"PK\x03\x04", b"<html", b"<!DOCTYPE html")):
            raise Failure(f"third-party or archive bytes entered candidate: {name}")
    ignore = text(f"{RESEARCH_PREFIX}/.gitignore", snapshot)
    for token in ("sources/official/*.pdf", "sources/supplemental/*.pdf", "DOWNLOAD_RECEIPT.json", "VERIFICATION_REPORT.json", "*_complete.zip", "repo_context/*", "sources/**/*.part"):
        if token not in ignore:
            raise Failure(f"research ignore fence missing: {token}")
    rights = text(f"{RESEARCH_PREFIX}/THIRD_PARTY_RIGHTS.md", snapshot)
    install = text(f"{RESEARCH_PREFIX}/notes/INSTALL_IN_REPO.md", snapshot)
    if "repository-safe `SOURCE_CUSTODY.json` retrieval summary" not in rights or "raw `DOWNLOAD_RECEIPT.json` and `VERIFICATION_REPORT.json` private and ignored" not in rights or "repository-safe `SOURCE_CUSTODY.json`" not in install or "raw downloader/verifier receipts" not in install:
        raise Failure("repository-safe retrieval receipt and third-party rights policy")
    script_report = {}
    for name in RESEARCH_NAMES:
        if name.endswith(".py"):
            full_name = f"{RESEARCH_PREFIX}/{name}"
            ast.parse(text(full_name, snapshot), filename=full_name)
            script_report[name] = sha256(snapshot[full_name])
    return {
        "custody_state_counts": dict(sorted(counts.items())),
        "manifest_schema": manifest["schema"],
        "manifest_sha256": sha256(snapshot[manifest_name]),
        "record_count": 35,
        "repository_safe_file_count": len(RESEARCH_NAMES),
        "scripts": script_report,
        "source_commit": RESEARCH_SOURCE_COMMIT,
    }


def check_netlist(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    netlist = parse_json("P0_FINAL_NETLIST.json", snapshot)
    required_root = {
        "admittance_budget", "authority", "boards", "channel_map", "claim_ceiling", "components", "connector_map",
        "continuity_tests", "control_fixtures", "external_cables", "failure_state_table", "forbidden_connections",
        "ground_model", "inter_enclosure_harnesses", "intra_enclosure_harnesses", "nets", "power_domains",
        "relay_coil_driver_map", "relay_contact_map", "revision", "schema", "source_off_sequence",
        "source_off_state_table", "status", "switch_truth_table", "test_point_map", "witness_law",
    }
    if set(netlist) != required_root or netlist["schema"] != "p0.final-netlist.v3" or netlist["revision"] != "P0-NETLIST-REV-C-20260717" or netlist["authority"] != AUTHORITY or netlist["claim_ceiling"] != CEILING:
        raise Failure("netlist root/schema/authority/ceiling")
    components = netlist["components"]
    if not isinstance(components, list) or not components:
        raise Failure("empty component inventory")
    refs = [item.get("ref") for item in components]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(refs) != len(set(refs)):
        raise Failure("component references must be unique nonempty strings")
    by_ref = {item["ref"]: item for item in components}
    if by_ref.get("X1_A", {}).get("part_number") != "Q13FC1350000401" or by_ref["X1_A"].get("manufacturer") != "Epson":
        raise Failure("exact FC-135 order identity")
    if any(ref in by_ref for ref in ("X1_B", "X1_C", "C_DUMMY_A", "C_DUMMY_B")) or by_ref.get("C_DUMMY_C", {}).get("part_number") != "GJM1555C1H1R0BB01D":
        raise Failure("DUT/detector-only/exact-C0 population identity")
    expected_members: dict[str, list[str]] = {}
    no_connects: set[str] = set()
    for item in components:
        if set(item) != {"board", "domain", "exact_document", "function", "manufacturer", "part_number", "pins", "ref"} or not isinstance(item["pins"], dict) or not item["pins"]:
            raise Failure(f"component schema: {item.get('ref')}")
        for pin, net in item["pins"].items():
            if not isinstance(pin, str) or not isinstance(net, str) or not net:
                raise Failure(f"component pin type: {item['ref']}")
            if net.startswith("NC::"):
                expected = f"NC::{item['ref']}.{pin}"
                if net != expected or net in no_connects:
                    raise Failure(f"no-connect identity: {item['ref']}.{pin} -> {net}")
                no_connects.add(net)
            elif net.startswith("DNP::"):
                continue
            else:
                expected_members.setdefault(net, []).append(f"{item['ref']}.{pin}")
    connectors = netlist["connector_map"]
    if len(connectors) != 18:
        raise Failure("exactly 18 fixture-local connectors required")
    connector_ids: set[str] = set()
    for item in connectors:
        if item["id"] in connector_ids or item["part_number"] != "031-10-RFX" or "integral nylon" not in item["body_isolation"]:
            raise Failure(f"connector identity/isolation: {item.get('id')}")
        connector_ids.add(item["id"])
        expected_members.setdefault(item["center"], []).append(f"{item['id']}.center")
        if not item["return"].startswith("NC::"):
            expected_members.setdefault(item["return"], []).append(f"{item['id']}.return")
    actual_nets = {item["name"]: item["members"] for item in netlist["nets"]}
    expected_members = {name: sorted(members) for name, members in expected_members.items()}
    if actual_nets != dict(sorted(expected_members.items())):
        raise Failure("net membership is not an exact projection of component/connector pins")
    if (
        len(netlist["boards"]) != 12 or len(netlist["channel_map"]) != 12 or len(netlist["inter_enclosure_harnesses"]) != 3
        or len(netlist["intra_enclosure_harnesses"]) != 6 or len(netlist["external_cables"]) != 18
        or len(netlist["power_domains"]) != 12 or len(netlist["control_fixtures"]) != 3
        or len(netlist["relay_contact_map"]) != 18 or len(netlist["relay_coil_driver_map"]) != 9
        or len(netlist["test_point_map"]) != 24 or len(netlist["failure_state_table"]) != 9
    ):
        raise Failure("three-complete-fixture cardinality")
    if {item.get("fixture_end") for item in netlist["external_cables"]} != connector_ids or any(item.get("part_number") != "2249-C-24" for item in netlist["external_cables"]):
        raise Failure("external cable map must cover every fixture connector exactly once")
    if any(not item.get("scientific_action", "").startswith("REJECT") for item in netlist["failure_state_table"]):
        raise Failure("every declared failure must reject scientific use")
    expected_switch_truth = [
        {"adg1419_pin_1_d": "N_SRC", "adg1419_pin_2_sa": "N_GATE_TERM", "adg1419_pin_8_sb": "N_GATE_OUT", "in_logic": 0, "route": "D-to-SA; C1 terminated through exact 50.00 ohm R_TERM", "state": "OFF_TERMINATE"},
        {"adg1419_pin_1_d": "N_SRC", "adg1419_pin_2_sa": "N_GATE_TERM", "adg1419_pin_8_sb": "N_GATE_OUT", "in_logic": 1, "route": "D-to-SB; C1 presented to K1", "state": "DRIVE"},
    ]
    if netlist["switch_truth_table"] != expected_switch_truth:
        raise Failure("ADG1419 D/SA/SB truth table")
    expected_source_states = [
        {"adg_route": "DRIVE D-to-SB", "admissible_for_science": False, "expected_ch2_code": 7, "gate": 1, "k1": 1, "k2": 1, "k3": 1, "state": "S0_DRIVE"},
        {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "expected_ch2_code": 6, "gate": 0, "k1": 1, "k2": 1, "k3": 1, "state": "S1_GATE_TERMINATED"},
        {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "allowed_ch2_codes": [0, 2, 4, 6], "gate": 0, "k1": "RELEASING", "k2": "RELEASING", "k3": 1, "state": "S2_OPEN_SERIES_BARRIERS"},
        {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "expected_ch2_code": 0, "gate": 0, "k1": 0, "k2": 0, "k3": 1, "state": "S3_SERIES_OPEN_STABLE_1000_SAMPLES"},
        {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "allowed_ch2_codes": [0, 8], "gate": 0, "k1": 0, "k2": 0, "k3": "RELEASING_TO_GUARD", "state": "S4_GUARD_TRANSITION"},
        {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": True, "expected_ch2_code": 8, "gate": 0, "k1": 0, "k2": 0, "k3": 0, "state": "S5_STABLE_OFF_AFTER_1000_SAMPLES_AND_GUARD"},
        {"adg_route": "unpowered continuity state only", "admissible_for_science": False, "expected_ch2_code": None, "gate": 0, "k1": 0, "k2": 0, "k3": 0, "state": "S6_UNPOWERED_FAIL_SAFE"},
    ]
    states = netlist["source_off_state_table"]
    if states != expected_source_states:
        raise Failure("source-off state table")
    for suffix in "ABC":
        required = (f"U_GATE_{suffix}", f"K1_{suffix}", f"K2_{suffix}", f"K3_{suffix}", f"U_SENSE_{suffix}", f"U_REF_{suffix}", f"UISO_GATE_{suffix}", f"UISO_RELAY_{suffix}", f"U_ACCEL_{suffix}", f"U_TEMP_RH_{suffix}", f"CTRL1_{suffix}")
        if any(ref not in by_ref for ref in required):
            raise Failure(f"fixture {suffix} incomplete")
        if by_ref[f"U_SENSE_{suffix}"]["part_number"] != "OPA810IDT" or by_ref[f"U_SENSE_{suffix}"]["pins"] != {"1": f"NC::U_SENSE_{suffix}.1", "2": f"N_SENSE_OUT_{suffix}", "3": f"N_ELECTRODE_A_{suffix}", "4": f"-5V_SENSE_{suffix}", "5": f"NC::U_SENSE_{suffix}.5", "6": f"N_SENSE_OUT_{suffix}", "7": f"+5V_SENSE_{suffix}", "8": f"NC::U_SENSE_{suffix}.8"}:
            raise Failure(f"OPA810 physical pin map: fixture {suffix}")
        if by_ref[f"U_REF_{suffix}"]["pins"] != {"1": f"NC::U_REF_{suffix}.1", "2": f"+5V_SENSE_{suffix}", "3": f"NC::U_REF_{suffix}.3", "4": f"AGND_EXPORT_{suffix}", "5": f"NC::U_REF_{suffix}.5", "6": f"ADR_REF_3V3_{suffix}", "7": f"NC::U_REF_{suffix}.7", "8": f"NC::U_REF_{suffix}.8"}:
            raise Failure(f"ADR4533 physical pin map: fixture {suffix}")
        expected_gate_isolator = {"1": f"CTRL_3V3_{suffix}", "2": f"CTRL_GND_{suffix}", "3": f"CMD_GATE_{suffix}", "4": f"CTRL_GND_{suffix}", "5": f"CTRL_GND_{suffix}", "6": f"CTRL_GND_{suffix}", "7": f"CTRL_GND_{suffix}", "8": f"CTRL_GND_{suffix}", "9": f"AGND_EXPORT_{suffix}", "10": f"NC::UISO_GATE_{suffix}.10", "11": f"NC::UISO_GATE_{suffix}.11", "12": f"NC::UISO_GATE_{suffix}.12", "13": f"NC::UISO_GATE_{suffix}.13", "14": f"IN_GATE_{suffix}", "15": f"AGND_EXPORT_{suffix}", "16": f"ADR_REF_3V3_{suffix}"}
        expected_relay_isolator = {"1": f"CTRL_3V3_{suffix}", "2": f"CTRL_GND_{suffix}", "3": f"CMD_K1_{suffix}", "4": f"CMD_K2_{suffix}", "5": f"CMD_K3_{suffix}", "6": f"CTRL_GND_{suffix}", "7": f"CTRL_GND_{suffix}", "8": f"CTRL_GND_{suffix}", "9": f"GND_RELAY_{suffix}", "10": f"NC::UISO_RELAY_{suffix}.10", "11": f"NC::UISO_RELAY_{suffix}.11", "12": f"DRV_K3_{suffix}", "13": f"DRV_K2_{suffix}", "14": f"DRV_K1_{suffix}", "15": f"GND_RELAY_{suffix}", "16": f"+5V_RELAY_{suffix}"}
        if by_ref[f"UISO_GATE_{suffix}"]["pins"] != expected_gate_isolator or by_ref[f"UISO_RELAY_{suffix}"]["pins"] != expected_relay_isolator:
            raise Failure(f"ADuM140D physical pin map/NIC isolation: fixture {suffix}")
        if by_ref[f"C_ISO_GATE_SEC_{suffix}"]["pins"] != {"1": f"ADR_REF_3V3_{suffix}", "2": f"AGND_EXPORT_{suffix}"}:
            raise Failure(f"gate-secondary 3.3 V bypass/reference identity: fixture {suffix}")
        expected_gate_switch = {"1": f"N_SRC_{suffix}", "2": f"N_GATE_TERM_{suffix}", "3": f"AGND_EXPORT_{suffix}", "4": f"+5V_GATE_{suffix}", "5": f"NC::U_GATE_{suffix}.5", "6": f"IN_GATE_{suffix}", "7": f"-5V_GATE_{suffix}", "8": f"N_GATE_OUT_{suffix}"}
        if by_ref[f"U_GATE_{suffix}"]["pins"] != expected_gate_switch:
            raise Failure(f"ADG1419 MSOP physical pin map: fixture {suffix}")
        if by_ref[f"R_MON_C1_{suffix}"]["pins"] != {"1": f"C1_IN_{suffix}", "2": f"N_SOURCE_MONITOR_SUM_{suffix}"}:
            raise Failure(f"C1 monitor must remain upstream of limiter/gate: fixture {suffix}")
        expected_accel = {"1": f"AGND_STAR_{suffix}", "2": f"AGND_STAR_{suffix}", "3": f"AGND_STAR_{suffix}", "4": f"NC::U_ACCEL_{suffix}.4", "5": f"ADR_REF_3V3_{suffix}", "6": f"AGND_STAR_{suffix}", "7": f"ADR_REF_3V3_{suffix}", "8": f"ADXL_1V8_DIG_{suffix}", "9": f"AGND_STAR_{suffix}", "10": f"ADXL_1V8_ANA_{suffix}", "11": f"ADR_REF_3V3_{suffix}", "12": f"ADXL_XOUT_{suffix}", "13": f"ADXL_YOUT_{suffix}", "14": f"N_ACCEL_Z_{suffix}"}
        if by_ref[f"U_ACCEL_{suffix}"]["pins"] != expected_accel:
            raise Failure(f"ADXL354 physical pin/power map: fixture {suffix}")
        relay_maps = {
            f"K1_{suffix}": {"1": f"+5V_RELAY_{suffix}", "2": f"NC::K1_{suffix}.2", "3": f"N_GATE_OUT_{suffix}", "4": f"N_MIDPOINT_{suffix}", "5": f"N_WIT_K1_{suffix}", "6": f"ADR_REF_3V3_{suffix}", "7": f"NC::K1_{suffix}.7", "8": f"K1_COIL_LOW_{suffix}"},
            f"K2_{suffix}": {"1": f"+5V_RELAY_{suffix}", "2": f"NC::K2_{suffix}.2", "3": f"N_MIDPOINT_{suffix}", "4": f"N_ELECTRODE_A_{suffix}", "5": f"N_WIT_K2_{suffix}", "6": f"ADR_REF_3V3_{suffix}", "7": f"NC::K2_{suffix}.7", "8": f"K2_COIL_LOW_{suffix}"},
            f"K3_{suffix}": {"1": f"+5V_RELAY_{suffix}", "2": f"N_GUARD_TERM_{suffix}", "3": f"N_MIDPOINT_{suffix}", "4": f"NC::K3_{suffix}.4", "5": f"NC::K3_{suffix}.5", "6": f"ADR_REF_3V3_{suffix}", "7": f"N_WIT_K3_{suffix}", "8": f"K3_COIL_LOW_{suffix}"},
        }
        if any(by_ref[ref]["pins"] != pins for ref, pins in relay_maps.items()):
            raise Failure(f"G6K-2F-Y terminal map: fixture {suffix}")
        ladder = {f"R_W{index}_{suffix}": by_ref[f"R_W{index}_{suffix}"]["pins"] for index in range(4)}
        if any(pins["2"] != f"N_WITNESS_{suffix}" for pins in ladder.values()) or ladder[f"R_W0_{suffix}"]["1"] != f"IN_GATE_{suffix}":
            raise Failure(f"witness ladder output continuity: fixture {suffix}")
        if any(by_ref[f"R_W{index}_{suffix}"]["pins"]["1"] != f"N_WIT_K{index}_{suffix}" for index in (1, 2, 3)):
            raise Failure(f"witness contact selection continuity: fixture {suffix}")
        internal = {item["id"]: item for item in netlist["intra_enclosure_harnesses"] if item.get("assembly") == f"P0-{('DUT' if suffix == 'A' else 'DETECTOR' if suffix == 'B' else 'DUMMY-C0')}-{suffix}"}
        expected_internal_ids = {f"P0-CONTROL-INTERNAL-WIRING-REV-B-{suffix}", f"P0-CARRIER-INTERNAL-WIRING-REV-B-{suffix}"}
        if set(internal) != expected_internal_ids or any(item.get("no_connector_or_splice") is not True for item in internal.values()):
            raise Failure(f"internal harness identity: fixture {suffix}")
        control_paths = internal[f"P0-CONTROL-INTERNAL-WIRING-REV-B-{suffix}"]["paths"]
        carrier_paths = internal[f"P0-CARRIER-INTERNAL-WIRING-REV-B-{suffix}"]["paths"]
        if len(control_paths) != 17 or len(carrier_paths) != 6:
            raise Failure(f"internal harness path count: fixture {suffix}")
        if sorted({path["finished_length_mm"] for path in control_paths}) != [0, 55, 60] or sorted({path["finished_length_mm"] for path in carrier_paths}) != [35, 45]:
            raise Failure(f"internal harness frozen lengths: fixture {suffix}")
        c2_shell = [path for path in control_paths if path.get("net") == f"NC::J_SRC_C2_{suffix}.shell"]
        if len(c2_shell) != 1 or c2_shell[0].get("conductor") != "NONE" or c2_shell[0].get("to") != "NO_CONNECTION":
            raise Failure(f"C2 fixture shell must have no internal conductor: fixture {suffix}")
        contacts = [item for item in netlist["relay_contact_map"] if item.get("ref", "").endswith(f"_{suffix}")]
        if len(contacts) != 6 or {(item["ref"], item["pole"], item["common_pin"], item["deenergized_pin"], item["energized_pin"]) for item in contacts} != {
            (f"K1_{suffix}", "signal", 3, 2, 4), (f"K1_{suffix}", "witness", 6, 7, 5),
            (f"K2_{suffix}", "signal", 3, 2, 4), (f"K2_{suffix}", "witness", 6, 7, 5),
            (f"K3_{suffix}", "signal", 3, 2, 4), (f"K3_{suffix}", "witness", 6, 7, 5),
        }:
            raise Failure(f"relay contact map: fixture {suffix}")
        drivers = [item for item in netlist["relay_coil_driver_map"] if item.get("relay", "").endswith(f"_{suffix}")]
        if len(drivers) != 3 or {item["isolator_output"] for item in drivers} != {f"UISO_RELAY_{suffix}.12", f"UISO_RELAY_{suffix}.13", f"UISO_RELAY_{suffix}.14"}:
            raise Failure(f"relay coil-driver map: fixture {suffix}")
        access = [item for item in netlist["test_point_map"] if item.get("id", "").endswith(f"-{suffix}")]
        if len(access) != 8 or [item for item in access if item["id"] == f"TA-ELECTRODE-{suffix}"] != [{"access": "NONE", "assembly": f"P0-{('DUT' if suffix == 'A' else 'DETECTOR' if suffix == 'B' else 'DUMMY-C0')}-{suffix}", "dedicated_hardware": "FORBIDDEN", "id": f"TA-ELECTRODE-{suffix}", "law": f"no test point, cable, clamp or added copper on N_ELECTRODE_A_{suffix}"}]:
            raise Failure(f"test-access/electrode loading law: fixture {suffix}")
    witness = netlist["witness_law"]
    expected_centroids = []
    resistors = [80600.0, 40200.0, 20000.0, 10000.0]
    for code in range(16):
        conductance = 1.0 / 1000.0 + 1.0 / resistors[0]
        numerator = 3.3 / resistors[0] if code & 1 else 0.0
        for bit, resistance in enumerate(resistors[1:], start=1):
            if code & (1 << bit):
                conductance += 1.0 / resistance
                numerator += 3.3 / resistance
        expected_centroids.append({"code": code, "millivolts": round(1000.0 * numerator / conductance, 6)})
    minimum_gap = round(min(expected_centroids[index]["millivolts"] - expected_centroids[index - 1]["millivolts"] for index in range(1, 16)), 6)
    if witness.get("nominal_centroids_mv") != expected_centroids or witness.get("minimum_nominal_adjacent_gap_mv") != minimum_gap or expected_centroids[8]["millivolts"] != 296.654026 or minimum_gap != 24.996066 or witness.get("nominal_reference_v") != 3.3 or witness.get("pulldown_ohms") != 1000.0 or witness.get("stable_off_code") != 8 or "R0 is always present" not in witness.get("equation", ""):
        raise Failure("witness resistor calculation and ordered nominal centroids")
    budget = netlist["admittance_budget"]
    if budget["hard_execution"] != {"cin_u95_pf_max": 4.0, "rin_u95_mohm_min": 100.0} or budget["planning_pf"]["total"] != 3.6 or budget["planning_pf"]["closure_margin"] != 0.4:
        raise Failure("sense admittance budget")
    protection = budget.get("input_protection", {})
    if protection.get("selection") != "NONE" or protection.get("external_components") != [] or "100 kohm source limiter" not in protection.get("operating_boundary", "") or "grounded ESD controls" not in protection.get("handling_boundary", ""):
        raise Failure("sense input-protection/loading law")
    source = netlist["source_off_sequence"]
    if set(source) != {"acquisition", "contact_limit_samples", "drive", "first_admissible", "gate", "guard_samples", "relay_order", "relay_release_delay_us", "series_open_stable_samples", "signal_pole_evidence_boundary", "source_persistence_proof", "source_setup", "stable_off_samples"}:
        raise Failure("source-off sequence exact fields")
    setup = source["source_setup"]
    if set(setup) != {"c1", "c2", "conservative_c1_resistive_bound", "load_mode", "model", "output_mode", "physical_output_ohms", "qualified_preparation_cycles", "queryback_fields_per_channel"}:
        raise Failure("source setup exact fields")
    if setup["c1"] != {"amplitude_vpp": 0.4, "frequency_hz": 32768, "offset_v": 0.0, "phase_command_rad": [0.0, 3.141592653589793]} or setup["c2"] != {"amplitude_vpp": 0.1, "frequency_hz": 65536, "offset_v": 0.0, "phase_command_rad": 0.0}:
        raise Failure("source frequency/amplitude/offset/phase setup")
    if setup["model"] != "SIGLENT SDG1032X" or setup["load_mode"] != "HIGH_Z" or setup["physical_output_ohms"] != 50.0 or setup["output_mode"] != "CONTINUOUS_SINE" or setup["qualified_preparation_cycles"] != 32768:
        raise Failure("source identity/load/mode setup")
    if setup["conservative_c1_resistive_bound"] != {"carrier_esr_ohms_max": 70000.0, "carrier_terminal_vpp_max": 0.164658, "motional_current_ua_max": 0.831646, "motional_power_uw_max": 0.048415, "series_limiter_ohms": 100000.0, "source_output_ohms": 50.0, "source_vrms": 0.141421356}:
        raise Failure("source conservative drive bound")
    if setup["queryback_fields_per_channel"] != ["model", "serial", "firmware", "load_mode", "physical_output_ohms", "waveform", "frequency_hz", "amplitude_vpp", "offset_v", "phase_command_rad", "output_mode", "output_state"]:
        raise Failure("source queryback contract")
    expected_signal_boundary = {
        "accepted_prerequisites": ["PER_EVENT_ACTUAL_SIGNAL_PATH_WITNESS", "EXACT_FORCE_GUIDED_CONTACT_GUARANTEE"],
        "auxiliary_contacts_sufficient": False,
        "physical_execution_authorized": False,
        "source_disconnect_claim_authorized": False,
    }
    expected_persistence = "CH0 must match the sample-level reconstructed 32768 Hz C1 plus 65536 Hz C2 waveform from t_gate through record end with peak residual <=5 percent of the fitted C1 amplitude, and both tones must remain within 2 percent amplitude and 0.010 rad phase in contiguous 100000-sample segments beginning at t_gate and covering record end; the bounded passive C2-to-C1 coupling through the monitor network is modeled and controlled rather than called zero"
    if "continuous" not in source["drive"].lower() or "0.400 Vpp" not in source["drive"] or "0.100 Vpp" not in source["drive"] or source["source_persistence_proof"] != expected_persistence or source["relay_release_delay_us"] != 250 or source["series_open_stable_samples"] != 1000 or source["stable_off_samples"] != 1000 or source["contact_limit_samples"] != 14500 or "deenergize K1 and K2" not in source["relay_order"] or source["signal_pole_evidence_boundary"] != expected_signal_boundary or netlist["ground_model"]["external_trigger"] != "absent":
        raise Failure("continuous dual-tone/no-trigger source-off law")
    if netlist["source_off_state_table"] != expected_source_states:
        raise Failure("ordered K1/K2-open-before-K3-guard state machine")
    return {"boards": 12, "channels": 12, "components": len(components), "connectors": 18, "fixtures": 3, "nets": len(actual_nets), "sha256": sha256(snapshot["P0_FINAL_NETLIST.json"])}


def check_bom(snapshot: Mapping[str, bytes], netlist: dict[str, Any] | None = None) -> dict[str, Any]:
    bom = parse_json("P0_NONPURCHASING_BOM.json", snapshot)
    if bom.get("schema") != "p0.nonpurchasing-bom.v2" or bom.get("authority") != AUTHORITY or bom.get("supplier") != "not selected or contacted" or bom.get("price") != "not requested or frozen":
        raise Failure("BOM schema/authority/non-purchasing law")
    items = bom.get("items")
    if not isinstance(items, list) or not items or [item.get("line") for item in items] != list(range(1, len(items) + 1)):
        raise Failure("BOM line sequence")
    required_fields = {"allowed_substitution", "assembly_gate", "category", "conservative_limit", "critical_value", "datasheet_hash", "execution_gate", "forbidden_substitution", "function", "line", "manufacturer", "netlist_refs", "package", "part_number", "procurement_gate", "quantity"}
    for item in items:
        if set(item) != required_fields or not isinstance(item["quantity"], int) or isinstance(item["quantity"], bool) or item["quantity"] < 1 or item["allowed_substitution"] != "NONE":
            raise Failure(f"BOM item schema/type: line {item.get('line')}")
    siglent = [item for item in items if item["manufacturer"] == "SIGLENT" and item["part_number"] == "SDG1032X"]
    if len(siglent) != 1 or siglent[0]["datasheet_hash"] != "LEGACY_EXPECTED_SHA256__BYTES_NOT_LOCAL__11c325f98fea514659be9790a001e90e445119584e31fd8d796b33e92d6e4bed":
        raise Failure("SIGLENT manual custody label")
    nl = parse_json("P0_FINAL_NETLIST.json", snapshot) if netlist is None else netlist
    grouped: dict[tuple[str, str], list[str]] = {}
    for component in nl["components"]:
        grouped.setdefault((component["manufacturer"], component["part_number"]), []).append(component["ref"])
    bom_groups = {(item["manufacturer"], item["part_number"]): item for item in items if item["netlist_refs"]}
    if set(bom_groups) != set(grouped):
        raise Failure("BOM does not exactly cover netlist manufacturer/part identities")
    for key, refs in grouped.items():
        item = bom_groups[key]
        if item["quantity"] != len(refs) or item["netlist_refs"] != sorted(refs):
            raise Failure(f"BOM count/reference mismatch: {key}")
    if bom["electrical_component_count"] != len(nl["components"]):
        raise Failure("BOM electrical component total")
    expected_mechanical_inventory = {
        "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A": 6,
        "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B": 3,
        "P0-PEEK-M3-STANDOFF-REV-A": 24,
        "P0-PEEK-M2P5-STANDOFF-REV-A": 12,
        "P0-PEEK-WASHER-M3-6X3P2X0P5-REV-A": 24,
        "P0-PEEK-WASHER-M2P5-5X2P7X0P5-REV-A": 12,
        "ISO 4762 M3x0.5x8 A2-70": 60,
        "ISO 4762 M3x0.5x6 A2-70": 24,
        "ISO 4762 M2.5x0.45x5 A2-70": 12,
        "P0-PEEK-CSK-M2X0P4X6-REV-A": 12,
    }
    actual_mechanical_inventory = {item["part_number"]: item["quantity"] for item in items if item["part_number"] in expected_mechanical_inventory}
    if actual_mechanical_inventory != expected_mechanical_inventory:
        raise Failure("complete enclosure/retention mechanical sub-inventory")
    return {"electrical_component_count": len(nl["components"]), "line_count": len(items), "sha256": sha256(snapshot["P0_NONPURCHASING_BOM.json"])}


def check_fabrication(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    netlist = parse_json("P0_FINAL_NETLIST.json", snapshot)
    fab = parse_json("P0_PCB_FABRICATION_RELEASE.json", snapshot)
    expected_root_fields = {"authority", "boards", "enclosure_design", "enclosures", "fabrication_authority", "harness", "mechanical_clearance", "schema", "status"}
    if set(fab) != expected_root_fields:
        raise Failure("fabrication release exact root fields")
    if fab.get("schema") != "p0.pcb-fabrication-release.v1" or fab.get("authority") != AUTHORITY or "NOT_AUTHORIZED" not in fab.get("status", "") or fab.get("fabrication_authority") != "NONE; coordinates are a reviewed prospective release only":
        raise Failure("fabrication release authority/schema")
    enclosure_design = fab["enclosure_design"]
    if sha256(canonical(enclosure_design)) != "78d8e4a4f04070fbc852f916f9d467b00a310ab863b9c1182259ba43f3156aa5":
        raise Failure("exact custom enclosure design identity")
    if enclosure_design.get("schema") != "p0.custom-enclosure-design.v1" or enclosure_design.get("design_id") != "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A" or enclosure_design.get("clear_internal_min_mm") != [124.0, 82.0, 36.0] or enclosure_design.get("body_outer_mm") != [140.0, 98.0, 41.0] or enclosure_design.get("closed_outer_mm") != [140.0, 98.0, 44.0]:
        raise Failure("custom enclosure dimensional identity")
    if enclosure_design.get("material") != "ASTM B209 6061-T6 aluminum plate or billet with lot and mill-certificate custody" or enclosure_design.get("finish") != "bare machined aluminum, deburred and solvent-cleaned; no anodize, paint, conversion coating, gasket, or insulating film on body, lid, fastener seats, or seam":
        raise Failure("custom enclosure material/finish identity")
    if len(enclosure_design.get("lid_fasteners", {}).get("positions_outer_xy_mm", [])) != 10 or len(enclosure_design.get("retention_systems", [])) != 3 or len(enclosure_design.get("mechanical_acceptance", [])) != 8:
        raise Failure("custom enclosure retention/acceptance completeness")
    tray = [item for item in enclosure_design["retention_systems"] if item.get("id") == "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B"]
    if len(tray) != 1 or set(tray[0]) != {"anchor_hole", "anchor_holes_tray_xy_mm", "assembly_method", "board_admission_outline_mm", "board_seat", "clip_profile", "floor_fastener", "id", "insertion_clearance_budget_mm", "integral_clip_contact_rectangles_board_xy_mm", "keepout", "material", "retention_acceptance", "tolerances_mm", "tray_origin_from_board_origin_mm", "tray_outer_mm"} or len(tray[0]["retention_acceptance"]) != 6:
        raise Failure("fabrication-complete Nucleo tray identity")
    tray = tray[0]
    if tray["tray_outer_mm"] != [56.0, 24.0, 3.0] or tray["tray_origin_from_board_origin_mm"] != [-3.0, -3.0, -3.0] or tray["board_seat"].get("lower_left_tray_xy_mm") != [3.0, 3.0] or tray["board_seat"].get("envelope_mm") != [50.2, 18.2, 1.8]:
        raise Failure("Nucleo tray/seat geometry")
    if tray["board_admission_outline_mm"] != {"law": "measure the bare PCB edge outline before insertion; reject any board outside both closed intervals", "measured_max": [50.05, 18.05], "measured_min": [49.95, 17.95], "nominal": [50.0, 18.0]}:
        raise Failure("Nucleo admitted-board envelope")
    if tray["integral_clip_contact_rectangles_board_xy_mm"] != [[0.0, 0.0, 1.0, 4.0], [49.0, 0.0, 1.0, 4.0], [0.0, 14.0, 1.0, 4.0], [49.0, 14.0, 1.0, 4.0]] or tray["clip_profile"].get("overhang_inward_mm") != 0.4:
        raise Failure("in-outline Nucleo clip contacts/profile")
    expected_clearance_budget = {"admitted_board_span_max": [50.05, 18.05], "allowed_outward_clip_deflection_max": 0.5, "outward_clip_deflection_required_max": 0.425, "overhang_inward_max": 0.45, "per_side_clearance_min": [0.025, 0.025], "seat_span_min": [50.1, 18.1]}
    if tray["insertion_clearance_budget_mm"] != expected_clearance_budget or tray["insertion_clearance_budget_mm"]["outward_clip_deflection_required_max"] > tray["insertion_clearance_budget_mm"]["allowed_outward_clip_deflection_max"]:
        raise Failure("Nucleo clip worst-case displacement budget")
    if tray["anchor_holes_tray_xy_mm"] != [[2.5, 2.5], [53.5, 2.5], [2.5, 21.5], [53.5, 21.5]]:
        raise Failure("Nucleo tray anchor coordinates")
    if sha256(canonical(fab["enclosures"])) != "7e4f4d3979c9caf95e57e8fe44d69938a7992f631368cf6dddb600d58cd28e79":
        raise Failure("exact enclosure instance rows")
    boards = {item["board_id"]: item for item in fab["boards"]}
    expected_boards = {item["board_id"] for item in netlist["boards"] if not item["board_id"].startswith("NUCLEO")}
    if set(boards) != expected_boards or len(boards) != 9 or len(fab["enclosures"]) != 6:
        raise Failure("fabrication board/enclosure coverage")
    control_enclosures = {row["enclosure"]: row for row in fab["enclosures"] if row["enclosure"].startswith("ENC-CONTROL-")}
    if set(control_enclosures) != {"ENC-CONTROL-A", "ENC-CONTROL-B", "ENC-CONTROL-C"}:
        raise Failure("exact control enclosure instances")
    transform = {"board_local_xyz_to_machine_xyz": "[x_m,y_m,z_m]=[117-y_b,16+x_b,3+z_b]", "floor_anchor_centers_machine_xy_mm": [[117.5, 15.5], [117.5, 66.5], [98.5, 15.5], [98.5, 66.5]], "tray_axes_machine": "+x_tray=+y_machine; +y_tray=-x_machine; +z_tray=+z_machine", "tray_footprint_machine_xy_closed_mm": [[96.0, 13.0], [120.0, 69.0]], "tray_local_xyz_to_machine_xyz": "[x_m,y_m,z_m]=[120-y_t,13+x_t,z_t]", "tray_origin_machine_mm": [120.0, 13.0, 0.0]}
    for suffix in ("A", "B", "C"):
        mounts = [mount for mount in control_enclosures[f"ENC-CONTROL-{suffix}"]["mounts"] if mount.get("object") == f"NUCLEO-G031K8-{suffix}"]
        expected_mount = {"local_to_machine_transform": transform, "object": f"NUCLEO-G031K8-{suffix}", "origin_datum": "board-local lower-left underside [0,0,0] at the tray seat", "origin_mm": [117.0, 16.0, 3.0], "retention_id": "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B", "rotation_axis": "machine +z through origin_mm", "rotation_convention": "counterclockwise from machine +x toward machine +y when viewed from +z toward the floor", "rotation_deg": 90, "standoff_mm": 3.0}
        if mounts != [expected_mount]:
            raise Failure(f"unique Nucleo tray machine transform: {suffix}")
    by_board: dict[str, set[str]] = {board: set() for board in expected_boards}
    for component in netlist["components"]:
        if component["board"] in by_board:
            by_board[component["board"]].add(component["ref"])
    for board_id, release in boards.items():
        placements = release["placements"]
        refs = [item["ref"] for item in placements]
        if len(refs) != len(set(refs)) or set(refs) != by_board[board_id] or len(release["mount_holes_mm"]) != 4:
            raise Failure(f"fabrication placement coverage: {board_id}")
        width, height, _ = release["outline_mm"]
        if any(not (0 < item["x_mm"] < width and 0 < item["y_mm"] < height) for item in placements):
            raise Failure(f"fabrication placement outside outline: {board_id}")
    outlines = {item["board_id"]: item["outline_mm"] for item in netlist["boards"]}
    clearances = fab.get("mechanical_clearance", {})
    if clearances != {"board_to_board_mm_min": 4.0, "board_to_lid_mm_min": 5.0, "board_to_wall_mm_min": 4.0, "coax_bend_radius_mm_min": 15.0, "gland_to_board_mm_min": 8.0}:
        raise Failure("mechanical clearance law")
    retention_ids = {item["id"] for item in enclosure_design["retention_systems"]}
    seen_mounts: set[str] = set()
    seen_panel_holes: set[str] = set()
    for enclosure in fab["enclosures"]:
        if set(enclosure) != {"assembly", "enclosure", "minimum_clear_envelope_mm", "mounts", "panel_holes", "part_number"} or not enclosure["panel_holes"] or not enclosure["mounts"] or enclosure["part_number"] != "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A" or enclosure["minimum_clear_envelope_mm"] != [124.0, 82.0, 36.0]:
            raise Failure(f"incomplete enclosure coordinates: {enclosure.get('enclosure')}")
        envelope_x, envelope_y, envelope_z = enclosure["minimum_clear_envelope_mm"]
        mounted: list[tuple[str, float, float, float, float, float, float]] = []
        for mount in enclosure["mounts"]:
            base_mount_fields = {"object", "origin_mm", "retention_id", "rotation_deg", "standoff_mm"}
            tray_mount_fields = base_mount_fields | {"local_to_machine_transform", "origin_datum", "rotation_axis", "rotation_convention"}
            expected_mount_fields = tray_mount_fields if mount.get("retention_id") == "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B" else base_mount_fields
            if set(mount) != expected_mount_fields or mount["object"] not in outlines or mount["retention_id"] not in retention_ids or mount["rotation_deg"] not in {0, 90, 180, 270} or mount["object"] in seen_mounts:
                raise Failure(f"mount identity/rotation: {enclosure['enclosure']}")
            seen_mounts.add(mount["object"])
            width, height, thickness = outlines[mount["object"]]
            board_x0, board_y0, board_z0 = mount["origin_mm"]
            if mount["standoff_mm"] != board_z0:
                raise Failure(f"mount standoff mismatch: {enclosure['enclosure']}:{mount['object']}")
            if mount["retention_id"] == "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B":
                footprint = mount["local_to_machine_transform"]["tray_footprint_machine_xy_closed_mm"]
                x0, y0 = footprint[0]
                x1, y1 = footprint[1]
                z0, z1 = 0.0, board_z0 + thickness
            else:
                if mount["rotation_deg"] in {90, 270}:
                    width, height = height, width
                x0, y0, z0 = board_x0, board_y0, board_z0
                x1, y1, z1 = x0 + width, y0 + height, z0 + thickness
            if x0 < clearances["board_to_wall_mm_min"] or y0 < clearances["board_to_wall_mm_min"] or envelope_x - x1 < clearances["board_to_wall_mm_min"] or envelope_y - y1 < clearances["board_to_wall_mm_min"] or envelope_z - z1 < clearances["board_to_lid_mm_min"]:
                raise Failure(f"mounted board outside clearance envelope: {enclosure['enclosure']}:{mount['object']}")
            mounted.append((mount["object"], x0, y0, z0, x1, y1, z1))
        for index, first in enumerate(mounted):
            for second in mounted[index + 1 :]:
                z_overlap = first[3] < second[6] and second[3] < first[6]
                x_gap = max(second[1] - first[4], first[1] - second[4])
                y_gap = max(second[2] - first[5], first[2] - second[5])
                if z_overlap and x_gap < clearances["board_to_board_mm_min"] and y_gap < clearances["board_to_board_mm_min"]:
                    raise Failure(f"mounted board collision/clearance: {enclosure['enclosure']}:{first[0]}:{second[0]}")
        holes_by_face: dict[str, list[tuple[str, float, float, float]]] = {}
        face_bounds = {"east": (envelope_z, envelope_y), "floor": (envelope_x, envelope_y), "north": (envelope_x, envelope_z), "south": (envelope_x, envelope_z), "west": (envelope_z, envelope_y)}
        for hole in enclosure["panel_holes"]:
            hole_id = hole.get("id")
            face = hole.get("face")
            if not isinstance(hole_id, str) or hole_id in seen_panel_holes or face not in face_bounds or not isinstance(hole.get("center_mm"), list) or len(hole["center_mm"]) != 2:
                raise Failure(f"panel-hole identity/schema: {enclosure['enclosure']}:{hole_id}")
            seen_panel_holes.add(hole_id)
            if face == "floor":
                if set(hole) != {"center_mm", "diameter_mm", "face", "id", "insulated_from_enclosure"} or hole.get("insulated_from_enclosure") is not True:
                    raise Failure(f"floor-hole identity/schema: {hole_id}")
            elif set(hole) != {"center_mm", "counterbore_depth_mm", "counterbore_diameter_mm", "diameter_mm", "face", "id"} or hole.get("counterbore_depth_mm") != 5.0:
                raise Failure(f"vertical-panel-hole identity/schema: {hole_id}")
            radius = max(float(hole["diameter_mm"]), float(hole.get("counterbore_diameter_mm", hole["diameter_mm"]))) / 2.0
            first_coordinate, second_coordinate = (float(value) for value in hole["center_mm"])
            first_bound, second_bound = face_bounds[face]
            if not (radius <= first_coordinate <= first_bound - radius and radius <= second_coordinate <= second_bound - radius):
                raise Failure(f"panel hole outside face: {enclosure['enclosure']}:{hole_id}")
            for other_id, other_first, other_second, other_radius in holes_by_face.setdefault(face, []):
                if (first_coordinate - other_first) ** 2 + (second_coordinate - other_second) ** 2 < (radius + other_radius) ** 2:
                    raise Failure(f"panel hole overlap: {enclosure['enclosure']}:{other_id}:{hole_id}")
            holes_by_face[face].append((hole_id, first_coordinate, second_coordinate, radius))
        if enclosure["enclosure"].startswith("ENC-CONTROL-"):
            if envelope_y - max(item[5] for item in mounted) < clearances["gland_to_board_mm_min"]:
                raise Failure(f"north-gland board clearance: {enclosure['enclosure']}")
        elif min(item[2] for item in mounted) < clearances["gland_to_board_mm_min"]:
            raise Failure(f"south-gland board clearance: {enclosure['enclosure']}")
    expected_panel_holes = {item["id"] for item in netlist["connector_map"]} | {f"GLAND-C-{suffix}" for suffix in "ABC"} | {f"GLAND-R-{suffix}" for suffix in "ABC"} | {f"AGND-STAR-{suffix}" for suffix in "ABC"}
    if seen_mounts != set(outlines) or seen_panel_holes != expected_panel_holes:
        raise Failure("complete unique mount/panel-hole coverage")
    return {"board_instances": 9, "enclosure_instances": 6, "placement_count": sum(len(item["placements"]) for item in boards.values()), "sha256": sha256(snapshot["P0_PCB_FABRICATION_RELEASE.json"])}


def check_documents(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    docs = parse_json("P0_COMPONENT_DOCUMENTS.json", snapshot)
    if set(docs) != {"authority", "automated_public_source_requests_recorded", "records", "research_bundle", "schema", "zero_human_vendor_outreach"} or docs.get("schema") != "p0.component-documents.v2" or docs.get("authority") != AUTHORITY or docs.get("zero_human_vendor_outreach") is not True or docs.get("automated_public_source_requests_recorded") is not True:
        raise Failure("component-document registry")
    records = docs.get("records")
    if not isinstance(records, list) or not records:
        raise Failure("component-document records")
    ids = [record.get("document_id") for record in records]
    if len(ids) != len(set(ids)):
        raise Failure("duplicate component-document identity")
    netlist = parse_json("P0_FINAL_NETLIST.json", snapshot)
    required = {item["exact_document"] for item in netlist["components"]}
    custody_name = f"{RESEARCH_PREFIX}/SOURCE_CUSTODY.json"
    manifest_name = f"{RESEARCH_PREFIX}/MANIFEST.json"
    custody = parse_json(custody_name, snapshot)
    custody_by_id = {record["source_id"]: record for record in custody["records"]}
    core_ids = {identity for identity, record in custody_by_id.items() if record["collection"] == "core_component_document"}
    if set(ids) != core_ids or not required.issubset(core_ids) or len(records) != 24:
        raise Failure("component document exact coverage")
    record_fields = {
        "bytes", "custody_state", "direct_download_url", "document_id", "download_result",
        "download_result_detail", "legacy_expected_bytes", "legacy_expected_sha256",
        "license_or_redistribution_note", "official_url", "publisher", "relevance_to_p0",
        "revision", "sha256", "title",
    }
    state_counts: dict[str, int] = {}
    for record in records:
        if set(record) != record_fields:
            raise Failure(f"component document exact fields: {record.get('document_id')}")
        source = custody_by_id.get(record["document_id"])
        if source is None or source["collection"] != "core_component_document":
            raise Failure(f"component document missing research source: {record['document_id']}")
        expected = {
            "bytes": source["current_bytes"], "custody_state": source["custody_state"],
            "direct_download_url": source["direct_download_url"], "document_id": source["source_id"],
            "download_result": source["download_result"], "download_result_detail": source["download_result_detail"],
            "legacy_expected_bytes": source["legacy_expected_bytes"], "legacy_expected_sha256": source["legacy_expected_sha256"],
            "license_or_redistribution_note": source["license_or_redistribution_note"],
            "official_url": source["official_product_page"], "publisher": source["publisher"],
            "relevance_to_p0": source["relevance_to_p0"], "revision": source["revision_and_date"],
            "sha256": source["current_sha256"], "title": source["title"],
        }
        if record != expected:
            raise Failure(f"component/research custody mismatch: {record['document_id']}")
        state = record["custody_state"]
        state_counts[state] = state_counts.get(state, 0) + 1
        if state.startswith("LOCAL_"):
            if not isinstance(record["bytes"], int) or isinstance(record["bytes"], bool) or record["bytes"] < 1 or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64 or any(character not in HEX for character in record["sha256"]):
                raise Failure(f"component local-byte custody: {record['document_id']}")
        elif record["bytes"] is not None or record["sha256"] is not None:
            raise Failure(f"component absent-byte custody: {record['document_id']}")
    research = docs["research_bundle"]
    expected_research = {
        "canonical_relative_path": RESEARCH_PREFIX,
        "custody_snapshot_sha256": sha256(snapshot[custody_name]),
        "custody_state_counts": custody["custody_state_counts"],
        "manifest_schema": "p0.research-bundle-manifest.v1",
        "manifest_sha256": sha256(snapshot[manifest_name]),
        "source_commit": RESEARCH_SOURCE_COMMIT,
        "source_record_count": 35,
        "third_party_byte_policy": custody["third_party_byte_policy"],
    }
    if research != expected_research:
        raise Failure("component registry research-bundle binding")
    return {
        "component_custody_state_counts": dict(sorted(state_counts.items())),
        "records": len(records),
        "research_custody_state_counts": custody["custody_state_counts"],
        "research_record_count": 35,
        "sha256": sha256(snapshot["P0_COMPONENT_DOCUMENTS.json"]),
    }


def check_analyzer(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    fixtures = parse_json("P0_SCIENTIFIC_FIXTURES.json", snapshot)
    results = parse_json("P0_ANALYZER_REFERENCE_RESULTS.json", snapshot)
    schemas = parse_json("P0_BUILD_READINESS_SCHEMAS.json", snapshot)
    counts = (len(fixtures.get("positive", [])), len(fixtures.get("scientific_negative", [])), len(fixtures.get("malformed_or_custody_negative", [])))
    if counts != (8, 32, 15):
        raise Failure(f"fixture counts: {counts}")
    if len(fixtures.get("raw_adversary", [])) != 31 or fixtures.get("scope_law") != {"raw_adversaries": "actual canonical raw-bundle analyzer execution", "semantic_controls": "summary schema and decision-law conformance only", "topology_only_cases": "require separately bound non-analyzer adjudication receipts"}:
        raise Failure("raw/semantic fixture scope")
    if results.get("schema") != "p0.analyzer-reference-results.v1" or results.get("fixture_count") != 55 or results.get("positive_count") != 8 or results.get("scientific_negative_count") != 32 or results.get("malformed_or_custody_negative_count") != 15 or results.get("raw_adversary_count") != 31:
        raise Failure("reference-result counts/schema")
    if results.get("physical_claim_authorized") is not False or results.get("claim_ceiling") != CEILING:
        raise Failure("analyzer physical-claim fence")
    if any(item.get("outcome") != "PASS" for item in results["semantic_outcomes"] + results["malformed_outcomes"] + results["raw_adversary_outcomes"]):
        raise Failure("reference outcome failure")
    if any(item.get("execution") != "actual_bundle_analyzer" for item in results["malformed_outcomes"] + results["raw_adversary_outcomes"]):
        raise Failure("raw/malformed adversary did not execute actual bundle analyzer")
    required_raw = {"source_muted_preparation_middle_raw", "source_muted_at_gate_raw", "source_muted_gate_guard_only_raw", "source_muted_late_1300000_raw", "source_muted_late_2000000_raw", "source_muted_late_3000000_raw", "guard_before_series_open_raw", "matched_drive_phase_mismatch_raw", "reference_tone_missing_raw", "source_amplitude_out_of_contract", "environment_crc_mutation", "environment_monotonic_drift", "environment_sensor_identity_drift", "instrument_configuration_drift", "source_configuration_drift"}
    if not required_raw.issubset({item["case"] for item in results["raw_adversary_outcomes"]}):
        raise Failure("missing raw adversary")
    custody = results.get("artifact_custody", {})
    if custody.get("analyzer_sha256") != sha256(snapshot["p0_scientific_analyzer.py"]) or custody.get("fixture_sha256") != sha256(snapshot["P0_SCIENTIFIC_FIXTURES.json"]) or custody.get("schema_sha256") != sha256(snapshot["P0_BUILD_READINESS_SCHEMAS.json"]):
        raise Failure("analyzer artifact custody")
    raw = results.get("raw_numerical_reference", {})
    if raw.get("scientific_pass") is not True or raw.get("physical_claim_authorized") is not False or raw.get("matched_guard_samples") != 12499 or raw.get("claim_token") != "SYNTHETIC_ANALYZER_PASS":
        raise Failure("raw numerical reference")
    # The zero-drive, resonator-removed, and exact-C0 dummy records are the
    # mandatory physical-topology null baselines; a positive pair alone cannot
    # satisfy the analyzer custody gate.
    null_payloads = raw.get("input_custody", {}).get("payload_sha256", {})
    if set(null_payloads) != {"arm_0", "arm_pi", "dummy_c0", "resonator_removed", "zero_drive"}:
        raise Failure("exact positive and null-baseline payload custody")
    relation = raw.get("relation_metrics", {})
    if relation.get("common_window_count", 0) % 8 != 0 or relation.get("jackknife_blocks") != relation.get("common_window_count", 0) // 8 or relation.get("common_window_count_before_blocking", 0) - relation.get("common_window_count", 0) != relation.get("discarded_incomplete_block_windows") or not 0 <= relation.get("discarded_incomplete_block_windows", -1) < 8:
        raise Failure("identical blocked common grid for relation point estimate and jackknife")
    drive_phase = raw.get("drive_phase", {})
    if set(drive_phase) != {"arm_0", "arm_pi", "matched_acceptance_lhs_rad", "matched_error_difference_rad", "matched_u95_sum_rad"} or drive_phase.get("matched_acceptance_lhs_rad", 1.0) > 0.010:
        raise Failure("matched drive-phase uncertainty law")
    for role in ("arm_0", "arm_pi"):
        drive = drive_phase.get(role, {})
        expected_drive_fields = {"drive_reference_phase_covariance_rad2", "error_phase_fit_standard_uncertainty_rad", "error_rad", "gauge_phase_fit_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "phase_fit_standard_uncertainty_rad", "phase_skew_standard_uncertainty_rad", "preparation_first_sample", "preparation_last_sample", "preparation_residual_ratio_max", "reference_phase_fit_standard_uncertainty_rad", "u95_drive_rad"}
        if set(drive) != expected_drive_fields or any(not isinstance(drive.get(key), (int, float)) or isinstance(drive.get(key), bool) for key in expected_drive_fields - {"preparation_first_sample", "preparation_last_sample"}) or drive.get("reference_phase_fit_standard_uncertainty_rad", -1.0) < 0 or drive.get("gauge_phase_fit_standard_uncertainty_rad") != 0.5 * drive.get("reference_phase_fit_standard_uncertainty_rad", -1.0) or drive.get("error_phase_fit_standard_uncertainty_rad", -1.0) < 0 or abs(drive.get("error_rad", 1.0)) + drive.get("u95_drive_rad", 1.0) > 0.010 or drive.get("preparation_first_sample") != raw["timing"][role]["n_gate"] - 1_000_000 or drive.get("preparation_last_sample") != raw["timing"][role]["n_gate"] - 1 or drive.get("preparation_residual_ratio_max", 1.0) > 0.05:
            raise Failure(f"per-arm drive/preparation uncertainty law: {role}")
        reconstructed_fit_variance = drive["phase_fit_standard_uncertainty_rad"] ** 2 + drive["gauge_phase_fit_standard_uncertainty_rad"] ** 2 - drive["drive_reference_phase_covariance_rad2"]
        if reconstructed_fit_variance < 0 or not math.isclose(drive["error_phase_fit_standard_uncertainty_rad"] ** 2, reconstructed_fit_variance, rel_tol=1e-12, abs_tol=1e-24):
            raise Failure(f"joint C1/C2 phase covariance reconstruction: {role}")
        reconstructed_u95 = 1.96 * math.sqrt(reconstructed_fit_variance + drive["phase_skew_standard_uncertainty_rad"] ** 2 + drive["phase_drive_cal_standard_uncertainty_rad"] ** 2)
        if not math.isclose(drive["u95_drive_rad"], reconstructed_u95, rel_tol=1e-12, abs_tol=1e-18):
            raise Failure(f"expanded drive-phase uncertainty reconstruction: {role}")
    noise_counts = raw.get("noise_nonoverlapping_window_counts", {})
    if set(noise_counts) != {"dummy_c0", "resonator_removed", "zero_drive"} or len(set(noise_counts.values())) != 1 or any(not isinstance(value, int) or value < 8 for value in noise_counts.values()):
        raise Failure("nonoverlapping control-noise windows")
    if schemas.get("schema") != "p0.build-readiness-schemas.v1":
        raise Failure("analyzer schema identity")
    source_schema = schemas.get("enforced_objects", {}).get("p0.raw-bundle.v1", {})
    if source_schema.get("source_frozen_setup") != {"c1_amplitude_vpp": "0.4", "c1_offset_v": "0", "c2_amplitude_vpp": "0.1", "c2_offset_v": "0", "load_mode": "HIGH_Z", "physical_output_ohms": "50"}:
        raise Failure("analyzer source schema envelope")
    if not {"phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad"}.issubset(set(source_schema.get("source_exact_fields", []))):
        raise Failure("drive-phase uncertainty metadata schema")
    result_schema = schemas.get("enforced_objects", {}).get("p0.scientific-result.v1", {})
    if result_schema.get("drive_phase_uncertainty") != {"cross_covariance_included": True, "derived_error": "wrap(phi_C1-0.5*phi_C2-delta_command)", "expanded_law": "1.96*sqrt(u_error_fit^2+u_skew^2+u_drive_cal^2)", "fit_gradient": "g_C1-0.5*g_C2", "hac_lag": 7, "joint_design_columns": ["cos(f_ref)", "-sin(f_ref)", "cos(2*f_ref)", "-sin(2*f_ref)", "constant"]}:
        raise Failure("joint C1/C2 drive-phase covariance schema")
    dependency = results.get("artifact_custody", {}).get("dependency_identity", {})
    required_dependency = {"byteorder", "machine", "numpy_config", "numpy_core_binary_sha256", "numpy_distribution_record_sha256", "numpy_version", "platform", "python_cache_tag", "python_executable_sha256", "python_implementation", "python_version", "thread_environment"}
    if set(dependency) != required_dependency or dependency.get("thread_environment") != {"MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}:
        raise Failure("numerical runtime/dependency custody")
    semantic_schema = schemas.get("enforced_objects", {}).get("p0.synthetic-control-evidence.v1", {})
    if semantic_schema.get("scientific_negative_execution") != "summary_schema_and_decision_law_only__not_raw_analyzer_evidence":
        raise Failure("summary-vs-raw scientific fixture scope")
    return {"fixture_count": 55, "positive": 8, "scientific_negative": 32, "malformed_or_custody_negative": 15, "raw_adversaries": 31, "fixture_sha256": sha256(snapshot["P0_SCIENTIFIC_FIXTURES.json"]), "result_sha256": sha256(snapshot["P0_ANALYZER_REFERENCE_RESULTS.json"]), "schema_sha256": sha256(snapshot["P0_BUILD_READINESS_SCHEMAS.json"])}


def check_claims_and_findings(snapshot: Mapping[str, bytes]) -> dict[str, Any]:
    packet = text("P0_BUILD_READINESS_PACKET.md", snapshot)
    drawings = text("P0_AUTHORED_ASSEMBLY_DRAWINGS.md", snapshot)
    assembly = text("P0_UNPOWERED_ASSEMBLY_PACKET.md", snapshot)
    execution = text("P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md", snapshot)
    authority = text("P0_BUILD_READINESS_AUTHORITY.md", snapshot)
    contract = text("PHYSICAL_PHASE_CARRIER_P0_CONTRACT.md", snapshot)
    roadmap = text("../AUDIO_SIDE_QUEST_ROADMAP.md", snapshot)
    rights = text(f"{RESEARCH_PREFIX}/THIRD_PARTY_RIGHTS.md", snapshot)
    for token in (AUTHORITY, CEILING, NEXT, "No stage bootstraps authority", "No human vendor outreach", "Automated public-source HTTP retrieval attempts"):
        if token not in packet:
            raise Failure(f"packet claim/authority token: {token}")
    if assembly.count("After separate authority only") != 8 or assembly.count("UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY") != 8 or "Status: **NOT AUTHORIZED**" not in assembly or "Status: **NOT AUTHORIZED**" not in execution:
        raise Failure("assembly/execution authority fence")
    if any(f"D{index:03d}" not in drawings for index in range(1, 11)):
        raise Failure("drawing-set coverage")
    for content, name in ((authority, "authority"), (contract, "contract"), (roadmap, "roadmap")):
        if AUTHORITY not in content or CEILING not in content or NEXT not in content or "P0_BUILD_READINESS_BLOCKED" not in content:
            raise Failure(f"lane authority alignment: {name}")
    if "commits, and pushes" in authority or "commit or push without a separate explicit user instruction" not in authority or "human vendor communication or quote request" not in authority:
        raise Failure("active authority commit/push/vendor-communication fence")
    if "Git LFS" in rights or "private local artifact cache outside Git history" not in rights:
        raise Failure("third-party byte handling must remain private and outside Git")
    normative_names = (
        "P0_BUILD_READINESS_AUTHORITY.md", "PHYSICAL_PHASE_CARRIER_P0_CONTRACT.md",
        "P0_BUILD_READINESS_PACKET.md", "P0_FINAL_NETLIST.json", "P0_NONPURCHASING_BOM.json",
        "P0_AUTHORED_ASSEMBLY_DRAWINGS.md", "P0_UNPOWERED_ASSEMBLY_PACKET.md",
        "P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md", "../AUDIO_SIDE_QUEST_ROADMAP.md",
    )
    combined = b"\0".join(snapshot[name] for name in normative_names)
    for forbidden in (b"ADA4530", b"MASTER_TRIGGER", b"burst_cycles", b"USER_AUTHORITY_FOR_P0_BUILD_PURCHASE_OR_PHYSICAL_EXECUTION"):
        if forbidden in combined:
            raise Failure(f"retired design token remains: {forbidden.decode('ascii')}")
    findings = parse_json("P0_BUILD_READINESS_FINDINGS.json", snapshot)
    if findings.get("schema") != "p0.build-readiness-findings.v2" or findings.get("authority") != AUTHORITY or findings.get("claim_ceiling") != CEILING or findings.get("decision") != "P0_BUILD_READINESS_BLOCKED" or findings.get("open_material_findings") != 1:
        raise Failure("findings root/closure")
    entries = findings.get("findings")
    if not isinstance(entries, list) or len(entries) != 17:
        raise Failure("normalized finding cardinality")
    closed = [entry for entry in entries if entry.get("status") == "CLOSED"]
    opened = [entry for entry in entries if entry.get("status") == "OPEN"]
    if len(closed) != 16 or any(not entry.get("closure_evidence") for entry in closed):
        raise Failure("all sixteen repaired findings must remain explicitly closed")
    expected_open = {
        "closure_evidence": ["P0_FINAL_NETLIST.json#source_off_sequence.signal_pole_evidence_boundary", "P0_BUILD_READINESS_PACKET.md#physical-source-off-sequence", "P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md#future-source-off-contract"],
        "finding_id": "P0BR-R3-SIGNAL-POLE",
        "original_summary": "Auxiliary CH2 contacts do not provide per-event evidence that K1/K2 signal poles opened; physical source-disconnect claims remain blocked pending an actual-path witness or exact force-guided guarantee.",
        "severity": "BLOCKER",
        "status": "OPEN",
    }
    if opened != [expected_open]:
        raise Failure("exact actual-signal-pole blocker")
    if len({entry.get("finding_id") for entry in entries}) != 17:
        raise Failure("finding identities")
    attestation = findings.get("contact_attestation", {})
    if set(attestation) != {"audio_playback_or_recording", "cart_or_stock_check", "hardware", "human_vendor_outreach", "instrument_command", "purchase", "target"} or any(value != 0 for value in attestation.values()):
        raise Failure("zero-contact attestation")
    retrieval = findings.get("public_source_retrieval", {})
    if retrieval != {"automated_http_attempts_occurred": True, "repository_safe_receipt": f"{RESEARCH_PREFIX}/SOURCE_CUSTODY.json", "third_party_bytes_private_and_ignored": True}:
        raise Failure("public-source retrieval disclosure")
    historical_report = text("P0_REVIEW_REPORTS.md", snapshot)
    historical_findings = parse_json("P0_FINDINGS_NORMALIZED.json", snapshot, canonical_required=False)
    if "Historical record only" not in historical_report or "HISTORICAL_OBSOLETE_REVIEW_RECORD__NO_CURRENT_AUTHORITY" not in historical_report or historical_findings.get("active_authority") is not False or historical_findings.get("decision") != "HISTORICAL_OBSOLETE_REVIEW_RECORD__NO_CURRENT_AUTHORITY" or "no current candidate review" not in historical_findings.get("historical_scope", ""):
        raise Failure("obsolete review/finding retirement")
    return {"claim_ceiling": CEILING, "closed_findings": 16, "decision": "P0_BUILD_READINESS_BLOCKED", "open_material_findings": 1, "next_boundary": NEXT, "zero_contact": attestation}


def validate_candidate(snapshot: Mapping[str, bytes], expected_root: str | None = None) -> dict[str, Any]:
    for name in CANDIDATE_NAMES:
        if name not in snapshot:
            raise Failure(f"missing candidate snapshot member: {name}")
    root = candidate_root(snapshot)
    if expected_root is not None and root != expected_root:
        raise Failure("candidate qualification-root mismatch")
    for name in PRETTY_JSON:
        parse_json(name, snapshot)
    scripts = check_scripts(snapshot)
    research_summary = check_research(snapshot)
    netlist_summary = check_netlist(snapshot)
    bom_summary = check_bom(snapshot)
    fabrication_summary = check_fabrication(snapshot)
    document_summary = check_documents(snapshot)
    analyzer_summary = check_analyzer(snapshot)
    claims_summary = check_claims_and_findings(snapshot)
    return {"candidate_root": root, "scripts": scripts, "research": research_summary, "netlist": netlist_summary, "bom": bom_summary, "fabrication": fabrication_summary, "component_documents": document_summary, "analyzer": analyzer_summary, "claims": claims_summary}


def check_reviews(snapshot: Mapping[str, bytes], root: str) -> dict[str, Any]:
    reviews = parse_json(REVIEWS, snapshot)
    if set(reviews) != {"authority", "candidate_root", "claim_ceiling", "reviews", "schema"} or reviews["schema"] != "p0.build-readiness-reviews.v1" or reviews["authority"] != AUTHORITY or reviews["claim_ceiling"] != CEILING or reviews["candidate_root"] != root:
        raise Failure("structured review root/schema")
    entries = reviews["reviews"]
    required_roles = {"parts_and_carrier", "topology_and_source_off", "analyzer_and_evidence", "claims_safety_and_authority"}
    if not isinstance(entries, list) or len(entries) != 4 or {entry.get("role") for entry in entries} != required_roles:
        raise Failure("four exact independent review roles required")
    agent_ids = [entry.get("agent_id") for entry in entries]
    if any(not isinstance(agent_id, str) or not agent_id for agent_id in agent_ids) or len(set(agent_ids)) != 4:
        raise Failure("four distinct reviewer IDs required")
    for entry in entries:
        if set(entry) != {"agent_id", "findings", "independent", "reviewed_root", "role", "verdict"} or entry["verdict"] != "PASS" or entry["findings"] != [] or entry["independent"] is not True or entry["reviewed_root"] != root:
            raise Failure(f"review did not close cleanly: {entry.get('role')}")
    report = text("P0_BUILD_READINESS_REVIEW_REPORTS.md", snapshot)
    if root not in report or any(agent_id not in report for agent_id in agent_ids) or report.count("verdict: PASS") != 4:
        raise Failure("human-readable review report binding")
    return {"agent_ids": agent_ids, "count": 4, "open_findings": 0, "verdict": "PASS"}


def check_mutation_result(snapshot: Mapping[str, bytes], root: str) -> dict[str, Any]:
    result = parse_json(MUTATION_RESULT, snapshot)
    if result.get("schema") != "p0.build-readiness-mutation-results.v1" or result.get("candidate_root") != root or result.get("status") != "PASS" or result.get("evidence_class") != "ROOT_BINDING_TAMPER_DETECTION_ONLY__NOT_SEMANTIC_VALIDATION":
        raise Failure("mutation result root/status")
    if result.get("mutations_total") != result.get("mutations_rejected") or not isinstance(result.get("mutations_total"), int) or result["mutations_total"] < 1000 or result.get("accepted") != 0:
        raise Failure("mutation coverage/rejection")
    return result


def final_files(snapshot: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": len(snapshot[name]), "sha256": sha256(snapshot[name])} for name in sorted(CANDIDATE_NAMES + FINAL_ONLY)]


def build() -> dict[str, Any]:
    raise Failure("AUTOMATIC_FREEZE_DISABLED__EXTERNAL_REVIEW_PROVENANCE_AND_ZERO_OPEN_FINDINGS_REQUIRED")


def verify() -> dict[str, Any]:
    raise Failure("AUTOMATIC_FREEZE_DISABLED__EXTERNAL_REVIEW_PROVENANCE_AND_ZERO_OPEN_FINDINGS_REQUIRED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("candidate-root", "pre-review", "build", "verify"))
    args = parser.parse_args(argv)
    try:
        if args.mode in {"candidate-root", "pre-review"}:
            value = validate_candidate(read_snapshot())
            if args.mode == "candidate-root":
                value = {"candidate_root": value["candidate_root"]}
            else:
                value["status"] = "PASS"
        elif args.mode == "build":
            value = build()
        else:
            value = verify()
        print(canonical(value).decode("utf-8"), end="")
        return 0
    except (Failure, OSError, SyntaxError) as exc:
        print(canonical({"reason": str(exc), "status": "FAIL"}).decode("utf-8"), end="", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

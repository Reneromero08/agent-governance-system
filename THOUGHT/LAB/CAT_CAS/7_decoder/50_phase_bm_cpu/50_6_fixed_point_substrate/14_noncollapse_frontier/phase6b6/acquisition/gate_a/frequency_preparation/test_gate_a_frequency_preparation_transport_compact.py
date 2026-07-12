#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

import gate_a_frequency_preparation as core
import gate_a_frequency_preparation_authority as authority
import gate_a_frequency_preparation_bundle as bundle
import gate_a_frequency_preparation_live as live
import gate_a_frequency_preparation_transport as transport

PAYLOAD = {
    "gate_a_frequency_preparation.py": "reviewed_preparation_core",
    "gate_a_frequency_preparation_authority.py": "authority_validator",
    "gate_a_frequency_preparation_bundle.py": "target_bundle",
    "gate_a_frequency_preparation_live.py": "live_transaction",
    "gate_a_frequency_preparation_target.py": "target_runner",
}


def manifest(root: Path) -> tuple[dict[str, Any], bytes]:
    files = []
    named = []
    for index, name in enumerate(sorted(PAYLOAD), start=1):
        data = (root / name).read_bytes()
        files.append({
            "package_path": name,
            "source_repository_path": f"repo/{name}",
            "git_mode": "100644",
            "git_blob_sha1": f"{index:040x}",
            "byte_size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "role": PAYLOAD[name],
        })
        named.append((name, data))
    core_manifest = {
        "schema_id": bundle.MANIFEST_SCHEMA_ID,
        "files": files,
        "authority_artifact_created": False,
        "live_frequency_preparation_authorized": False,
        "target_contact_authorized": False,
    }
    value = dict(core_manifest)
    value["bundle_sha256"] = hashlib.sha256(bundle.canonical_bytes(core_manifest)).hexdigest()
    value["deterministic_archive_sha256"] = hashlib.sha256(bundle.payload_archive_bytes(named)).hexdigest()
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    return value, raw


def authority_value(m: dict[str, Any], raw_manifest: bytes) -> dict[str, Any]:
    authority_id = "gate_a_freqprep_deadbeef_01"
    roles = {entry["role"]: entry for entry in m["files"]}
    identity = {
        "hostname": authority.EXPECTED_HOSTNAME,
        "architecture": authority.EXPECTED_ARCHITECTURE,
        "cpu_model": authority.EXPECTED_CPU_MODEL,
    }
    return {
        "schema_id": authority.AUTHORITY_SCHEMA_ID,
        "authority_id": authority_id,
        "reviewed_source_commit": "a" * 40,
        "reviewed_source_tree_sha1": "b" * 40,
        "independent_review_id": 123,
        "bundle_sha256": m["bundle_sha256"],
        "deterministic_archive_sha256": m["deterministic_archive_sha256"],
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "source_git_blobs": {
            "host_adapter": "c" * 40,
            "host_transport": "d" * 40,
            "authority_validator": roles["authority_validator"]["git_blob_sha1"],
            "live_transaction": roles["live_transaction"]["git_blob_sha1"],
            "target_runner": roles["target_runner"]["git_blob_sha1"],
            "target_bundle": roles["target_bundle"]["git_blob_sha1"],
            "reviewed_preparation_core": roles["reviewed_preparation_core"]["git_blob_sha1"],
        },
        "target": authority.EXPECTED_TARGET,
        "target_identity": identity,
        "target_identity_sha256": authority.target_identity_digest(identity),
        "sysfs_root": authority.EXPECTED_SYSFS_ROOT,
        **authority.expected_remote_paths(authority_id),
        "required_frequency_khz": 1_600_000,
        "expected_baseline_min_khz": 800_000,
        "expected_baseline_max_khz": 3_200_000,
        "sample_count": 200,
        "sample_interval_ms": 10,
        "maximum_transaction_count": 1,
        "maximum_write_attempt_count": 8,
        "consumed": False,
        "project_owner_approved": True,
        "authority_state": {
            "authorization_artifact_created": True,
            "frequency_preparation_authorized": True,
            "restoration_authorized": True,
            "ssh_authorized": True,
            "scp_authorized": True,
            "target_filesystem_staging_authorized": True,
            "engineering_smoke_authorized": False,
            "hardware_execution_authorized": False,
            "calibration_authorized": False,
            "scientific_acquisition_authorized": False,
            "target_coupling_authorized": False,
            "small_wall_authorized": False,
            "automatic_retry": False,
        },
    }


def permit(root: Path) -> tuple[dict[str, Any], bytes, authority.PreparationPermit]:
    m, raw_manifest = manifest(root)
    value = authority_value(m, raw_manifest)
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    validated = authority.validate_authority(
        value,
        authority_bytes=raw,
        authority_sha256=hashlib.sha256(raw).hexdigest(),
        exact_manifest=m,
        expected_reviewed_source_commit="a" * 40,
        expected_independent_review_id=123,
    )
    return m, raw, validated


def synthetic_sysfs(root: Path) -> Path:
    sysfs = root / "sys"
    for cpu in (4, 5):
        policy = sysfs / "devices" / "system" / "cpu" / "cpufreq" / f"policy{cpu}"
        policy.mkdir(parents=True)
        values = {
            "scaling_driver": "acpi-cpufreq\n",
            "scaling_governor": "schedutil\n",
            "cpuinfo_min_freq": "800000\n",
            "cpuinfo_max_freq": "3200000\n",
            "scaling_min_freq": "800000\n",
            "scaling_max_freq": "3200000\n",
            "scaling_available_frequencies": "3200000 2400000 1600000 800000\n",
            "affected_cpus": f"{cpu}\n",
            "related_cpus": f"{cpu}\n",
            "scaling_cur_freq": "1600000\n",
        }
        for name, text in values.items():
            (policy / name).write_text(text, encoding="ascii")
    return sysfs


class CompactAuthorityTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent
        self.manifest, self.authority_bytes, self.permit = permit(self.root)

    def test_authority_is_opaque_and_retry_mutation_rejects(self) -> None:
        with self.assertRaises(authority.AuthorityError):
            authority.require_permit({"authority_id": self.permit.authority_id})
        value = authority_value(self.manifest, json.dumps(self.manifest, sort_keys=True, indent=2).encode() + b"\n")
        value["authority_state"]["automatic_retry"] = True
        raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
        with self.assertRaises(authority.AuthorityError):
            authority.validate_authority(
                value,
                authority_bytes=raw,
                authority_sha256=hashlib.sha256(raw).hexdigest(),
                exact_manifest=self.manifest,
                expected_reviewed_source_commit="a" * 40,
                expected_independent_review_id=123,
            )

    def test_synthetic_core_is_exactly_eight_writes_and_live_wrapper_refuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sysfs = synthetic_sysfs(Path(temporary))
            writes: list[bytes] = []

            def writer(path: Path, data: bytes) -> None:
                writes.append(data)
                path.write_bytes(data)

            receipt = core.qualify_preparation_restoration(
                sysfs_root=sysfs,
                read_bytes=lambda path: path.read_bytes(),
                write_bytes=writer,
                sleep=lambda _seconds: None,
                monotonic_ns=lambda: 1,
            )
            self.assertEqual(receipt["status"], "QUALIFIED_PREPARATION_AND_RESTORATION")
            self.assertEqual(receipt["frequency_write_attempt_count"], 8)
            self.assertEqual(receipt["retry_count"], 0)
            self.assertEqual(writes[:4], [b"1600000\n"] * 4)
            self.assertEqual(writes[4:], [b"800000\n", b"3200000\n", b"800000\n", b"3200000\n"])

            with self.assertRaises(live.LivePreparationError):
                live.execute_authorized_preparation_restoration(
                    self.permit,
                    sysfs_root=sysfs,
                    read_bytes=lambda path: path.read_bytes(),
                    write_bytes=writer,
                    sleep=lambda _seconds: None,
                    monotonic_ns=lambda: 1,
                )

    def request(self, root: Path) -> transport.TransportRequest:
        authority_path = root / "AUTHORITY.json"
        authority_path.write_bytes(self.authority_bytes)
        return transport.TransportRequest(
            permit=self.permit,
            authority_path=authority_path,
            authority_bytes=self.authority_bytes,
            manifest=self.manifest,
            manifest_bytes=json.dumps(self.manifest, sort_keys=True, indent=2).encode() + b"\n",
            deployment_archive=b"bundle",
            local_evidence_root=root / "evidence",
            source_review_binding={"reviewed_source_commit": "a" * 40},
        )

    def test_process_scan_command_hides_forbidden_markers(self) -> None:
        command = " ".join(transport._remote_python(transport.PROCESS_SCRIPT))
        self.assertNotIn("gate_a_frequency_preparation_target.py", command)
        self.assertNotIn("catcas_phase6b6_gate_a_freqprep_", command)

    def test_namespace_collision_never_consumes_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self.request(Path(temporary))
            runner = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 3, stdout=b'{"status":"collision"}\n', stderr=b"")
            result = transport.run_transport(request, runner=runner)
            self.assertEqual(result["target_contacts"], 1)
            self.assertEqual(result["transaction_invocations"], 0)
            self.assertFalse(result["claim_retained"])
            self.assertEqual(result["retry_count"], 0)

    def test_timeout_consumes_claim_and_never_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = self.request(Path(temporary))

            def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
                text = " ".join(argv)
                if "pid=,comm=,args=" in text:
                    receipt = {
                        "status": "complete",
                        "returncode": 0,
                        "hits": [],
                        "stdout_sha256": "0" * 64,
                        "stderr_sha256": "0" * 64,
                    }
                    return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(receipt).encode(), stderr=b"")
                if "-B" in argv and any(item.endswith("gate_a_frequency_preparation_target.py") for item in argv):
                    raise subprocess.TimeoutExpired(argv, 90, output=b"", stderr=b"")
                if "with tarfile.open(out" in text:
                    return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")
                return subprocess.CompletedProcess(argv, 0, stdout=b"{}\n", stderr=b"")

            result = transport.run_transport(request, runner=runner)
            self.assertEqual(result["status"], "FAILED_CLOSED_TRANSPORT")
            self.assertEqual(result["transaction_invocations"], 1)
            self.assertTrue(result["claim_retained"])
            self.assertTrue(result["remote_writer_absent"])
            self.assertEqual(result["retry_count"], 0)
            self.assertFalse(result["automatic_retry"])


if __name__ == "__main__":
    unittest.main()

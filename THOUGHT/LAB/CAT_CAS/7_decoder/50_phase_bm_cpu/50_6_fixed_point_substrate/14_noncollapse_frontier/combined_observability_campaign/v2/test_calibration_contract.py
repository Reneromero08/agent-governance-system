from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import calibration_contract
from calibration_contract import (
    FALSE_AUTHORIZATIONS,
    ROUTE_CORES,
    RUNTIME_PARAMETERS,
    build_plan,
    build_source_bundle_manifest,
    canonical_bytes,
    compile_sessions,
    validate_authorization,
    write_immutable,
)
from waveform_reference import intended_v2_gate, phase_index, tone_hz


def authorization(plan_digest: str, bundle: dict) -> dict:
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1",
        "calibration_authorized": True,
        **FALSE_AUTHORIZATIONS,
        "automatic_retry": False,
        "campaign_plan_sha256": plan_digest,
        "executor_commit": "b" * 40,
        "executor_sha256": "c" * 64,
        "source_bundle_sha256": hashlib.sha256(canonical_bytes(bundle)).hexdigest(),
        "session_ids": list(bundle["sessions"]),
        "route_cores": ROUTE_CORES,
        "pin_khz": RUNTIME_PARAMETERS["pin_khz"],
        "slot_s": RUNTIME_PARAMETERS["slot_s"],
        "off_window_s": RUNTIME_PARAMETERS["off_window_s"],
        "read_hz": RUNTIME_PARAMETERS["read_hz"],
        "temperature_veto_c": RUNTIME_PARAMETERS["temperature_veto_c"],
        "authorized_output_root": "/tmp/calibration",
        "authorized_by": "PROJECT_OWNER_TEST",
    }


class CalibrationContractTests(unittest.TestCase):
    def test_plan_is_exact_complete_and_non_authorizing(self) -> None:
        plan = build_plan()
        self.assertEqual(plan["session_count"], 4)
        self.assertEqual(plan["windows_per_session"], 588)
        self.assertEqual(plan["windows_per_route"], {"v4s5": 1176, "v2s3": 1176})
        self.assertEqual(plan["total_window_count"], 2352)
        self.assertFalse(plan["calibration_authorized"])
        for key, value in FALSE_AUTHORIZATIONS.items():
            self.assertIs(plan[key], value)
        for session in plan["sessions"]:
            rows = session["windows"]
            self.assertEqual(len(rows), 588)
            self.assertEqual([row["window_index"] for row in rows], list(range(588)))
            self.assertEqual(sum(row["drive_on"] for row in rows), 576)
            self.assertEqual(sum(row["sender_off_required"] for row in rows), 12)
            self.assertEqual({row["physical_tone_index"] for row in rows if row["drive_on"]}, set(range(12)))
            self.assertEqual({row["amplitude_level"] for row in rows if row["drive_on"]}, {1, 2, 3})
            self.assertEqual({row["receiver_theta_idx"] for row in rows if row["drive_on"]}, set(range(8)))
            self.assertEqual({row["expected_code_sign"] for row in rows if row["drive_on"]}, {-1, 1})
            driven = [row for row in rows if row["drive_on"]]
            conditions = {
                (row["physical_tone_index"], row["amplitude_level"],
                 row["sender_theta_idx"], row["expected_code_sign"])
                for row in driven
            }
            self.assertEqual(len(conditions), 12 * 3 * 8 * 2)
            self.assertEqual(len(conditions), len(driven))
            for row in driven:
                self.assertIn(
                    row["sender_codeword_source_index"],
                    calibration_contract._sources_for_sign(row["expected_code_sign"]),
                )
                if row["shared_schedule"]:
                    self.assertEqual(row["sender_codeword_source_index"],
                                     row["receiver_codeword_source_index"])
                    self.assertEqual(row["sender_theta_idx"], row["receiver_theta_idx"])
                else:
                    self.assertEqual(row["receiver_theta_idx"],
                                     (row["sender_theta_idx"] + 1) % 8)
                timestamps = np.arange(8, dtype=np.uint64)
                tsc_hz = 8.0 * tone_hz(row["physical_tone_index"])
                sender_gate = intended_v2_gate(
                    timestamps,
                    origin_tsc=0,
                    tsc_hz=tsc_hz,
                    tone_index=row["physical_tone_index"],
                    phase_index_value=phase_index(
                        0, row["sender_codeword_source_index"], row["sender_theta_idx"]
                    ),
                    amplitude_level=row["amplitude_level"],
                )
                receiver_gate = intended_v2_gate(
                    timestamps,
                    origin_tsc=0,
                    tsc_hz=tsc_hz,
                    tone_index=row["physical_tone_index"],
                    phase_index_value=phase_index(
                        0, row["receiver_codeword_source_index"], row["receiver_theta_idx"]
                    ),
                    amplitude_level=row["amplitude_level"],
                )
                sender_digest = hashlib.sha256(sender_gate.tobytes()).hexdigest()
                receiver_digest = hashlib.sha256(receiver_gate.tobytes()).hexdigest()
                self.assertEqual(sender_gate.mean(), receiver_gate.mean())
                self.assertEqual(sender_digest == receiver_digest, row["shared_schedule"])

    def test_compiled_sessions_bind_exact_ordered_plan(self) -> None:
        plan = build_plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bindings = compile_sessions(plan, root)
            self.assertEqual(set(bindings), set(plan["session_ids"]))
            for session in plan["sessions"]:
                directory = root / session["session_id"]
                self.assertEqual(
                    {path.name for path in directory.iterdir()},
                    {"session.json", "windows.jsonl", "session_manifest.json"},
                )
                rows = [json.loads(line) for line in (directory / "windows.jsonl").read_text().splitlines()]
                self.assertEqual(rows, session["windows"])

    def test_authorization_requires_exact_false_flags_and_sessions(self) -> None:
        bundle = build_source_bundle_manifest({"a": "1" * 64, "b": "2" * 64})
        auth = authorization("a" * 64, bundle)
        validate_authorization(auth, "a" * 64, bundle)
        auth["acquisition_authorized"] = True
        with self.assertRaisesRegex(ValueError, "must remain false"):
            validate_authorization(auth, "a" * 64, bundle)


class ImmutableWriterTests(unittest.TestCase):
    def test_existing_json_or_checksum_rejected(self) -> None:
        for existing in ("json", "checksum"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "artifact.json"
                target = path if existing == "json" else path.with_suffix(".json.sha256")
                target.write_text("existing")
                with self.assertRaises(FileExistsError):
                    write_immutable(path, {"a": 1})
                self.assertEqual(target.read_text(), "existing")

    def test_checksum_failure_removes_new_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "artifact.json"
            real_open = calibration_contract._open_exclusive

            def fail_sidecar(target: Path, mode: str, **kwargs):
                if target.name.endswith(".sha256"):
                    raise OSError("injected checksum failure")
                return real_open(target, mode, **kwargs)

            with mock.patch.object(calibration_contract, "_open_exclusive", fail_sidecar):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_immutable(path, {"a": 1})
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json.sha256").exists())

    def test_canonical_bytes_and_digest_are_deterministic(self) -> None:
        value = {"z": [3, 2, 1], "a": {"x": True}}
        self.assertEqual(canonical_bytes(value), canonical_bytes(value))
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.json", Path(temp) / "b.json"
            first_digest = write_immutable(first, value)
            second_digest = write_immutable(second, value)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(first_digest, hashlib.sha256(first.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

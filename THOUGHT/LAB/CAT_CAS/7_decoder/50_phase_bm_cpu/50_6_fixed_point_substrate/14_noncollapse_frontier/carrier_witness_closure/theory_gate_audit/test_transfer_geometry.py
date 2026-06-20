#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import analyze_transfer_geometry as analysis

MODE_NAMES = ("basis", "rotation", "residual", "mini")
CODEBOOK = {
    "basis": np.array([1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=int),
    "rotation": np.array([1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1], dtype=int),
    "residual": np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1], dtype=int),
    "mini": np.array([1,1,-1,1,-1,-1,1,-1,-1,1,-1,1], dtype=int),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_symbols(seed: int, trials: int = 16) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols: list[dict[str, object]] = []
    for mode in MODE_NAMES:
        symbols.append({
            "symbol_index": len(symbols), "family": "preamble",
            "declared_mode": mode, "actual_mode": mode, "trial": -1-len(symbols),
            "hash_restored": 1, "theta_idx": 0,
            "bin_permutation": list(range(12)),
        })
    for trial in range(trials):
        for family in ("real", "pseudo", "wrong"):
            actual_index = int(rng.integers(0, 4))
            actual = MODE_NAMES[actual_index]
            theta_idx = int(rng.integers(0, 8))
            declared = actual
            permutation = list(range(12))
            if family == "wrong":
                declared = MODE_NAMES[(actual_index + 1 + int(rng.integers(0, 3))) % 4]
            elif family == "pseudo":
                declared = MODE_NAMES[int(rng.integers(0, 4))]
                permutation = rng.permutation(12).tolist()
            symbols.append({
                "symbol_index": len(symbols), "family": family,
                "declared_mode": declared, "actual_mode": actual, "trial": trial,
                "hash_restored": 1, "theta_idx": theta_idx,
                "bin_permutation": permutation,
            })
    return symbols


def write_run(root: Path, run_id: str, route: str, seed: int, condition: str, anomalous: bool = False) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    symbols = make_symbols(seed)
    phase_levels = 8
    gains = {
        "v4s5": np.linspace(0.25, 2.0, 12) * np.exp(1j * np.linspace(-0.9, 0.8, 12)),
        "v2s3": np.linspace(1.8, 0.35, 12) * np.exp(1j * np.linspace(0.6, -0.7, 12)),
    }[route]
    rng = np.random.default_rng(seed + (9000 if condition != "matrix" else 0))
    tones = np.geomspace(20.0, 1500.0, 12).tolist()
    summary_rows: list[list[object]] = []
    windows: list[dict[str, object]] = []
    t0 = 1_000_000_000
    for symbol in symbols:
        actual = str(symbol["actual_mode"])
        code = CODEBOOK[actual][np.asarray(symbol["bin_permutation"], dtype=int)]
        theta = 2 * np.pi * int(symbol["theta_idx"]) / phase_levels
        x = code * np.exp(1j * theta)
        if condition == "silent":
            z = 0.005 * (rng.normal(size=12) + 1j*rng.normal(size=12))
        elif condition == "scramble":
            z = 0.20 * (rng.normal(size=12) + 1j*rng.normal(size=12))
        else:
            active_gains = gains.copy()
            if anomalous and int(symbol["trial"]) >= 0 and int(symbol["trial"]) % 2 == 1:
                active_gains = np.roll(gains, 3) * np.exp(1j * np.linspace(0.0, 1.8, 12))
            z = x * active_gains + 0.01 * (rng.normal(size=12) + 1j*rng.normal(size=12))
        family_out = "real" if symbol["family"] == "preamble" else symbol["family"]
        row: list[object] = [family_out, symbol["declared_mode"], actual, symbol["trial"], 1, symbol["theta_idx"]]
        for value in z:
            row.extend([f"{value.real:.12g}", f"{value.imag:.12g}"])
        row.extend(["0.01"] * 12)
        summary_rows.append(row)
        drive_signs = np.sign(code).astype(int).tolist()
        phase_fractions = [((0.5 if sign < 0 else 0.0) + int(symbol["theta_idx"])/8.0) % 1.0 for sign in drive_signs]
        symbol["drive_signs"] = drive_signs
        symbol["phase_fractions"] = phase_fractions
        for b, value in enumerate(z):
            slot = t0 + (int(symbol["symbol_index"]) * 12 + b) * 1_600_000_000
            windows.append({
                "window_index": len(windows), "sample_offset_records": len(windows)*4,
                "sample_count": 4, "symbol_index": symbol["symbol_index"], "bin_index": b,
                "family": symbol["family"], "declared_mode": symbol["declared_mode"],
                "actual_mode": actual, "trial": symbol["trial"], "hash_restored": 1,
                "theta_idx": symbol["theta_idx"], "tone_hz": tones[b],
                "drive_sign": drive_signs[b] if condition != "silent" else 0,
                "phase_fraction": phase_fractions[b], "control": condition,
                "slot_start_tsc": slot, "capture_deadline_tsc": slot+1_200_000_000,
                "first_sample_tsc": slot+1000, "last_sample_tsc": slot+1_100_000_000,
                "temp_before_c": 41.0, "temp_after_c": 41.1,
                "cur_khz_before": 1600000, "cur_khz_after": 1600000,
                "cofvid_pstate_before": 1, "cofvid_pstate_after": 1,
                "computed_I": value.real, "computed_Q": value.imag,
                "computed_magnitude": abs(value), "computed_floor": 0.01,
            })

    dump(run_dir / "schedule.json", {
        "schema_id": "CAT_CAS_PDN_CARRIER_SCHEDULE_V1", "campaign_id": "synthetic",
        "run_id": run_id, "condition": condition, "seed": seed, "phase_levels": phase_levels,
        "t0_tsc": t0, "tones_hz": tones,
        "codebook": {name: CODEBOOK[name].tolist() for name in MODE_NAMES},
        "symbols": symbols,
    })
    dump(run_dir / "run.json", {
        "schema_id": "CAT_CAS_PDN_CARRIER_RUN_V1", "schema_version": "1.0.0",
        "campaign_id": "synthetic", "run_id": run_id, "condition": condition,
        "route": {"victim": 4 if route == "v4s5" else 2, "sender": 5 if route == "v4s5" else 3, "label": route},
        "seed": seed, "source_commit": "d"*40,
        "timing": {"t0_tsc": t0, "tsc_hz": 3_200_000_000.0, "slot_s": 0.5, "gap_s": 0.12, "read_hz": 4000},
    })
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["family","declared_mode","actual_mode","trial","hash_restored","theta_idx"] + [item for b in range(12) for item in (f"b{b:02d}_I", f"b{b:02d}_Q")] + [f"fl{b:02d}" for b in range(12)])
        writer.writerows(summary_rows)
    with (run_dir / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(windows)
    (run_dir / "raw_samples.bin").write_bytes(b"\0" * (16 * 4 * len(windows)))
    dump(run_dir / "analysis.json", {"verdict": "SYNTHETIC"})
    (run_dir / "stdout.log").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    files = {}
    for name in ("run.json","schedule.json","windows.csv","raw_samples.bin","summary.csv","analysis.json","stdout.log","stderr.log"):
        path = run_dir / name
        files[name] = {"size": path.stat().st_size, "sha256": sha(path)}
    dump(run_dir / "run_manifest.json", {"schema_id":"CAT_CAS_PDN_CARRIER_RUN_MANIFEST_V1","run_id":run_id,"source_commit":"d"*40,"files":files})


def build_campaign(root: Path) -> Path:
    campaign_root = root / "synthetic"
    campaign_root.mkdir()
    runs = [
        ("v4s5_matrix_seed0", "v4s5", 0, "matrix", False),
        ("v2s3_matrix_seed0", "v2s3", 0, "matrix", False),
        ("v4s5_matrix_seed4", "v4s5", 4, "matrix", True),
        ("v2s3_matrix_seed4", "v2s3", 4, "matrix", False),
        ("v4s5_silent_seed900", "v4s5", 900, "silent", False),
        ("v4s5_scramble_seed901", "v4s5", 901, "scramble", False),
    ]
    for args in runs:
        write_run(campaign_root, *args)
    dump(campaign_root / "campaign.json", {
        "schema_id":"CAT_CAS_PDN_CARRIER_CAMPAIGN_V1", "schema_version":"1.0.0",
        "contract_id":"synthetic", "campaign_id":"synthetic", "status":"COMPLETE",
        "source_commit":"d"*40, "primary_route":"v4s5", "comparator_routes":["v2s3"],
        "runs":[{"run_id": item[0]} for item in runs],
    })
    run_manifests: dict[str, dict[str, object]] = {}
    for run_id, *_ in runs:
        manifest_path = campaign_root / "runs" / run_id / "run_manifest.json"
        run_manifests[run_id] = {
            "path": f"runs/{run_id}/run_manifest.json",
            "size": manifest_path.stat().st_size,
            "sha256": sha(manifest_path),
        }
    dump(campaign_root / "campaign_manifest.json", {
        "schema_id": "CAT_CAS_PDN_CARRIER_CAMPAIGN_MANIFEST_V1",
        "campaign_id": "synthetic",
        "source_commit": "d" * 40,
        "files": {
            "campaign.json": {
                "size": (campaign_root / "campaign.json").stat().st_size,
                "sha256": sha(campaign_root / "campaign.json"),
            },
        },
        "run_manifests": run_manifests,
    })
    return campaign_root


class TransferGeometryTests(unittest.TestCase):
    def test_gate_layer_binding_is_source_relative_and_hashed(self) -> None:
        binding = analysis.gate_layer_reconciliation(Path("/external/campaign"))
        self.assertEqual(
            binding["source"],
            "replication_discrepancy/results/official_gate_decomposition.json",
        )
        self.assertFalse(binding["available_in_campaign_bundle"])
        self.assertTrue(binding["available_in_analysis_source"])
        self.assertEqual(len(binding["source_sha256"]), 64)
        self.assertGreater(binding["source_size"], 0)

    def test_full_synthetic_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            output = root / "output"
            manifest = analysis.build_outputs(campaign, output, True, 65005)
            self.assertTrue((output / "analysis_manifest.json").is_file())
            self.assertEqual(set(manifest["outputs"]), set(analysis.OUTPUT_NAMES))
            verification = analysis.verify_outputs(output)
            self.assertTrue(verification["valid"], verification["errors"])

            calibration = analysis.json_load(output / "chart_calibration.json")
            selected = calibration["runs"]["v4s5_matrix_seed0"]["selected_chart"]["chart_id"]
            self.assertEqual(selected, "C1_diagonal")

            execution = analysis.json_load(output / "execution_relation.json")
            wrong = execution["runs"]["v4s5_matrix_seed0"]["wrong"]
            self.assertGreater(wrong["actual_better_than_declared_fraction"], 0.8)

            pseudo = analysis.json_load(output / "pseudo_permutation_covariance.json")
            covariance = pseudo["runs"]["v4s5_matrix_seed0"]
            self.assertGreater(covariance["exact_better_than_alternative_fraction"], 0.75)

            phase = analysis.json_load(output / "phase_equivariance.json")
            self.assertLess(phase["runs"]["v4s5_matrix_seed0"]["mean_absolute_circular_error"], 0.15)

            seed4 = analysis.json_load(output / "seed4_transfer_report.json")
            self.assertIn(seed4["classification"], {"TRANSFER_REGIME_SHIFT", "MIXED_FAILURE", "UNRESOLVED"})

    def test_manifest_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            output = root / "output"
            analysis.build_outputs(campaign, output, False, 65005)
            path = output / "phase_equivariance.json"
            path.write_text(path.read_text() + " ", encoding="utf-8")
            verification = analysis.verify_outputs(output)
            self.assertFalse(verification["valid"])
            self.assertTrue(any("phase_equivariance.json" in error for error in verification["errors"]))

    def test_campaign_manifest_passes_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            manifest = analysis.verify_campaign_manifest(campaign)
            self.assertEqual(
                manifest["schema_id"],
                "CAT_CAS_PDN_CARRIER_CAMPAIGN_MANIFEST_V1",
            )

    def test_tampered_run_manifest_fails_campaign_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            run_manifest = campaign / "runs" / "v4s5_matrix_seed0" / "run_manifest.json"
            run_manifest.write_text(run_manifest.read_text() + " ", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                analysis.verify_campaign_manifest(campaign)
            self.assertIn("sha256 mismatch", str(ctx.exception))

    def test_tampered_campaign_json_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            campaign_json = campaign / "campaign.json"
            data = json.loads(campaign_json.read_text())
            data["campaign_id"] = "tampered"
            campaign_json.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                analysis.verify_campaign_manifest(campaign)
            self.assertIn("campaign ID mismatch", str(ctx.exception))

    def test_unbound_run_dir_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            extra = campaign / "runs" / "v9s9_matrix_seed77"
            extra.mkdir()
            with self.assertRaises(ValueError) as ctx:
                analysis.build_outputs(campaign, root / "output", False, 65005)
            self.assertIn("unbound run directories", str(ctx.exception))

    def test_manifest_id_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            manifest_path = campaign / "campaign_manifest.json"
            data = json.loads(manifest_path.read_text())
            data["campaign_id"] = "wrong_campaign_id"
            manifest_path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                analysis.verify_campaign_manifest(campaign)
            self.assertIn("campaign ID mismatch", str(ctx.exception))

    def test_malformed_manifest_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = build_campaign(root)
            manifest_path = campaign / "campaign_manifest.json"
            data = json.loads(manifest_path.read_text())
            data["run_manifests"]["evil"] = {
                "path": "../../etc/passwd",
                "size": 0,
                "sha256": "0" * 64,
            }
            manifest_path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                analysis.verify_campaign_manifest(campaign)
            self.assertIn("path traversal", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import gate_a_frequency_precondition_probe as probe


class FakeClock:
    def __init__(self, start: int, step: int):
        self.value = start
        self.step = step

    def __call__(self) -> int:
        current = self.value
        self.value += self.step
        return current


class GateAFrequencyPreconditionProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for core in probe.DEFAULT_CORES:
            policy = (
                self.root
                / "devices"
                / "system"
                / "cpu"
                / f"cpu{core}"
                / "cpufreq"
            )
            policy.mkdir(parents=True)
            values = {
                "scaling_driver": "acpi-cpufreq\n",
                "scaling_governor": "ondemand\n",
                "cpuinfo_min_freq": "800000\n",
                "cpuinfo_max_freq": "3200000\n",
                "scaling_min_freq": "800000\n",
                "scaling_max_freq": "3200000\n",
                "scaling_available_governors": "conservative ondemand userspace powersave performance\n",
                "scaling_available_frequencies": "3200000 2400000 1600000 800000\n",
                "affected_cpus": f"{core}\n",
                "related_cpus": f"{core}\n",
                "transition_latency": "10000\n",
                "scaling_cur_freq": "1600000\n",
            }
            for name, value in values.items():
                (policy / name).write_text(value, encoding="ascii")
        self.before = self._snapshot()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _snapshot(self) -> dict[str, bytes]:
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def _observe(self, *, sample_count: int = 4, read_bytes=None):
        return probe.observe_frequency_precondition(
            sysfs_root=self.root,
            sample_count=sample_count,
            interval_ms=1,
            read_bytes=read_bytes or (lambda path: path.read_bytes()),
            sleep=lambda _seconds: None,
            monotonic_ns=FakeClock(1_000, 10),
            utc_ns=FakeClock(2_000, 10),
        )

    def test_static_required_frequency_passes_without_mutation(self) -> None:
        receipt = self._observe()
        self.assertEqual(receipt["status"], "PASS_STATIC_PRECONDITION_OBSERVED")
        self.assertTrue(receipt["summary"]["all_pairs_exact"])
        self.assertEqual(receipt["summary"]["pair_exact_sample_count"], 4)
        self.assertEqual(receipt["control_writes"], 0)
        self.assertEqual(receipt["network_operations"], 0)
        self.assertEqual(self.before, self._snapshot())

    def test_static_800mhz_fails_observation(self) -> None:
        for core in probe.DEFAULT_CORES:
            path = (
                self.root
                / "devices"
                / "system"
                / "cpu"
                / f"cpu{core}"
                / "cpufreq"
                / "scaling_cur_freq"
            )
            path.write_text("800000\n", encoding="ascii")
        baseline = self._snapshot()
        receipt = self._observe()
        self.assertEqual(receipt["status"], "FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED")
        self.assertFalse(receipt["summary"]["any_pair_exact"])
        self.assertEqual(receipt["summary"]["per_core"]["4"]["unique_khz"], [800000])
        self.assertEqual(baseline, self._snapshot())

    def test_dynamic_frequency_is_inconclusive_not_pass(self) -> None:
        sequences = {
            4: iter([800000, 1600000, 1600000, 800000]),
            5: iter([800000, 1600000, 1600000, 800000]),
        }

        def read_bytes(path: Path) -> bytes:
            if path.name == "scaling_cur_freq":
                core = int(path.parent.parent.name.removeprefix("cpu"))
                return f"{next(sequences[core])}\n".encode("ascii")
            return path.read_bytes()

        receipt = self._observe(read_bytes=read_bytes)
        self.assertEqual(receipt["status"], "INCONCLUSIVE_DYNAMIC_PRECONDITION")
        self.assertEqual(receipt["summary"]["pair_exact_sample_count"], 2)
        self.assertEqual(receipt["summary"]["longest_consecutive_exact_pairs"], 2)
        self.assertFalse(receipt["summary"]["all_pairs_exact"])

    def test_core_pair_is_frozen(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "cores are frozen"):
            probe.observe_frequency_precondition(
                sysfs_root=self.root,
                cores=(0, 1),
                sample_count=2,
                interval_ms=1,
                sleep=lambda _seconds: None,
            )

    def test_missing_current_frequency_fails_closed(self) -> None:
        path = (
            self.root
            / "devices"
            / "system"
            / "cpu"
            / "cpu4"
            / "cpufreq"
            / "scaling_cur_freq"
        )
        path.unlink()
        with self.assertRaisesRegex(probe.ProbeError, "current frequency unreadable"):
            self._observe(sample_count=2)

    def test_inverted_policy_bounds_fail_closed(self) -> None:
        path = (
            self.root
            / "devices"
            / "system"
            / "cpu"
            / "cpu4"
            / "cpufreq"
            / "scaling_min_freq"
        )
        path.write_text("3300000\n", encoding="ascii")
        with self.assertRaisesRegex(probe.ProbeError, "escape cpuinfo bounds"):
            self._observe(sample_count=2)

    def test_required_frequency_support_is_bound(self) -> None:
        receipt = self._observe(sample_count=2)
        for metadata in receipt["policy_metadata"]:
            self.assertTrue(metadata["required_frequency_supported"])
            available = metadata["files"]["scaling_available_frequencies"]
            self.assertIn(1600000, available["parsed"])

    def test_receipt_key_set_is_closed(self) -> None:
        receipt = self._observe(sample_count=2)
        self.assertEqual(
            set(receipt),
            {
                "schema_id",
                "observation_mode",
                "sysfs_root",
                "required_frequency_khz",
                "cores",
                "sample_count_requested",
                "sample_interval_ms",
                "control_writes",
                "frequency_writes",
                "voltage_writes",
                "msr_reads",
                "msr_writes",
                "network_operations",
                "policy_metadata",
                "samples",
                "summary",
                "status",
                "failure",
            },
        )
        encoded = json.dumps(receipt, sort_keys=True)
        self.assertIn(probe.SCHEMA_ID, encoded)


if __name__ == "__main__":
    unittest.main()

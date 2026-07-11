#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gate_a_frequency_preparation as prep


class FakeSysfs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.write_calls: list[tuple[str, bytes]] = []
        self.fail_write_number: int | None = None
        self.fail_after_write_number: int | None = None
        self.freeze_current = False
        self.replace_policy_after_write_number: int | None = None
        self.replace_policy_after_frequency_read_number: int | None = None
        self.frequency_read_count = 0
        self._make()

    def _make(self) -> None:
        for core in prep.CORES:
            policy = self.root / prep.POLICY_RELATIVE_PATHS[core]
            policy.mkdir(parents=True)
            values = {
                "scaling_driver": "acpi-cpufreq\n",
                "scaling_governor": "schedutil\n",
                "cpuinfo_min_freq": "800000\n",
                "cpuinfo_max_freq": "3200000\n",
                "scaling_min_freq": "800000\n",
                "scaling_max_freq": "3200000\n",
                "scaling_available_frequencies": "3200000 2400000 1600000 800000\n",
                "affected_cpus": f"{core}\n",
                "related_cpus": f"{core}\n",
                "scaling_cur_freq": "800000\n",
            }
            for name, value in values.items():
                (policy / name).write_text(value, encoding="ascii")

    def read(self, path: Path) -> bytes:
        data = path.read_bytes()
        if path.name == "scaling_cur_freq":
            self.frequency_read_count += 1
            if (
                self.replace_policy_after_frequency_read_number
                == self.frequency_read_count
            ):
                self._replace_policy(path.parent)
        return data

    @staticmethod
    def _replace_policy(policy: Path) -> None:
        backup = policy.with_name(policy.name + "_replaced")
        policy.rename(backup)
        policy.mkdir()
        for child in backup.iterdir():
            (policy / child.name).write_bytes(child.read_bytes())

    def write(self, path: Path, data: bytes) -> None:
        self.write_calls.append((str(path), data))
        if self.fail_write_number == len(self.write_calls):
            raise OSError("injected write failure")
        path.write_bytes(data)
        if not self.freeze_current:
            policy = path.parent
            minimum = int((policy / "scaling_min_freq").read_text().strip())
            maximum = int((policy / "scaling_max_freq").read_text().strip())
            if minimum == maximum:
                (policy / "scaling_cur_freq").write_text(
                    f"{minimum}\n", encoding="ascii"
                )
            elif minimum == 800000 and maximum == 3200000:
                (policy / "scaling_cur_freq").write_text(
                    "800000\n", encoding="ascii"
                )
        if self.replace_policy_after_write_number == len(self.write_calls):
            self._replace_policy(path.parent)
        if self.fail_after_write_number == len(self.write_calls):
            raise OSError("injected post-write failure")

    def value(self, core: int, name: str) -> int:
        return int(
            (self.root / prep.POLICY_RELATIVE_PATHS[core] / name)
            .read_text()
            .strip()
        )


class FrequencyPreparationTests(unittest.TestCase):
    def run_transaction(self, fake: FakeSysfs):
        return prep.qualify_preparation_restoration(
            sysfs_root=fake.root,
            read_bytes=fake.read,
            write_bytes=fake.write,
            sample_count=4,
            interval_ms=1,
            sleep=lambda _seconds: None,
            monotonic_ns=self.clock(),
        )

    @staticmethod
    def clock():
        state = {"value": 0}

        def now() -> int:
            state["value"] += 1
            return state["value"]

        return now

    def assert_restored(self, fake: FakeSysfs) -> None:
        for core in prep.CORES:
            self.assertEqual(fake.value(core, "scaling_min_freq"), 800000)
            self.assertEqual(fake.value(core, "scaling_max_freq"), 3200000)

    def test_success_uses_exact_eight_writes_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "QUALIFIED_PREPARATION_AND_RESTORATION"
            )
            self.assertEqual(receipt["frequency_write_attempt_count"], 8)
            self.assertEqual(receipt["write_call_returned_count"], 8)
            self.assertTrue(receipt["pinned_observation"]["all_pairs_exact"])
            self.assertTrue(receipt["restoration_complete"])
            self.assert_restored(fake)
            expected = []
            for phase, fields in (
                ("prepare", ("scaling_max_freq", "scaling_min_freq")),
                ("restore", ("scaling_min_freq", "scaling_max_freq")),
            ):
                for core in prep.CORES:
                    for field in fields:
                        expected.append((phase, core, field))
            observed = [
                (e["phase"], e["core"], e["field"])
                for e in receipt["write_ledger"]
            ]
            self.assertEqual(observed, expected)

    def test_prepare_write_failure_restores_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.fail_write_number = 2
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PREPARATION__RESTORED"
            )
            self.assertEqual(receipt["retry_count"], 0)
            self.assertTrue(receipt["restoration_complete"])
            self.assert_restored(fake)

    def test_accepted_then_raised_write_restores_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.fail_after_write_number = 1
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PREPARATION__RESTORED"
            )
            self.assertEqual(receipt["retry_count"], 0)
            self.assertFalse(receipt["write_ledger"][0]["write_call_returned"])
            self.assertTrue(receipt["restoration_complete"])
            self.assert_restored(fake)

    def test_policy_identity_replacement_is_terminal_and_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.replace_policy_after_write_number = 1
            receipt = self.run_transaction(fake)
            self.assertEqual(receipt["status"], "FAILED_CLOSED_RESTORATION")
            self.assertIn("identity changed", receipt["preparation_failure"])
            self.assertIsNotNone(receipt["restoration_failure"])
            policy4_writes = [
                entry for entry in receipt["write_ledger"] if entry["core"] == 4
            ]
            self.assertEqual(len(policy4_writes), 1)
            self.assertTrue(policy4_writes[0]["write_call_returned"])

    def test_policy_replacement_during_observation_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.replace_policy_after_frequency_read_number = 1
            receipt = self.run_transaction(fake)
            self.assertEqual(receipt["status"], "FAILED_CLOSED_RESTORATION")
            self.assertIn(
                "identity changed after pinned sample",
                receipt["preparation_failure"],
            )
            self.assertIsNotNone(receipt["restoration_failure"])
            policy4_restores = [
                entry
                for entry in receipt["write_ledger"]
                if entry["core"] == 4 and entry["phase"] == "restore"
            ]
            self.assertEqual(policy4_restores, [])

    def test_static_observation_failure_restores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.freeze_current = True
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PREPARATION__RESTORED"
            )
            self.assertIn("not static", receipt["preparation_failure"])
            self.assert_restored(fake)

    def test_restoration_failure_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            fake.fail_write_number = 5
            receipt = self.run_transaction(fake)
            self.assertEqual(receipt["status"], "FAILED_CLOSED_RESTORATION")
            self.assertIsNotNone(receipt["restoration_failure"])
            self.assertEqual(receipt["retry_count"], 0)

    def test_wrong_governor_rejects_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            policy = fake.root / prep.POLICY_RELATIVE_PATHS[4]
            (policy / "scaling_governor").write_text(
                "performance\n", encoding="ascii"
            )
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PRECONDITION_NO_WRITES"
            )
            self.assertEqual(fake.write_calls, [])

    def test_missing_required_frequency_rejects_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            policy = fake.root / prep.POLICY_RELATIVE_PATHS[5]
            (policy / "scaling_available_frequencies").write_text(
                "3200000 2400000 800000\n", encoding="ascii"
            )
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PRECONDITION_NO_WRITES"
            )
            self.assertEqual(fake.write_calls, [])

    def test_policy_ownership_mismatch_rejects_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = FakeSysfs(Path(td))
            policy = fake.root / prep.POLICY_RELATIVE_PATHS[4]
            (policy / "affected_cpus").write_text("4 5\n", encoding="ascii")
            receipt = self.run_transaction(fake)
            self.assertEqual(
                receipt["status"], "FAILED_CLOSED_PRECONDITION_NO_WRITES"
            )
            self.assertEqual(fake.write_calls, [])

    def test_live_sysfs_refused_without_authority(self) -> None:
        with self.assertRaises(prep.PreparationError):
            prep.qualify_preparation_restoration(
                sysfs_root=Path("/sys"),
                read_bytes=lambda path: path.read_bytes(),
                write_bytes=lambda path, data: path.write_bytes(data),
                sample_count=2,
                interval_ms=1,
            )


if __name__ == "__main__":
    unittest.main()

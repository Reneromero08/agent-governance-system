from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "combined_observability_campaign" / "analysis"
import sys
sys.path.insert(0, str(REFERENCE))
from waveform_reference import intended_v2_gate  # noqa: E402


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(HERE / "waveform_fixture.c"), "-lm", "-o", str(HERE / "waveform_fixture")],
            check=True,
        )

    def test_c_matches_python_for_all_waveform_parameters(self) -> None:
        origin, tsc_hz, count = 1000, 8_000_000.0, 4096
        timestamps = np.arange(origin, origin + count, dtype=np.uint64)
        for tone in range(12):
            for phase in range(8):
                for level in (1, 2, 3):
                    output = subprocess.check_output([
                        str(HERE / "waveform_fixture"), str(origin), str(tsc_hz),
                        str(tone), str(phase), str(level), str(count),
                    ], text=True)
                    actual = np.fromstring(output, dtype=np.int8, sep="\n")
                    expected = intended_v2_gate(
                        timestamps, origin_tsc=origin, tsc_hz=tsc_hz,
                        tone_index=tone, phase_index_value=phase,
                        amplitude_level=level,
                    ).astype(np.int8)
                    np.testing.assert_array_equal(actual, expected)

    def test_phase_and_duty_contract(self) -> None:
        origin, tsc_hz, count = 0, 8_000_000.0, 800_000
        timestamps = np.arange(count, dtype=np.uint64)
        for level, duty in ((1, .125), (2, .25), (3, .375)):
            gate = intended_v2_gate(
                timestamps, origin_tsc=origin, tsc_hz=tsc_hz,
                tone_index=11, phase_index_value=0, amplitude_level=level,
            )
            self.assertAlmostEqual(float(gate.mean()), duty, delta=.002)


if __name__ == "__main__":
    unittest.main(verbosity=2)

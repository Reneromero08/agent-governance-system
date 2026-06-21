#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import numpy as np

from waveform_reference import (
    CODEBOOK,
    acquisition_gate,
    intended_v2_gate,
    lockin,
    phase_index,
    tone_hz,
)


class WaveformReferenceTests(unittest.TestCase):
    def test_codebook_shape_and_distance(self) -> None:
        self.assertEqual(CODEBOOK.shape, (4, 12))
        self.assertTrue(np.isin(CODEBOOK, (-1, 1)).all())
        distances = [
            int(np.count_nonzero(CODEBOOK[left] != CODEBOOK[right]))
            for left in range(4)
            for right in range(left + 1, 4)
        ]
        self.assertEqual(min(distances), 7)

    def test_acquisition_gate_repeats_every_four_requested_periods(self) -> None:
        tsc_hz = 1_000_000.0
        frequency = tone_hz(5)
        period_ticks = tsc_hz / frequency
        timestamps = np.arange(0, int(period_ticks * 8), dtype=np.uint64)
        gate = acquisition_gate(
            timestamps,
            origin_tsc=0,
            tsc_hz=tsc_hz,
            tone_index=5,
            phase_index_value=0,
            amplitude_level=3,
        )
        shift = int(round(period_ticks * 4))
        self.assertGreater(shift, 0)
        self.assertGreater(np.mean(gate[:-shift] == gate[shift:]), 0.99)

    def test_v2_gate_repeats_every_requested_period(self) -> None:
        tsc_hz = 8_000_000.0
        frequency = tone_hz(5)
        period_ticks = tsc_hz / frequency
        timestamps = np.arange(0, int(period_ticks * 4), dtype=np.uint64)
        gate = intended_v2_gate(
            timestamps,
            origin_tsc=0,
            tsc_hz=tsc_hz,
            tone_index=5,
            phase_index_value=0,
            amplitude_level=3,
        )
        shift = int(round(period_ticks))
        self.assertGreater(np.mean(gate[:-shift] == gate[shift:]), 0.99)

    def test_acquisition_requested_frequency_is_ideal_null(self) -> None:
        tsc_hz = 16_000_000.0
        tone_index = 4
        frequency = tone_hz(tone_index)
        count = 16000
        timestamps = np.arange(count, dtype=np.uint64) * 1000
        for level in (1, 2, 3):
            gate = acquisition_gate(
                timestamps,
                origin_tsc=0,
                tsc_hz=tsc_hz,
                tone_index=tone_index,
                phase_index_value=0,
                amplitude_level=level,
            )
            at_f = lockin(
                timestamps,
                gate,
                origin_tsc=0,
                tsc_hz=tsc_hz,
                frequency_hz=frequency,
            )
            at_f_over_4 = lockin(
                timestamps,
                gate,
                origin_tsc=0,
                tsc_hz=tsc_hz,
                frequency_hz=frequency / 4.0,
            )
            self.assertLess(abs(at_f), abs(at_f_over_4) * 0.05)

    def test_sign_offset_is_four_phase_indices(self) -> None:
        for mode in range(4):
            for source in range(12):
                positive = phase_index(mode, source, 0)
                expected = 4 if CODEBOOK[mode, source] < 0 else 0
                self.assertEqual(positive, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

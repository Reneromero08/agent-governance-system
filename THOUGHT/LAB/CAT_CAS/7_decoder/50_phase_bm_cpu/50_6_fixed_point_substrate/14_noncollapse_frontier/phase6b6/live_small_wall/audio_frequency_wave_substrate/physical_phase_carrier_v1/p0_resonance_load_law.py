#!/usr/bin/env python3
"""Prospective, non-executing resonance/load law for the bounded P0 carrier."""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEARCH_MIN_HZ = 32_760.0
SEARCH_MAX_HZ = 32_820.0
ACCEPT_MIN_HZ = 32_768.0
ACCEPT_MAX_HZ = 32_820.0
COARSE_STEP_HZ = 1.0
FINE_HALF_SPAN_HZ = 1.0125
FINE_STEP_HZ = 0.025
CALIBRATION_DRIVE_VPP = 0.100
MAX_U95_HZ = 0.050
MIN_Q = 4_000.0
MAX_Q = 60_000.0
MIN_PREPARATION_SECONDS = 3.0


@dataclass(frozen=True)
class BvdCorner:
    rm_ohm: float
    lm_h: float
    cm_f: float
    c0_f: float
    sense_load_f: float


def series_resonance_hz(corner: BvdCorner) -> float:
    return 1.0 / (2.0 * math.pi * math.sqrt(corner.lm_h * corner.cm_f))


def loaded_state(corner: BvdCorner, frequency_hz: float, source_ohm: float = 50.0, limiter_ohm: float = 100_000.0, shunt_ohm: float = 100_000.0) -> tuple[complex, complex, complex]:
    omega = 2.0 * math.pi * frequency_hz
    motional = complex(corner.rm_ohm, omega * corner.lm_h - 1.0 / (omega * corner.cm_f))
    static_and_sense = complex(0.0, -1.0 / (omega * (corner.c0_f + corner.sense_load_f)))
    carrier = 1.0 / (1.0 / motional + 1.0 / static_and_sense)
    series_branch = limiter_ohm + carrier
    source_load = 1.0 / (1.0 / shunt_ohm + 1.0 / series_branch)
    source_node = source_load / (source_ohm + source_load)
    carrier_terminal = source_node * carrier / series_branch
    motional_current_per_source_v = carrier_terminal / motional
    return source_node, carrier_terminal, motional_current_per_source_v


def prospective_sanity_document() -> dict[str, Any]:
    def cm_for(frequency_hz: float, lm_h: float) -> float:
        return 1.0 / ((2.0 * math.pi * frequency_hz) ** 2 * lm_h)

    corners = [
        BvdCorner(70_000.0, 8_000.0, cm_for(32_774.0, 8_000.0), 0.8e-12, 3.2e-12),
        BvdCorner(50_000.0, 8_000.0, cm_for(32_786.0, 8_000.0), 1.2e-12, 4.0e-12),
        BvdCorner(70_000.0, 12_000.0, cm_for(32_800.0, 12_000.0), 0.8e-12, 4.0e-12),
        BvdCorner(50_000.0, 12_000.0, cm_for(32_810.0, 12_000.0), 1.2e-12, 3.2e-12),
    ]
    rows = []
    for corner in corners:
        frequency = series_resonance_hz(corner)
        q_factor = 2.0 * math.pi * frequency * corner.lm_h / corner.rm_ohm
        decay_seconds = q_factor / (math.pi * frequency)
        source_node, terminal_transfer, motional_current = loaded_state(corner, frequency)
        source_v_rms = CALIBRATION_DRIVE_VPP / (2.0 * math.sqrt(2.0))
        motional_current_rms = source_v_rms * abs(motional_current)
        rows.append(
            {
                "c0_f": corner.c0_f,
                "cm_f": corner.cm_f,
                "decay_seconds": decay_seconds,
                "lm_h": corner.lm_h,
                "loaded_terminal_vpp_at_0p100_vpp": CALIBRATION_DRIVE_VPP * abs(terminal_transfer),
                "motional_current_ua_rms_at_0p100_vpp": 1e6 * motional_current_rms,
                "motional_power_uw_at_0p100_vpp": 1e6 * motional_current_rms * motional_current_rms * corner.rm_ohm,
                "q_factor": q_factor,
                "rm_ohm": corner.rm_ohm,
                "ring_up_fraction_after_3s": 1.0 - math.exp(-MIN_PREPARATION_SECONDS / decay_seconds),
                "sense_load_f": corner.sense_load_f,
                "series_resonance_hz": frequency,
                "source_node_vpp_at_0p100_vpp": CALIBRATION_DRIVE_VPP * abs(source_node),
            }
        )
    frequencies = [row["series_resonance_hz"] for row in rows]
    q_values = [row["q_factor"] for row in rows]
    decays = [row["decay_seconds"] for row in rows]
    return {
        "schema": "p0.resonance-load-sanity.v1",
        "authority": "PROSPECTIVE_MODEL_ONLY__NO_PHYSICAL_MEASUREMENT",
        "calibration_law": {
            "accept_frequency_hz": [ACCEPT_MIN_HZ, ACCEPT_MAX_HZ],
            "calibration_drive_vpp": CALIBRATION_DRIVE_VPP,
            "coarse_search_hz": [SEARCH_MIN_HZ, SEARCH_MAX_HZ, COARSE_STEP_HZ],
            "fine_search": {"half_span_hz": FINE_HALF_SPAN_HZ, "step_hz": FINE_STEP_HZ},
            "fit": "H(f)=B+C/(1+i*2Q*(f-f0)/f0)",
            "fit_condition_number_maximum": 100_000_000.0,
            "maximum_f_carrier_u95_hz": MAX_U95_HZ,
            "maximum_off_resonance_ratio_plus_u95": 0.030,
            "maximum_q_u95_fraction": 0.10,
            "maximum_reduced_chi_square": 5.0,
            "minimum_resonance_snr": 25.0,
            "minimum_resonance_to_background": 0.20,
            "minimum_source_snr": 50.0,
            "q_factor": [MIN_Q, MAX_Q],
            "retry_law": "ONE_CALIBRATION_PASS__REJECT_ON_FAILURE",
            "selection": "deterministic_bounded_variable_projection__reject_boundary_or_nonconvergence",
        },
        "canonical_calibration_payload": {
            "channel_order": ["CH0_SOURCE_MONITOR", "CH1_CARRIER_RESPONSE"],
            "dtype": "little-endian-signed-int16",
            "frequency_blocks": 143,
            "layout": "frequency-major__sample-major__ch0-then-ch1-interleaved",
            "payload_bytes": 2_342_912,
            "sample_rate_hz": 1_000_000,
            "samples_per_channel_per_frequency": 4_096,
            "settling_seconds_before_block": 0.05,
            "source_amplitude_vpp": 0.1,
            "source_offset_v": 0.0,
        },
        "frequency_binding": {
            "f_witness_relation": "f_witness_hz == 2 * f_carrier_hz",
            "propagation": [
                "source_queryback",
                "drive_fit",
                "C2_transfer",
                "I_Q_projection",
                "reconstruction",
                "cycle_counting",
                "off_resonance_controls",
                "matched_comparison",
            ],
        },
        "selected_load_topology": {
            "drive_shunt_ohm": 100_000.0,
            "sense_load_f": [3.2e-12, 4.0e-12],
            "series_limiter_ohm": 100_000.0,
            "source_output_ohm": 50.0,
            "topology": "50R_SOURCE__100K_SHUNT__100K_SERIES_LIMITER__FC135_BVD_PARALLEL_C0_AND_SENSE_LOAD",
        },
        "prospective_bvd_binary_corners": rows,
        "predicted_envelope": {
            "decay_seconds": [min(decays), max(decays)],
            "q_factor": [min(q_values), max(q_values)],
            "series_resonance_hz": [min(frequencies), max(frequencies)],
        },
        "preparation_law": {
            "minimum_seconds": MIN_PREPARATION_SECONDS,
            "proof": "source_preparation_started_utc <= acquisition_started_utc - 3 seconds",
        },
        "scope": {
            "continuous_uncertainty_envelope_claimed": False,
            "physical_claim_authorized": False,
            "sweep_name": "complete binary-corner sweep",
        },
    }


def main() -> int:
    target = Path(__file__).with_name("P0_RESONANCE_LOAD_SANITY_MODEL.json")
    target.write_text(json.dumps(prospective_sanity_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

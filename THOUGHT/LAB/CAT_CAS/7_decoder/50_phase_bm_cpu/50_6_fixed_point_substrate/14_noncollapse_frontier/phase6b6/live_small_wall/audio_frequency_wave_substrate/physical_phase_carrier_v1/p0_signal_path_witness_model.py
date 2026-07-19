#!/usr/bin/env python3
"""Deterministic, non-executing P0 C2 signal-path witness model.

This is a prospective circuit-envelope calculation only.  It performs no
network, purchasing, audio, instrument, or hardware action.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import copy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json"
F1 = 32_768.0
F2 = 65_536.0
FS = 1_000_000
KB = 1.380649e-23
TEMPERATURE_K = 303.15
AUTHORITY = "AUTHORIZE P0 BUILD-READINESS ONLY"
CEILING = "NON_EXECUTING_P0_SIGNAL_PATH_WITNESS_REPAIR_ONLY"

# Endpoints are swept as a complete binary corner set.  These are prospective
# engineering bounds, not measured part claims.  The digitizer mode is frozen
# to 1 Mohm || 30 pF and the ADG OFF state is modeled with explicit SB, D and
# SA nodes; no metadata scalar substitutes for either loading path.
RANGES: dict[str, tuple[float, float]] = {
    "sdg_output_resistance_ohm": (47.5, 52.5),
    "monitor_resistor_ohm": (99_900.0, 100_100.0),
    "monitor_bias_resistor_ohm": (999_000.0, 1_001_000.0),
    "injection_resistor_fraction": (0.999, 1.001),
    "drive_shunt_resistor_fraction": (0.999, 1.001),
    "adg_sb_d_off_capacitance_f": (7.0e-12, 15.0e-12),
    "adg_sb_ground_off_capacitance_f": (2.0e-12, 7.0e-12),
    "relay_closed_resistance_ohm": (0.05, 0.20),
    "relay_open_capacitance_f": (0.30e-12, 1.00e-12),
    "detector_input_and_layout_capacitance_f": (3.20e-12, 4.00e-12),
    "digitizer_input_resistance_ohm": (950_000.0, 1_050_000.0),
    "digitizer_input_capacitance_f": (28.0e-12, 32.0e-12),
    "fc135_static_capacitance_f": (0.80e-12, 1.20e-12),
    "fc135_motional_capacitance_f": (0.003e-12, 0.015e-12),
    "fc135_motional_resistance_ohm": (30_000.0, 70_000.0),
    "opa810_closed_loop_gain": (0.995, 1.000),
    "combined_input_noise_v_per_sqrt_hz": (140e-9, 200e-9),
}

FIXED_PARAMETERS = {
    "adg_d_sa_on_resistance_ohm": 5.0,
    "adg_sa_termination_resistance_ohm": 50.05,
    "drive_shunt_part": "Vishay TNPW0805100KBEEN",
    "drive_shunt_resistance_ohm": 100_000.0,
    "digitizer_mode": "1_MOHM_PARALLEL_30_PF_TRUE_DIFFERENTIAL",
    "digitizer_negative_leg_reference": "CALIBRATED_AGND",
    "fc135_loaded_frequency_hz": F1,
    "fc135_loaded_frequency_capacitance_f": 12.5e-12,
    "opa810_output_resistance_ohm": 50.0,
}

RESISTOR_CANDIDATES_OHM = (1_000_000.0, 1_500_000.0, 2_200_000.0, 3_300_000.0, 4_700_000.0)
C2_AMPLITUDE_CANDIDATES_VPP = (0.100, 0.150, 0.200, 0.225, 0.250, 0.300, 0.350)
FIRST_CANDIDATE_RESISTOR_OHM = 1_000_000.0
FIRST_CANDIDATE_AMPLITUDE_VPP = 0.100
EXPECTED_SELECTED_RESISTOR_OHM = FIRST_CANDIDATE_RESISTOR_OHM
EXPECTED_SELECTED_AMPLITUDE_VPP = FIRST_CANDIDATE_AMPLITUDE_VPP
PRE_WINDOW_SAMPLES = 192
OPEN_WINDOW_SAMPLES = 960
ADC_LSB_MAX_V = 15.3e-6


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_branch(matrix: np.ndarray, a: int, b: int | None, admittance: complex) -> None:
    matrix[a, a] += admittance
    if b is not None:
        matrix[b, b] += admittance
        matrix[a, b] -= admittance
        matrix[b, a] -= admittance


def solve_nodes(matrix: np.ndarray, drive: np.ndarray) -> np.ndarray:
    condition = float(np.linalg.cond(matrix, 2))
    if not math.isfinite(condition) or condition > 1e12:
        raise RuntimeError(f"ill-conditioned circuit matrix: {condition}")
    return np.linalg.solve(matrix, drive)


def capacitor_y(frequency_hz: float, capacitance_f: float) -> complex:
    return 1j * 2.0 * math.pi * frequency_hz * capacitance_f


def bvd_elements(frequency_hz: float, values: dict[str, float]) -> tuple[complex, complex]:
    omega = 2.0 * math.pi * frequency_hz
    cm = values["fc135_motional_capacitance_f"]
    c0 = values["fc135_static_capacitance_f"]
    cl = FIXED_PARAMETERS["fc135_loaded_frequency_capacitance_f"]
    # The part identity is specified at the 12.5 pF loaded frequency.  Derive
    # series resonance from that identity instead of forcing F1 to be fs.
    series_frequency = F1 / math.sqrt(1.0 + cm / (c0 + cl))
    lm = 1.0 / ((2.0 * math.pi * series_frequency) ** 2 * cm)
    motional_z = values["fc135_motional_resistance_ohm"] + 1j * omega * lm + 1.0 / (1j * omega * cm)
    return capacitor_y(frequency_hz, c0) + 1.0 / motional_z, motional_z


def bvd_y(frequency_hz: float, values: dict[str, float]) -> complex:
    return bvd_elements(frequency_hz, values)[0]


def digitizer_y(frequency_hz: float, values: dict[str, float]) -> complex:
    return 1.0 / values["digitizer_input_resistance_ohm"] + capacitor_y(
        frequency_hz, values["digitizer_input_capacitance_f"]
    )


def contact_y(frequency_hz: float, state: str, values: dict[str, float]) -> complex:
    if state == "closed":
        return 1.0 / values["relay_closed_resistance_ohm"]
    if state == "open":
        return capacitor_y(frequency_hz, values["relay_open_capacitance_f"])
    raise ValueError(state)


def f2_transfer(
    values: dict[str, float],
    injection_ohm: float,
    k1: str,
    k2: str,
    guard: str = "open",
    wrong_node: bool = False,
    population: str = "dut",
) -> dict[str, float]:
    """Return phasor transfers for C2 source, CH0 monitor, gate, midpoint and CH1."""
    # Nodes: C2 port, monitor sum, SB/N_GATE_OUT, N_MIDPOINT,
    # N_ELECTRODE_A, D/N_SRC, SA/N_GATE_TERM.
    matrix = np.zeros((7, 7), dtype=np.complex128)
    drive = np.zeros(7, dtype=np.complex128)
    rsrc = values["sdg_output_resistance_ohm"]
    add_branch(matrix, 0, None, 1.0 / rsrc)
    drive[0] += 1.0 / rsrc  # unit-peak Thevenin C2 source

    rmon = values["monitor_resistor_ohm"]
    add_branch(matrix, 0, 1, 1.0 / rmon)
    add_branch(matrix, 1, None, 1.0 / values["monitor_bias_resistor_ohm"])
    add_branch(matrix, 1, None, digitizer_y(F2, values))
    # C1 monitor leg terminates in the inactive 50-ohm C1 source at f2.
    add_branch(matrix, 1, None, 1.0 / (rmon + rsrc))

    if wrong_node:
        # Deliberately wrong topology: C2 enters at D/N_SRC upstream of the
        # open SB pole while metadata may still claim N_GATE_OUT.
        add_branch(matrix, 0, 5, 1.0 / injection_ohm)
    else:
        add_branch(matrix, 0, 2, 1.0 / injection_ohm)

    # OFF means D-to-SA-to-50 ohm.  SB is isolated but retains both SB-D and
    # SB-ground parasitics.  Keeping D and SA as explicit nodes prevents a
    # fictitious 100 kohm return from entering the envelope.
    add_branch(matrix, 2, 5, capacitor_y(F2, values["adg_sb_d_off_capacitance_f"]))
    add_branch(matrix, 2, None, capacitor_y(F2, values["adg_sb_ground_off_capacitance_f"]))
    add_branch(matrix, 5, 6, 1.0 / FIXED_PARAMETERS["adg_d_sa_on_resistance_ohm"])
    add_branch(matrix, 6, None, 1.0 / FIXED_PARAMETERS["adg_sa_termination_resistance_ohm"])
    add_branch(
        matrix,
        5,
        None,
        1.0
        / (
            FIXED_PARAMETERS["drive_shunt_resistance_ohm"]
            * values["drive_shunt_resistor_fraction"]
        ),
    )
    add_branch(matrix, 2, 3, contact_y(F2, k1, values))
    add_branch(matrix, 3, 4, contact_y(F2, k2, values))
    if guard == "open":
        guard_z = 50.0 + 1.0 / capacitor_y(F2, values["relay_open_capacitance_f"])
        add_branch(matrix, 3, None, 1.0 / guard_z)
    elif guard == "closed":
        add_branch(matrix, 3, None, 1.0 / (50.0 + values["relay_closed_resistance_ohm"]))
    else:
        raise ValueError(guard)
    if population == "dut":
        electrode_y = bvd_y(F2, values)
    elif population == "detector_only":
        electrode_y = 0.0j
    elif population == "dummy_1pf":
        electrode_y = capacitor_y(F2, 1.0e-12)
    else:
        raise ValueError(population)
    electrode_y += capacitor_y(F2, values["detector_input_and_layout_capacitance_f"])
    electrode_y += 1.0 / 100_000_000.0
    add_branch(matrix, 4, None, electrode_y)
    nodes = solve_nodes(matrix, drive)
    ch0 = nodes[1]
    digitizer_load = 1.0 / digitizer_y(F2, values)
    output_divider = digitizer_load / (
        FIXED_PARAMETERS["opa810_output_resistance_ohm"] + digitizer_load
    )
    ch1 = values["opa810_closed_loop_gain"] * output_divider * nodes[4]
    h2 = ch1 / ch0
    bvd_admittance = bvd_y(F2, values) if population == "dut" else 0.0j
    return {
        "abs_ch0_per_source": abs(ch0),
        "abs_ch1_per_source": abs(ch1),
        "abs_h2": abs(h2),
        "h2_imag": h2.imag,
        "h2_real": h2.real,
        "phase_h2_rad": math.atan2(h2.imag, h2.real),
        "carrier_terminal_per_source": abs(nodes[4]),
        "carrier_current_per_source_a": abs(nodes[4] * bvd_admittance),
        "carrier_real_power_per_source_w": 0.5 * abs(nodes[4]) ** 2 * max(0.0, bvd_admittance.real),
    }


def f1_loading(values: dict[str, float], injection_ohm: float) -> dict[str, float]:
    # The permanent 100 kohm N_SRC shunt is upstream of the ADG contact.  It
    # enforces the inherited 0.200 Vpp carrier-terminal cap while remaining on
    # the D/SA side after ADG OFF, so it cannot erase the C2 witness injected at
    # SB/N_GATE_OUT.  The baseline includes that fixed shunt; perturbation is
    # the incremental effect of the C2 injection branch only.
    load_y = bvd_y(F1, values) + capacitor_y(F1, values["detector_input_and_layout_capacitance_f"]) + 1.0 / 100_000_000.0
    load_z = 1.0 / load_y
    source_z = values["sdg_output_resistance_ohm"] + 100_000.0
    path_z = FIXED_PARAMETERS["adg_d_sa_on_resistance_ohm"] + values["relay_closed_resistance_ohm"] * 2.0 + 2.0
    drive_shunt_z = FIXED_PARAMETERS["drive_shunt_resistance_ohm"] * values["drive_shunt_resistor_fraction"]
    injection_z = injection_ohm + values["sdg_output_resistance_ohm"]
    gate_load_z = 1.0 / (1.0 / load_z + 1.0 / injection_z)
    d_load_z = 1.0 / (1.0 / drive_shunt_z + 1.0 / (path_z + gate_load_z))
    d_voltage = d_load_z / (source_z + d_load_z)
    loaded_terminal = d_voltage * gate_load_z / (path_z + gate_load_z)
    baseline_d_load_z = 1.0 / (1.0 / drive_shunt_z + 1.0 / (path_z + load_z))
    baseline_d_voltage = baseline_d_load_z / (source_z + baseline_d_load_z)
    baseline = baseline_d_voltage * load_z / (path_z + load_z)
    perturbation = abs(abs(loaded_terminal) / abs(baseline) - 1.0)
    source_peak = 0.200
    terminal_peak = source_peak * abs(loaded_terminal)
    _, motional_z = bvd_elements(F1, values)
    motional_current_rms = terminal_peak / abs(motional_z) / math.sqrt(2.0)
    motional_power = motional_current_rms**2 * values["fc135_motional_resistance_ohm"]
    return {
        "fractional_terminal_perturbation": perturbation,
        "carrier_terminal_vpp": 2.0 * terminal_peak,
        "motional_current_ua_rms": 1e6 * motional_current_rms,
        "motional_power_uw": 1e6 * motional_power,
    }


def estimator_u95(values: dict[str, float], h: float, ch0_peak: float, samples: int) -> float:
    # Combined prospective front-end density includes the injection resistor,
    # OPA810 input noise, board reserve and the digitizer input.  Quantization
    # is added independently.  The LS complex-amplitude uncertainty uses the
    # exact N-sample scaling for orthogonalized sine/cosine columns.
    density = values["combined_input_noise_v_per_sqrt_hz"]
    sample_rms = math.sqrt(density * density * FS / 2.0 + ADC_LSB_MAX_V**2 / 12.0)
    amplitude_sigma = sample_rms * math.sqrt(2.0 / samples)
    # CH0 is lower impedance; conservatively assign the same density to it.
    return 1.96 * amplitude_sigma * math.sqrt(1.0 + h * h) / ch0_peak


def corner_values() -> list[dict[str, float]]:
    names = tuple(RANGES)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(RANGES[name] for name in names))]


def candidate_envelope(
    corners: list[dict[str, float]],
    injection_ohm: float,
    amplitude_vpp: float,
    full_controls: bool = False,
) -> dict[str, Any]:
    pre_h: list[float] = []
    pre_phase: list[float] = []
    open_h: list[float] = []
    open_phase: list[float] = []
    separation: list[float] = []
    r_drop: list[float] = []
    pre_snr: list[float] = []
    open_u95: list[float] = []
    perturb: list[float] = []
    f1_vpp: list[float] = []
    f1_current: list[float] = []
    f1_power: list[float] = []
    f2_vpp: list[float] = []
    f2_current: list[float] = []
    f2_power: list[float] = []
    wrong_node_h: list[float] = []
    wrong_node_phase: list[float] = []
    guard_mask_h: list[float] = []
    guard_mask_phase: list[float] = []
    detector_only_pre_h: list[float] = []
    detector_only_open_h: list[float] = []
    dummy_1pf_pre_h: list[float] = []
    dummy_1pf_open_h: list[float] = []
    source_peak = amplitude_vpp / 2.0
    for values in corners:
        effective_injection = injection_ohm * values["injection_resistor_fraction"]
        pre = f2_transfer(values, effective_injection, "closed", "closed")
        open_states = (
            f2_transfer(values, effective_injection, "open", "closed"),
            f2_transfer(values, effective_injection, "closed", "open"),
            f2_transfer(values, effective_injection, "open", "open"),
        )
        worst_open = max(open_states, key=lambda item: item["abs_h2"])
        pre_h.append(pre["abs_h2"])
        pre_phase.append(pre["phase_h2_rad"])
        open_h.extend(item["abs_h2"] for item in open_states)
        open_phase.extend(item["phase_h2_rad"] for item in open_states)
        separation.append(abs(complex(pre["h2_real"], pre["h2_imag"]) - complex(worst_open["h2_real"], worst_open["h2_imag"])))
        r_drop.append(worst_open["abs_ch1_per_source"] / pre["abs_ch1_per_source"])
        sample_noise = math.sqrt(values["combined_input_noise_v_per_sqrt_hz"] ** 2 * FS / 2.0 + ADC_LSB_MAX_V**2 / 12.0)
        pre_snr.append(source_peak * pre["abs_ch1_per_source"] / sample_noise)
        open_u95.append(estimator_u95(values, worst_open["abs_h2"], source_peak * worst_open["abs_ch0_per_source"], OPEN_WINDOW_SAMPLES))
        loading = f1_loading(values, effective_injection)
        perturb.append(loading["fractional_terminal_perturbation"])
        f1_vpp.append(loading["carrier_terminal_vpp"])
        f1_current.append(loading["motional_current_ua_rms"])
        f1_power.append(loading["motional_power_uw"])
        f2_vpp.append(amplitude_vpp * pre["carrier_terminal_per_source"])
        f2_current.append(1e6 * source_peak * pre["carrier_current_per_source_a"] / math.sqrt(2.0))
        f2_power.append(1e6 * source_peak**2 * pre["carrier_real_power_per_source_w"])
        if full_controls:
            wrong = f2_transfer(values, effective_injection, "closed", "closed", wrong_node=True)
            wrong_node_h.append(wrong["abs_h2"])
            wrong_node_phase.append(wrong["phase_h2_rad"])
            guarded = f2_transfer(values, effective_injection, "closed", "closed", guard="closed")
            guard_mask_h.append(guarded["abs_h2"])
            guard_mask_phase.append(guarded["phase_h2_rad"])
            for population, pre_values, open_values in (
                ("detector_only", detector_only_pre_h, detector_only_open_h),
                ("dummy_1pf", dummy_1pf_pre_h, dummy_1pf_open_h),
            ):
                population_pre = f2_transfer(values, effective_injection, "closed", "closed", population=population)
                population_open = (
                    f2_transfer(values, effective_injection, "open", "closed", population=population),
                    f2_transfer(values, effective_injection, "closed", "open", population=population),
                    f2_transfer(values, effective_injection, "open", "open", population=population),
                )
                pre_values.append(population_pre["abs_h2"])
                open_values.extend(item["abs_h2"] for item in population_open)
    envelope = {
        "f1_carrier_terminal_vpp": [min(f1_vpp), max(f1_vpp)],
        "f1_motional_current_ua_rms": [min(f1_current), max(f1_current)],
        "f1_motional_power_uw": [min(f1_power), max(f1_power)],
        "f2_carrier_terminal_vpp": [min(f2_vpp), max(f2_vpp)],
        "f2_motional_current_ua_rms": [min(f2_current), max(f2_current)],
        "f2_motional_real_power_uw": [min(f2_power), max(f2_power)],
        "isolated_abs_h2": [min(open_h), max(open_h)],
        "isolated_phase_h2_rad": [min(open_phase), max(open_phase)],
        "isolated_u95_h2": [min(open_u95), max(open_u95)],
        "pilot_f1_fractional_perturbation": [min(perturb), max(perturb)],
        "pre_abs_h2": [min(pre_h), max(pre_h)],
        "pre_phase_h2_rad": [min(pre_phase), max(pre_phase)],
        "pre_open_complex_h2_separation": [min(separation), max(separation)],
        "pre_pilot_snr": [min(pre_snr), max(pre_snr)],
        "r_drop": [min(r_drop), max(r_drop)],
    }
    if full_controls:
        envelope.update(
            {
                "guard_mask_abs_h2": [min(guard_mask_h), max(guard_mask_h)],
                "guard_mask_phase_h2_rad": [min(guard_mask_phase), max(guard_mask_phase)],
                "wrong_node_abs_h2": [min(wrong_node_h), max(wrong_node_h)],
                "wrong_node_phase_h2_rad": [min(wrong_node_phase), max(wrong_node_phase)],
                "detector_only_pre_abs_h2": [min(detector_only_pre_h), max(detector_only_pre_h)],
                "detector_only_isolated_abs_h2": [min(detector_only_open_h), max(detector_only_open_h)],
                "dummy_1pf_pre_abs_h2": [min(dummy_1pf_pre_h), max(dummy_1pf_pre_h)],
                "dummy_1pf_isolated_abs_h2": [min(dummy_1pf_open_h), max(dummy_1pf_open_h)],
            }
        )
    feasible = bool(
        envelope["pre_pilot_snr"][0] >= 20.0
        and envelope["pre_open_complex_h2_separation"][0] >= 0.025
        and envelope["pilot_f1_fractional_perturbation"][1] <= 0.050
        and envelope["f1_carrier_terminal_vpp"][1] <= 0.200
        and envelope["f1_motional_current_ua_rms"][1] <= 2.000
        and envelope["f1_motional_power_uw"][1] <= 0.100
        and envelope["f2_carrier_terminal_vpp"][1] <= 0.050
        and envelope["isolated_u95_h2"][1] <= 0.010
    )
    return {"amplitude_vpp": amplitude_vpp, "envelope": envelope, "feasible": feasible, "injection_resistor_ohm": injection_ohm}


def rescale_candidate(base: dict[str, Any], amplitude_vpp: float) -> dict[str, Any]:
    """Rescale the linear circuit result without re-solving identical matrices."""
    candidate = copy.deepcopy(base)
    ratio = amplitude_vpp / base["amplitude_vpp"]
    candidate["amplitude_vpp"] = amplitude_vpp
    envelope = candidate["envelope"]
    envelope["pre_pilot_snr"] = [value * ratio for value in envelope["pre_pilot_snr"]]
    for key in ("f2_carrier_terminal_vpp", "f2_motional_current_ua_rms"):
        envelope[key] = [value * ratio for value in envelope[key]]
    envelope["f2_motional_real_power_uw"] = [value * ratio * ratio for value in envelope["f2_motional_real_power_uw"]]
    candidate["feasible"] = bool(
        envelope["pre_pilot_snr"][0] >= 20.0
        and envelope["pre_open_complex_h2_separation"][0] >= 0.025
        and envelope["pilot_f1_fractional_perturbation"][1] <= 0.050
        and envelope["f1_carrier_terminal_vpp"][1] <= 0.200
        and envelope["f1_motional_current_ua_rms"][1] <= 2.000
        and envelope["f1_motional_power_uw"][1] <= 0.100
        and envelope["f2_carrier_terminal_vpp"][1] <= 0.050
        and envelope["isolated_u95_h2"][1] <= 0.010
    )
    return candidate


def selected_control_envelope(corners: list[dict[str, float]], injection_ohm: float) -> dict[str, list[float]]:
    """Full non-primary control envelope for the selected candidate only."""
    wrong_node_h: list[float] = []
    wrong_node_phase: list[float] = []
    guard_mask_h: list[float] = []
    guard_mask_phase: list[float] = []
    detector_only_pre_h: list[float] = []
    detector_only_open_h: list[float] = []
    dummy_1pf_pre_h: list[float] = []
    dummy_1pf_open_h: list[float] = []
    for values in corners:
        effective_injection = injection_ohm * values["injection_resistor_fraction"]
        wrong = f2_transfer(values, effective_injection, "closed", "closed", wrong_node=True)
        wrong_node_h.append(wrong["abs_h2"])
        wrong_node_phase.append(wrong["phase_h2_rad"])
        guarded = f2_transfer(values, effective_injection, "closed", "closed", guard="closed")
        guard_mask_h.append(guarded["abs_h2"])
        guard_mask_phase.append(guarded["phase_h2_rad"])
        for population, pre_values, open_values in (
            ("detector_only", detector_only_pre_h, detector_only_open_h),
            ("dummy_1pf", dummy_1pf_pre_h, dummy_1pf_open_h),
        ):
            population_pre = f2_transfer(values, effective_injection, "closed", "closed", population=population)
            population_open = (
                f2_transfer(values, effective_injection, "open", "closed", population=population),
                f2_transfer(values, effective_injection, "closed", "open", population=population),
                f2_transfer(values, effective_injection, "open", "open", population=population),
            )
            pre_values.append(population_pre["abs_h2"])
            open_values.extend(item["abs_h2"] for item in population_open)
    return {
        "guard_mask_abs_h2": [min(guard_mask_h), max(guard_mask_h)],
        "guard_mask_phase_h2_rad": [min(guard_mask_phase), max(guard_mask_phase)],
        "wrong_node_abs_h2": [min(wrong_node_h), max(wrong_node_h)],
        "wrong_node_phase_h2_rad": [min(wrong_node_phase), max(wrong_node_phase)],
        "detector_only_pre_abs_h2": [min(detector_only_pre_h), max(detector_only_pre_h)],
        "detector_only_isolated_abs_h2": [min(detector_only_open_h), max(detector_only_open_h)],
        "dummy_1pf_pre_abs_h2": [min(dummy_1pf_pre_h), max(dummy_1pf_pre_h)],
        "dummy_1pf_isolated_abs_h2": [min(dummy_1pf_open_h), max(dummy_1pf_open_h)],
    }


def round_numbers(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite output")
        return float(f"{value:.12g}")
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    return value


def build_document() -> dict[str, Any]:
    corners = corner_values()
    bases = {resistor: candidate_envelope(corners, resistor, 0.100) for resistor in RESISTOR_CANDIDATES_OHM}
    candidates = [rescale_candidate(bases[resistor], amplitude) for resistor in RESISTOR_CANDIDATES_OHM for amplitude in C2_AMPLITUDE_CANDIDATES_VPP]
    requested = next(item for item in candidates if item["injection_resistor_ohm"] == FIRST_CANDIDATE_RESISTOR_OHM and item["amplitude_vpp"] == FIRST_CANDIDATE_AMPLITUDE_VPP)
    selected = requested if requested["feasible"] else next((item for item in candidates if item["feasible"]), None)
    if selected is None:
        raise RuntimeError("no resistor/amplitude candidate satisfies the frozen feasibility gates")
    if selected["injection_resistor_ohm"] != EXPECTED_SELECTED_RESISTOR_OHM or selected["amplitude_vpp"] != EXPECTED_SELECTED_AMPLITUDE_VPP:
        raise RuntimeError(f"unexpected first feasible candidate: {selected['injection_resistor_ohm']} ohm at {selected['amplitude_vpp']} Vpp")
    selected = rescale_candidate(bases[selected["injection_resistor_ohm"]], selected["amplitude_vpp"])
    selected["envelope"].update(selected_control_envelope(corners, selected["injection_resistor_ohm"]))
    candidates = [
        selected if item["injection_resistor_ohm"] == selected["injection_resistor_ohm"] and item["amplitude_vpp"] == selected["amplitude_vpp"] else item
        for item in candidates
    ]
    envelope = selected["envelope"]
    # Thresholds are prospective and padded away from the complete selected
    # corner envelope.  They are fixed here before primary synthetic fixtures.
    thresholds = {
        "clipping_abs_v_max": 0.45,
        "common_mode_abs_v_max": 0.10,
        "detector_only_isolated_abs_h2_max": math.ceil((envelope["detector_only_isolated_abs_h2"][1] + 0.010) * 1000.0) / 1000.0,
        "dummy_1pf_isolated_abs_h2_max": math.ceil((envelope["dummy_1pf_isolated_abs_h2"][1] + 0.010) * 1000.0) / 1000.0,
        "fit_condition_number_max": 1_000_000.0,
        "fit_rank_required": 5,
        "frozen_before_primary": True,
        "isolated_abs_h2_max": math.ceil((envelope["isolated_abs_h2"][1] + 0.010) * 1000.0) / 1000.0,
        "isolated_phase_h2_rad": {
            "maximum": math.ceil((envelope["isolated_phase_h2_rad"][1] + 0.05) * 1000.0) / 1000.0,
            "minimum": math.floor((envelope["isolated_phase_h2_rad"][0] - 0.05) * 1000.0) / 1000.0,
        },
        "isolated_u95_h2_max": 0.010,
        "minimum_pre_abs_h2": math.floor((envelope["pre_abs_h2"][0] - 0.010) * 1000.0) / 1000.0,
        "maximum_pre_abs_h2": math.ceil((envelope["pre_abs_h2"][1] + 0.010) * 1000.0) / 1000.0,
        "minimum_pre_open_complex_separation": 0.020,
        "minimum_pre_pilot_snr": 20.0,
        "nonlinear_or_mechanical_2f_residue_ratio_max": 0.020,
        "open_window": {"end_offset_from_series_run_start_samples": 980, "samples": OPEN_WINDOW_SAMPLES, "start_offset_from_series_run_start_samples": 20, "valid_cycles": OPEN_WINDOW_SAMPLES * F2 / FS},
        "pilot_f1_fractional_perturbation_max": 0.050,
        "pre_phase_h2_rad": {"maximum": math.ceil((envelope["pre_phase_h2_rad"][1] + 0.05) * 1000.0) / 1000.0, "minimum": math.floor((envelope["pre_phase_h2_rad"][0] - 0.05) * 1000.0) / 1000.0},
        "pre_window": {"end_offset_from_gate_samples": 240, "samples": PRE_WINDOW_SAMPLES, "start_offset_from_gate_samples": 48, "valid_cycles": PRE_WINDOW_SAMPLES * F2 / FS},
        "r_drop_max": math.ceil((envelope["r_drop"][1] + 0.02) * 1000.0) / 1000.0,
        "same_adg_state_both_windows": "OFF_D_TO_SA_50R",
    }
    if envelope["pre_open_complex_h2_separation"][0] < thresholds["minimum_pre_open_complex_separation"]:
        raise RuntimeError("frozen direct complex-separation threshold is not satisfied")
    isolated_phase = thresholds["isolated_phase_h2_rad"]
    guard_phase = envelope["guard_mask_phase_h2_rad"]
    if not (guard_phase[1] < isolated_phase["minimum"] or guard_phase[0] > isolated_phase["maximum"]):
        raise RuntimeError("K3 guard phase overlaps legitimate isolated phase")
    body = {
        "authority": AUTHORITY,
        "candidate_grid": candidates,
        "claim_ceiling": CEILING,
        "contact_attestation": {"audio_playback_or_recording": 0, "cart_or_stock_check": 0, "hardware": 0, "human_vendor_outreach": 0, "instrument_command": 0, "purchase": 0, "target": 0},
        "corner_count_per_candidate": len(corners),
        "decision": "P0_SIGNAL_PATH_WITNESS_MODEL_FEASIBLE",
        "edge_conventions": {"complex_phase": "atan2_imag_real_in_minus_pi_pi", "intervals": "start_inclusive_stop_exclusive", "maxima": "all_complete_binary_corners_included", "rounding": "binary64_then_12_significant_decimal_digits", "uncertainty": "two_sided_95_percent_1.96_sigma"},
        "frozen_thresholds": thresholds,
        "mechanism": {"first_candidate_rejected": not requested["feasible"], "injection_node": "N_GATE_OUT", "path": ["N_GATE_OUT", "K1_SIGNAL", "N_MIDPOINT", "K2_SIGNAL", "N_ELECTRODE_A", "OPA810", "CH1"], "reference_monitor": "C2_REF_IN_TO_R_MON_C2_TO_CH0", "selected_amplitude_vpp": selected["amplitude_vpp"], "selected_network": {"drive_shunt_node": "N_SRC", "drive_shunt_part": "Vishay TNPW0805100KBEEN", "drive_shunt_resistance_ohm": FIXED_PARAMETERS["drive_shunt_resistance_ohm"], "injection_part": "Vishay TNPW08051M00BEEN", "injection_parts_per_channel": 1}, "selected_part": "Vishay TNPW08051M00BEEN", "selected_resistance_ohm": selected["injection_resistor_ohm"]},
        "fixed_parameters": FIXED_PARAMETERS,
        "model_scope": {"included": ["SDG_C2_50_OHM_OUTPUT", "C1_AND_C2_PASSIVE_MONITOR", "HIGH_VALUE_C2_INJECTION_WITH_PART_TOLERANCE", "N_SRC_DRIVE_SHUNT_WITH_PART_TOLERANCE", "ADG1419_EXPLICIT_SB_D_SA_OFF_TO_50_OHM", "K1_K2_CONTACT_R_AND_C", "K3_ENERGIZED_OPEN_CAPACITANCE", "FC135_BVD_DERIVED_FROM_12P5_PF_LOADED_FREQUENCY", "OPA810_INPUT_OUTPUT_AND_GAIN", "DIGITIZER_1_MOHM_PARALLEL_30_PF_DIFFERENTIAL_AND_COMMON_MODE_LOADING", "BOARD_CABLE_CONNECTOR_AND_ENCLOSURE_CAPACITANCE", "RESISTOR_AND_AMPLIFIER_NOISE", "C2_TO_C1_LOADING", "DUT_DETECTOR_ONLY_AND_EXACT_1PF_POPULATIONS", "RAW_BOUND_2F_NONLINEAR_RESIDUE_GATE"], "physical_observation": False},
        "parameter_ranges": {name: [low, high] for name, (low, high) in RANGES.items()},
        "selected_envelope": selected,
        "schema": "p0.signal-path-circuit-model.v1",
        "sweep_law": {"amplitude_candidates_vpp": list(C2_AMPLITUDE_CANDIDATES_VPP), "binary_corner_parameters": list(RANGES), "candidate_count": len(candidates), "complete_binary_corners": True, "feasibility_gates": {"f1_carrier_terminal_vpp_max": 0.200, "f1_fractional_perturbation_max": 0.050, "f1_motional_current_ua_rms_max": 2.000, "f1_motional_power_uw_max": 0.100, "f2_carrier_terminal_vpp_max": 0.050, "isolated_u95_h2_max": 0.010, "pre_open_separation_min": 0.025, "pre_snr_min": 20.0}, "resistor_candidates_ohm": list(RESISTOR_CANDIDATES_OHM)},
    }
    rounded = round_numbers(body)
    rounded["thresholds_sha256"] = sha256(canonical(rounded["frozen_thresholds"]))
    return rounded


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args not in (["build"], ["verify"]):
        print("usage: p0_signal_path_witness_model.py {build|verify}", file=sys.stderr)
        return 2
    data = canonical(build_document())
    if args == ["build"]:
        temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(OUTPUT)
    elif not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
        print("FAIL: P0_SIGNAL_PATH_CIRCUIT_MODEL.json is stale", file=sys.stderr)
        return 1
    print(json.dumps({"bytes": len(data), "result": "PASS", "sha256": sha256(data)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

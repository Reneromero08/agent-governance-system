#!/usr/bin/env python3
"""Generate the deterministic non-executing P0 physical design packet.

This generator performs no network, hardware, purchasing, audio, or instrument action.
It writes only prospective design documents beside itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESEARCH_RELATIVE_PATH = "research/P0_research_bundle_2026-07-18"
RESEARCH_ROOT = ROOT / RESEARCH_RELATIVE_PATH
RESEARCH_SOURCE_COMMIT = "cb53976612cbe83bec82df826a9889418f7e0b89"
AUTHORITY = "AUTHORIZE P0 BUILD-READINESS ONLY"
CEILING = "NON_EXECUTING_P0_BUILD_READINESS_ONLY"
NEXT = "USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD"
ASSEMBLIES = (
    ("A", "P0-DUT-A", "DUT", "FC-135 populated"),
    ("B", "P0-DETECTOR-B", "DETECTOR_ONLY", "carrier position deliberately unpopulated"),
    ("C", "P0-DUMMY-C0-C", "DUMMY_C0", "1.0 pF C0G dummy populated instead of carrier"),
)

DOC = {
    "EPSON_FC135": {"bytes": 161924, "sha256": "7906f6bce8e15c8e4c570e31969ec27c1f4880917e28160529e898a0cbbf48b9", "revision": "Q13FC13500004xx exact-product brief"},
    "SIGLENT_PROGRAMMING": {"bytes": 4240871, "sha256": "a27c841ef10ebeba8c437be88933079b358d80d55d20b0d3bbf032cbc8b7125d", "revision": "historical captured bytes; current product-page listing E05C 2026-06-30"},
    "SIGLENT_MANUAL": {"bytes": 2930139, "sha256": "11c325f98fea514659be9790a001e90e445119584e31fd8d796b33e92d6e4bed", "revision": "historical captured bytes; current product-page listing EN01J 2025-09-22"},
    "SIGLENT_DATASHEET": {"bytes": 2572702, "sha256": "ca889ea73c85de7aef40d1faf2e85212ea6ed1d16435ae17c279a858a1d99d3a", "revision": "historical captured bytes; current product-page listing EN01I 2025-03-18"},
    "SPECTRUM_DATASHEET": {"bytes": 1060774, "sha256": "5bba0c74b950ac27e447bb25df70973ce19ff8a7a3e4d784378e25a9407d8925", "revision": "official direct-URL bytes match retained hash; product-page listing 2026-05-19; no distinct newer byte revision asserted"},
    "SPECTRUM_MANUAL": {"bytes": 13171754, "sha256": "0cef0929de585c056ecc7605c570ba05c77b0f4fc6c414e393a2c1e578f6ca05", "revision": "historical captured bytes; current product-page listing 2026-05-19"},
    "OPA810": {"bytes": 3406332, "sha256": "74c61ac238989c94c1cf0d70da41bff6e167a590f86e06bae7cdd734d8fd26fa", "revision": "SBOS799E August 2019 revised August 2024"},
    "VISHAY_TNPW": {"bytes": 363615, "sha256": "7bc391d199cc9f9fd0047b6c406bff67d72ec834910b1ab1b4956fcfceb55e08", "revision": "TNPW e3 document 28758, revision 10-Apr-2026; legacy doc 31006 retained in research custody"},
    "VISHAY_CRHV": {"bytes": 117230, "sha256": "8826283500cc095ea3ba24dcf3080bd9072ec90afa8cf7487d68b99bd11b8d6b", "revision": "doc 68002 2024-11-12"},
    "NEXPERIA_2N7002": {"bytes": 284517, "sha256": "3ef49e87f0304160e534bb2750066bf596155b7d42578f3bc40a98955545e77e", "revision": "captured historical exact-part file; type is Not for Design In"},
    "NEXPERIA_1N4148": {"bytes": 16777, "sha256": "13f368ed2f370fe0a613fe1993183cce06717202bcd6a96b912bde017f5cea0e", "revision": "legacy exact-part hash; current first-party source unresolved"},
    "NEXPERIA_BZT52": {"bytes": 336144, "sha256": "a5e54de45bdf1af712c2d9fdfb0fd092804cd83b26eb66e3656b12dea0c8b592", "revision": "current official file"},
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_canonical_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if data != canonical(value):
        raise ValueError(f"noncanonical research metadata: {path.name}")
    if not isinstance(value, dict):
        raise ValueError(f"research metadata root must be an object: {path.name}")
    return value


def research_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_canonical_json(RESEARCH_ROOT / "MANIFEST.json")
    custody = read_canonical_json(RESEARCH_ROOT / "SOURCE_CUSTODY.json")
    if manifest.get("schema") != "p0.research-bundle-manifest.v1" or manifest.get("record_count") != 35:
        raise ValueError("research manifest identity/count")
    if custody.get("schema") != "p0.research-source-custody.v1" or custody.get("source_record_count") != 35:
        raise ValueError("research custody identity/count")
    if custody.get("source_commit") != RESEARCH_SOURCE_COMMIT:
        raise ValueError("research source commit")
    return manifest, custody


def component(ref: str, board: str, manufacturer: str, part: str, function: str, pins: dict[str, str], domain: str, exact_document: str) -> dict[str, Any]:
    return {
        "board": board,
        "domain": domain,
        "exact_document": exact_document,
        "function": function,
        "manufacturer": manufacturer,
        "part_number": part,
        "pins": pins,
        "ref": ref,
    }


def suffixed(mapping: dict[str, str], suffix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pin, net in mapping.items():
        if net.startswith("NC::"):
            ref_pin = net.removeprefix("NC::")
            ref, separator, isolated_pin = ref_pin.partition(".")
            if not separator:
                raise ValueError(f"invalid no-connect sentinel: {net}")
            result[pin] = f"NC::{ref}_{suffix}.{isolated_pin}"
        else:
            result[pin] = f"{net}_{suffix}"
    return result


def build_netlist() -> dict[str, Any]:
    signal_model = read_canonical_json(ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json")
    signal_envelope = signal_model["selected_envelope"]["envelope"]
    boards: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    connectors: list[dict[str, Any]] = []
    harnesses: list[dict[str, Any]] = []
    internal_harnesses: list[dict[str, Any]] = []
    external_cables: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []
    forbidden: list[dict[str, str]] = []
    channels: list[dict[str, str]] = []
    power_domains: list[dict[str, Any]] = []
    relay_contacts: list[dict[str, Any]] = []
    relay_drivers: list[dict[str, Any]] = []
    test_points: list[dict[str, Any]] = []

    for suffix, assembly, fixture_class, population in ASSEMBLIES:
        ctrl_board = f"P0-SOURCE-OFF-CONTROL-REV-B-{suffix}"
        carrier_board = f"P0-CARRIER-SENSE-{fixture_class}-REV-B-{suffix}"
        env_board = f"P0-ENV-SENSOR-REV-B-{suffix}"
        controller_board = f"NUCLEO-G031K8-{suffix}"
        boards.extend([
            {"assembly": assembly, "board_id": ctrl_board, "function": "source-off, passive dual-tone monitor, witness, isolated power and relay control", "outline_mm": [84.0, 64.0, 1.6], "population": "identical across A/B/C"},
            {"assembly": assembly, "board_id": carrier_board, "function": "K2 final barrier, carrier/control load and OPA810 sense", "outline_mm": [68.0, 32.0, 1.0], "population": population},
            {"assembly": assembly, "board_id": env_board, "function": "ADXL354 analog vibration sensor", "outline_mm": [24.0, 24.0, 1.0], "population": "identical across A/B/C"},
            {"assembly": assembly, "board_id": controller_board, "function": "prospective exact controller board; timing and SHT45 record only", "outline_mm": [50.0, 18.0, 7.0], "population": "prospective exact Nucleo-32 board; no inventory claim"},
        ])

        def c(ref: str, board: str, manufacturer: str, part: str, function: str, pins: dict[str, str], domain: str, doc: str) -> None:
            components.append(component(f"{ref}_{suffix}", board, manufacturer, part, function, suffixed(pins, suffix), domain, doc))

        c("R_LIMIT", ctrl_board, "Vishay", "TNPW0805100KBEEN", "C1 source limiter", {"1": "C1_IN", "2": "N_SRC"}, "ANALOG", "VISHAY_TNPW")
        c("R_MON_C1", ctrl_board, "Vishay", "TNPW0805100KBEEN", "upstream C1 monitor contribution", {"1": "C1_IN", "2": "N_SOURCE_MONITOR_SUM"}, "ANALOG", "VISHAY_TNPW")
        c("R_MON_C2", ctrl_board, "Vishay", "TNPW0805100KBEEN", "fixed 2x reference-tone monitor contribution", {"1": "C2_REF_IN", "2": "N_SOURCE_MONITOR_SUM"}, "ANALOG", "VISHAY_TNPW")
        c("R_MON_BIAS", ctrl_board, "Vishay", "TNPW08051M00BEEN", "CH0 passive monitor return", {"1": "N_SOURCE_MONITOR_SUM", "2": "AGND_EXPORT"}, "ANALOG", "VISHAY_TNPW")
        c("R_C2_INJECT", ctrl_board, "Vishay", "TNPW08051M00BEEN", "continuous 1.00 Mohm C2 actual-signal-path witness injection downstream of ADG1419 and upstream of K1", {"1": "C2_REF_IN", "2": "N_GATE_OUT"}, "ANALOG", "VISHAY_TNPW")
        c("R_DRIVE_SHUNT", ctrl_board, "Vishay", "TNPW0805100KBEEN", "100 kohm N_SRC drive shunt enforcing the 0.200 Vpp carrier-terminal cap; isolated from N_GATE_OUT when ADG1419 is OFF", {"1": "N_SRC", "2": "AGND_EXPORT"}, "ANALOG", "VISHAY_TNPW")
        c("U_GATE", ctrl_board, "Analog Devices", "ADG1419BRMZ", "C1 DRIVE/OFF route", {"1": "N_SRC", "2": "N_GATE_TERM", "3": "AGND_EXPORT", "4": "+5V_GATE", "5": "NC::U_GATE.5", "6": "IN_GATE", "7": "-5V_GATE", "8": "N_GATE_OUT"}, "GATE", "ADG1419_REV_A")
        c("R_TERM", ctrl_board, "Vishay", "TNPW080550R0BEEN", "normal 50.00 ohm OFF termination", {"1": "N_GATE_TERM", "2": "AGND_EXPORT"}, "ANALOG", "VISHAY_TNPW")
        c("R_TERM_CONTROL", ctrl_board, "Vishay", "TNPW080575R0BEEN", "75.00 ohm wrong-termination control; DNP in primary configuration", {"1": "DNP::TERM_CONTROL", "2": "DNP::TERM_CONTROL"}, "CONTROL_FIXTURE", "VISHAY_TNPW")
        c("K1", ctrl_board, "Omron", "G6K-2F-Y DC5", "first normally-open series barrier and auxiliary witness", {"1": "+5V_RELAY", "2": "NC::K1.2", "3": "N_GATE_OUT", "4": "N_MIDPOINT", "5": "N_WIT_K1", "6": "ADR_REF_3V3", "7": "NC::K1.7", "8": "K1_COIL_LOW"}, "RELAY", "OMRON_G6K_2026_06_01")
        c("K3", ctrl_board, "Omron", "G6K-2F-Y DC5", "normally-closed midpoint guard and auxiliary witness", {"1": "+5V_RELAY", "2": "N_GUARD_TERM", "3": "N_MIDPOINT", "4": "NC::K3.4", "5": "NC::K3.5", "6": "ADR_REF_3V3", "7": "N_WIT_K3", "8": "K3_COIL_LOW"}, "RELAY", "OMRON_G6K_2026_06_01")
        c("R_GUARD", ctrl_board, "Vishay", "TNPW080550R0BEEN", "midpoint guard termination", {"1": "N_GUARD_TERM", "2": "AGND_EXPORT"}, "ANALOG", "VISHAY_TNPW")
        c("K2", carrier_board, "Omron", "G6K-2F-Y DC5", "final normally-open series barrier and auxiliary witness", {"1": "+5V_RELAY", "2": "NC::K2.2", "3": "N_MIDPOINT", "4": "N_ELECTRODE_A", "5": "N_WIT_K2", "6": "ADR_REF_3V3", "7": "NC::K2.7", "8": "K2_COIL_LOW"}, "RELAY", "OMRON_G6K_2026_06_01")
        c("R_BIAS", carrier_board, "Vishay Techno", "CRHV1206AF150MFKFB", "electrode-A passive return", {"1": "N_ELECTRODE_A", "2": "AGND_STAR"}, "SENSE", "VISHAY_CRHV")
        c("U_SENSE", carrier_board, "Texas Instruments", "OPA810IDT", "unity-gain high-impedance sense buffer", {"1": "NC::U_SENSE.1", "2": "N_SENSE_OUT", "3": "N_ELECTRODE_A", "4": "-5V_SENSE", "5": "NC::U_SENSE.5", "6": "N_SENSE_OUT", "7": "+5V_SENSE", "8": "NC::U_SENSE.8"}, "SENSE", "OPA810")
        c("R_SENSE_OUT", carrier_board, "Vishay", "TNPW080549R9BEEN", "sense-output isolation resistor", {"1": "N_SENSE_OUT", "2": "N_CH1_OUT"}, "SENSE", "VISHAY_TNPW")
        if fixture_class == "DUT":
            c("X1", carrier_board, "Epson", "Q13FC1350000401", "mechanical quartz tuning-fork carrier", {"1": "N_ELECTRODE_A", "2": "AGND_STAR"}, "CARRIER", "EPSON_FC135")
        elif fixture_class == "DUMMY_C0":
            c("C_DUMMY", carrier_board, "Murata", "GJM1555C1H1R0BB01D", "1.0 pF C0G electrical dummy replacing the carrier", {"1": "N_ELECTRODE_A", "2": "AGND_STAR"}, "CONTROL_FIXTURE", "MURATA_GJM_CURRENT")

        c("UISO_GATE", ctrl_board, "Analog Devices", "ADuM140D0BRZ", "fail-safe-low isolated gate command; primary and secondary logic are 3.3 V", {"1": "CTRL_3V3", "2": "CTRL_GND", "3": "CMD_GATE", "4": "CTRL_GND", "5": "CTRL_GND", "6": "CTRL_GND", "7": "CTRL_GND", "8": "CTRL_GND", "9": "AGND_EXPORT", "10": "NC::UISO_GATE.10", "11": "NC::UISO_GATE.11", "12": "NC::UISO_GATE.12", "13": "NC::UISO_GATE.13", "14": "IN_GATE", "15": "AGND_EXPORT", "16": "ADR_REF_3V3"}, "ISOLATION", "ADUM140D_REV_K")
        c("UISO_RELAY", ctrl_board, "Analog Devices", "ADuM140D0BRZ", "fail-safe-low isolated K1/K2/K3 commands; primary is 3.3 V", {"1": "CTRL_3V3", "2": "CTRL_GND", "3": "CMD_K1", "4": "CMD_K2", "5": "CMD_K3", "6": "CTRL_GND", "7": "CTRL_GND", "8": "CTRL_GND", "9": "GND_RELAY", "10": "NC::UISO_RELAY.10", "11": "NC::UISO_RELAY.11", "12": "DRV_K3", "13": "DRV_K2", "14": "DRV_K1", "15": "GND_RELAY", "16": "+5V_RELAY"}, "ISOLATION", "ADUM140D_REV_K")
        c("PS_GATE", ctrl_board, "Murata Power Solutions", "MEV1D0505SC", "isolated gate supply", {"+VIN": "CTRL_5V_FUSED_GATE", "-VIN": "CTRL_GND", "+VOUT": "+5V_GATE", "0V": "AGND_EXPORT", "-VOUT": "-5V_GATE"}, "POWER", "MURATA_MEV1_CURRENT")
        c("PS_SENSE", ctrl_board, "Murata Power Solutions", "MEV1D0505SC", "isolated sense supply", {"+VIN": "CTRL_5V_FUSED_SENSE", "-VIN": "CTRL_GND", "+VOUT": "+5V_SENSE", "0V": "AGND_EXPORT", "-VOUT": "-5V_SENSE"}, "POWER", "MURATA_MEV1_CURRENT")
        c("PS_RELAY", ctrl_board, "Murata Power Solutions", "MEV1D0505SC", "isolated relay supply; negative output deliberately unused", {"+VIN": "CTRL_5V_FUSED_RELAY", "-VIN": "CTRL_GND", "+VOUT": "+5V_RELAY", "0V": "GND_RELAY", "-VOUT": "NC::PS_RELAY.-VOUT"}, "POWER", "MURATA_MEV1_CURRENT")
        for index, output in ((1, "CTRL_5V_FUSED_GATE"), (2, "CTRL_5V_FUSED_SENSE"), (3, "CTRL_5V_FUSED_RELAY")):
            c(f"F{index}", ctrl_board, "Littelfuse", "0467.500NR", f"500 mA primary protection for isolated converter {index}", {"1": "CTRL_5V", "2": output}, "POWER", "LITTELFUSE_0467_CURRENT")
        c("U_REF", ctrl_board, "Analog Devices", "ADR4533BRZ", "witness reference", {"1": "NC::U_REF.1", "2": "+5V_SENSE", "3": "NC::U_REF.3", "4": "AGND_EXPORT", "5": "NC::U_REF.5", "6": "ADR_REF_3V3", "7": "NC::U_REF.7", "8": "NC::U_REF.8"}, "ANALOG", "ADR45XX_REV_G")
        for index, value, net in ((0, "80K6", "N_WIT_GATE"), (1, "40K2", "N_WIT_K1"), (2, "20K0", "N_WIT_K2"), (3, "10K0", "N_WIT_K3")):
            selector = "IN_GATE" if index == 0 else net
            c(f"R_W{index}", ctrl_board, "Vishay", f"TNPW0805{value}BEEN", f"CH2 witness bit {index}", {"1": selector, "2": "N_WITNESS"}, "ANALOG", "VISHAY_TNPW")
        c("R_WPULL", ctrl_board, "Vishay", "TNPW08051K00BEEN", "CH2 witness pull-down", {"1": "N_WITNESS", "2": "AGND_EXPORT"}, "ANALOG", "VISHAY_TNPW")

        bypasses = (
            ("C_GATE_IN", "CTRL_5V_FUSED_GATE", "CTRL_GND"),
            ("C_GATE_POS", "+5V_GATE", "AGND_EXPORT"),
            ("C_GATE_NEG", "-5V_GATE", "AGND_EXPORT"),
            ("C_SENSE_IN", "CTRL_5V_FUSED_SENSE", "CTRL_GND"),
            ("C_SENSE_POS", "+5V_SENSE", "AGND_EXPORT"),
            ("C_SENSE_NEG", "-5V_SENSE", "AGND_EXPORT"),
            ("C_RELAY_IN", "CTRL_5V_FUSED_RELAY", "CTRL_GND"),
            ("C_RELAY_POS", "+5V_RELAY", "GND_RELAY"),
            ("C_ISO_GATE_PRI", "CTRL_3V3", "CTRL_GND"),
            ("C_ISO_GATE_SEC", "ADR_REF_3V3", "AGND_EXPORT"),
            ("C_ISO_RELAY_PRI", "CTRL_3V3", "CTRL_GND"),
            ("C_ISO_RELAY_SEC", "+5V_RELAY", "GND_RELAY"),
            ("C_REF_IN", "+5V_SENSE", "AGND_EXPORT"),
            ("C_REF_OUT", "ADR_REF_3V3", "AGND_EXPORT"),
        )
        for ref, positive, negative in bypasses:
            c(ref, ctrl_board, "Murata", "GRM21BR71C104KA01L", f"local 100 nF bypass for {positive}", {"1": positive, "2": negative}, "POWER", "MURATA_GRM_CURRENT")
        c("C_OPA_POS", carrier_board, "Murata", "GRM21BR71C104KA01L", "OPA810 +5 V local bypass", {"1": "+5V_SENSE", "2": "AGND_STAR"}, "SENSE", "MURATA_GRM_CURRENT")
        c("C_OPA_NEG", carrier_board, "Murata", "GRM21BR71C104KA01L", "OPA810 -5 V local bypass", {"1": "-5V_SENSE", "2": "AGND_STAR"}, "SENSE", "MURATA_GRM_CURRENT")

        for index in (1, 2, 3):
            c(f"Q{index}", ctrl_board, "Nexperia", "2N7002PW,115", f"K{index} low-side driver", {"1": f"Q{index}_GATE", "2": "GND_RELAY", "3": f"K{index}_COIL_LOW"}, "RELAY", "NEXPERIA_2N7002")
            c(f"D{index}", ctrl_board, "Nexperia", "1N4148W,115", f"K{index} clamp steering diode", {"A": f"K{index}_COIL_LOW", "K": f"K{index}_CLAMP"}, "RELAY", "NEXPERIA_1N4148")
            c(f"DZ{index}", ctrl_board, "Nexperia", "BZT52H-C12,115", f"K{index} 12 V release clamp", {"A": f"K{index}_CLAMP", "K": "+5V_RELAY"}, "RELAY", "NEXPERIA_BZT52")
            c(f"R_G{index}", ctrl_board, "Vishay", "TNPW08051K00BEEN", f"K{index} gate series resistor", {"1": f"DRV_K{index}", "2": f"Q{index}_GATE"}, "RELAY", "VISHAY_TNPW")
            c(f"R_GPD{index}", ctrl_board, "Vishay", "TNPW0805100KBEEN", f"K{index} gate fail-low pull-down", {"1": f"Q{index}_GATE", "2": "GND_RELAY"}, "RELAY", "VISHAY_TNPW")

        c("CTRL1", controller_board, "STMicroelectronics", "NUCLEO-G031K8", "sequencer and SHT45 recorder; never a waveform store", {"CN3-1/PB6": "ENV_SCL", "CN3-2/PB7": "ENV_SDA", "CN3-5/PA15": "CMD_GATE", "CN3-6/PB1": "CMD_K1", "CN3-7/PA10": "CMD_K2", "CN3-8/PA9": "CMD_K3", "CN3-9/PB0": "NC::CTRL1.CN3-9/PB0", "CN4-2": "CTRL_GND", "CN4-4": "CTRL_5V", "CN4-14": "CTRL_3V3"}, "CONTROL", "ST_UM2591_REV2")
        c("U_TEMP_RH", ctrl_board, "Sensirion", "SHT45-AD1B-R2", "ambient temperature/RH record with raw-word CRC", {"1": "ENV_SDA", "2": "ENV_SCL", "3": "CTRL_3V3", "4": "CTRL_GND"}, "CONTROL", "SHT4X_2025")
        c("C_SHT", ctrl_board, "Murata", "GRM21BR71C104KA01L", "SHT45 local bypass", {"1": "CTRL_3V3", "2": "CTRL_GND"}, "CONTROL", "MURATA_GRM_CURRENT")
        c("R_SDA", ctrl_board, "Vishay", "TNPW080510K0BEEN", "SHT45 SDA pull-up", {"1": "ENV_SDA", "2": "CTRL_3V3"}, "CONTROL", "VISHAY_TNPW")
        c("R_SCL", ctrl_board, "Vishay", "TNPW080510K0BEEN", "SHT45 SCL pull-up", {"1": "ENV_SCL", "2": "CTRL_3V3"}, "CONTROL", "VISHAY_TNPW")

        c("U_ACCEL", env_board, "Analog Devices", "ADXL354CEZ", "carrier-enclosure vibration witness", {"1": "AGND_STAR", "2": "AGND_STAR", "3": "AGND_STAR", "4": "NC::U_ACCEL.4", "5": "ADR_REF_3V3", "6": "AGND_STAR", "7": "ADR_REF_3V3", "8": "ADXL_1V8_DIG", "9": "AGND_STAR", "10": "ADXL_1V8_ANA", "11": "ADR_REF_3V3", "12": "ADXL_XOUT", "13": "ADXL_YOUT", "14": "N_ACCEL_Z"}, "ANALOG", "ADXL354_REV_D")
        for ref, net in (("C_1V8_DIG", "ADXL_1V8_DIG"), ("C_1V8_ANA", "ADXL_1V8_ANA"), ("C_VDDIO", "ADR_REF_3V3"), ("C_VSUPPLY", "ADR_REF_3V3")):
            c(ref, env_board, "Murata", "GRM21BR71C104KA01L", f"ADXL354 local bypass at {net}", {"1": net, "2": "AGND_STAR"}, "ANALOG", "MURATA_GRM_CURRENT")
        for ref, net in (("C_XOUT", "ADXL_XOUT"), ("C_YOUT", "ADXL_YOUT"), ("C_ZOUT", "N_ACCEL_Z")):
            c(ref, env_board, "KEMET", "C0805C103J5GACTU", f"ADXL354 exact 10 nF C0G output filter at {net}", {"1": net, "2": "AGND_STAR"}, "ANALOG", "KEMET_C0G_CURRENT")
        c("R_DISCH_DIG", env_board, "Vishay", "TNPW0805100KBEEN", "ADXL354 V1P8DIG discharge", {"1": "ADXL_1V8_DIG", "2": "AGND_STAR"}, "ANALOG", "VISHAY_TNPW")
        c("R_DISCH_ANA", env_board, "Vishay", "TNPW0805100KBEEN", "ADXL354 V1P8ANA discharge", {"1": "ADXL_1V8_ANA", "2": "AGND_STAR"}, "ANALOG", "VISHAY_TNPW")

        connectors.extend([
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon; no panel bond", "center": f"C1_IN_{suffix}", "id": f"J_SRC_C1_{suffix}", "part_number": "031-10-RFX", "remote": "SDG1032X C1 output", "return": f"AGND_EXPORT_{suffix}"},
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon; rig-end shell deliberately insulated", "center": f"C2_REF_IN_{suffix}", "id": f"J_SRC_C2_{suffix}", "part_number": "031-10-RFX", "remote": "SDG1032X C2 output; return is shared internally with C1 at the source", "return": f"NC::J_SRC_C2_{suffix}.shell"},
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon", "center": f"N_SOURCE_MONITOR_SUM_{suffix}", "id": f"J_CH0_{suffix}", "part_number": "031-10-RFX", "remote": "DN2.592-04 CH0 true-differential input", "return": f"AGND_EXPORT_{suffix}"},
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon", "center": f"N_CH1_OUT_{suffix}", "id": f"J_CH1_{suffix}", "part_number": "031-10-RFX", "remote": "DN2.592-04 CH1 true-differential input", "return": f"AGND_STAR_{suffix}"},
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon", "center": f"N_WITNESS_{suffix}", "id": f"J_CH2_{suffix}", "part_number": "031-10-RFX", "remote": "DN2.592-04 CH2 true-differential input", "return": f"AGND_EXPORT_{suffix}"},
            {"assembly": assembly, "body_isolation": "manufacturer-integral nylon", "center": f"N_ACCEL_Z_{suffix}", "id": f"J_CH3_{suffix}", "part_number": "031-10-RFX", "remote": "DN2.592-04 CH3 true-differential input", "return": f"AGND_STAR_{suffix}"},
        ])
        channels.extend([
            {"assembly": assembly, "channel": "CH0", "observable": "passive C1 bound-f_carrier source monitor plus C2 bound-2xf_carrier phase-gauge tone", "positive": f"N_SOURCE_MONITOR_SUM_{suffix}", "negative": f"AGND_EXPORT_{suffix}"},
            {"assembly": assembly, "channel": "CH1", "observable": "OPA810 buffered carrier/control node", "positive": f"N_CH1_OUT_{suffix}", "negative": f"AGND_STAR_{suffix}"},
            {"assembly": assembly, "channel": "CH2", "observable": "four-bit gate/K1/K2/K3 witness voltage", "positive": f"N_WITNESS_{suffix}", "negative": f"AGND_EXPORT_{suffix}"},
            {"assembly": assembly, "channel": "CH3", "observable": "ADXL354 Z-axis vibration", "positive": f"N_ACCEL_Z_{suffix}", "negative": f"AGND_STAR_{suffix}"},
        ])
        harnesses.append({
            "assembly": assembly,
            "cable_glands": [{"part_number": "1427CG13", "panel_hole_in": 0.804, "end": "control"}, {"part_number": "1427CG13", "panel_hole_in": 0.804, "end": "carrier"}],
            "finished_length_mm": 150,
            "id": f"P0-INTERENCLOSURE-HARNESS-REV-B-{suffix}",
            "no_connector_or_splice": True,
            "paths": [
                {"conductor": "Belden 83269 RG-178/U center", "net": f"N_MIDPOINT_{suffix}"},
                {"conductor": "Belden 83269 RG-178/U shield", "net": f"AGND_EXPORT_{suffix}<->AGND_STAR_{suffix}", "law": "sole intentional low-impedance AGND bond; nonzero digitizer common-mode admittances are separately calibrated parasitic returns"},
                {"conductor": "Alpha 2840/1 P01", "net": f"+5V_RELAY_{suffix}"},
                {"conductor": "Alpha 2840/1 P02", "net": f"K2_COIL_LOW_{suffix}"},
                {"conductor": "Alpha 2840/1 P03", "net": f"ADR_REF_3V3_{suffix}"},
                {"conductor": "Alpha 2840/1 P04", "net": f"N_WIT_K2_{suffix}"},
                {"conductor": "Alpha 2840/1 P05", "net": f"+5V_SENSE_{suffix}"},
                {"conductor": "Alpha 2840/1 P06", "net": f"-5V_SENSE_{suffix}"},
            ],
            "raw_materials": ["Belden 83269 RG-178/U", "Alpha Wire 2840/1 28-AWG PTFE"],
        })
        internal_harnesses.extend([
            {
                "assembly": assembly,
                "enclosure": f"ENC-CONTROL-{suffix}",
                "id": f"P0-CONTROL-INTERNAL-WIRING-REV-B-{suffix}",
                "no_connector_or_splice": True,
                "paths": [
                    {"cable_id": "P01", "conductor": "Belden 83269 RG-178/U center", "finished_length_mm": 55, "from": f"{ctrl_board}.C1_IN land", "net": f"C1_IN_{suffix}", "to": f"J_SRC_C1_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P01", "conductor": "Belden 83269 RG-178/U shield", "finished_length_mm": 55, "from": f"{ctrl_board}.AGND_EXPORT land", "net": f"AGND_EXPORT_{suffix}", "to": f"J_SRC_C1_{suffix}.return", "tolerance_mm": 2},
                    {"cable_id": "P02", "conductor": "Alpha Wire 2840/1 P10; center-only pigtail", "finished_length_mm": 55, "from": f"{ctrl_board}.C2_REF_IN land", "net": f"C2_REF_IN_{suffix}", "to": f"J_SRC_C2_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P02", "conductor": "NONE", "finished_length_mm": 0, "from": f"J_SRC_C2_{suffix}.shell", "law": "shell remains mechanically isolated with no internal conductor", "net": f"NC::J_SRC_C2_{suffix}.shell", "to": "NO_CONNECTION", "tolerance_mm": 0},
                    {"cable_id": "P03", "conductor": "Belden 83269 RG-178/U center", "finished_length_mm": 55, "from": f"{ctrl_board}.N_SOURCE_MONITOR_SUM land", "net": f"N_SOURCE_MONITOR_SUM_{suffix}", "to": f"J_CH0_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P03", "conductor": "Belden 83269 RG-178/U shield", "finished_length_mm": 55, "from": f"{ctrl_board}.AGND_EXPORT land", "net": f"AGND_EXPORT_{suffix}", "to": f"J_CH0_{suffix}.return", "tolerance_mm": 2},
                    {"cable_id": "P04", "conductor": "Belden 83269 RG-178/U center", "finished_length_mm": 55, "from": f"{ctrl_board}.N_WITNESS land", "net": f"N_WITNESS_{suffix}", "to": f"J_CH2_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P04", "conductor": "Belden 83269 RG-178/U shield", "finished_length_mm": 55, "from": f"{ctrl_board}.AGND_EXPORT land", "net": f"AGND_EXPORT_{suffix}", "to": f"J_CH2_{suffix}.return", "tolerance_mm": 2},
                    *[
                        {"cable_id": f"N{index:02d}", "conductor": f"Alpha Wire 2840/1 N{index:02d}", "finished_length_mm": 60, "from": f"CTRL1_{suffix}.{pin}", "net": f"{net}_{suffix}", "to": f"{ctrl_board}.{net} land", "tolerance_mm": 2}
                        for index, (pin, net) in enumerate((
                            ("CN3-1/PB6", "ENV_SCL"), ("CN3-2/PB7", "ENV_SDA"), ("CN3-5/PA15", "CMD_GATE"),
                            ("CN3-6/PB1", "CMD_K1"), ("CN3-7/PA10", "CMD_K2"), ("CN3-8/PA9", "CMD_K3"),
                            ("CN4-2", "CTRL_GND"), ("CN4-4", "CTRL_5V"), ("CN4-14", "CTRL_3V3"),
                        ), start=1)
                    ],
                ],
                "raw_materials": ["Belden 83269 RG-178/U", "Alpha Wire 2840/1 28-AWG PTFE"],
                "termination_law": "soldered point-to-point only after separate authority; strain-relieved at both ends; no detachable internal interface",
            },
            {
                "assembly": assembly,
                "enclosure": f"ENC-CARRIER-{suffix}",
                "id": f"P0-CARRIER-INTERNAL-WIRING-REV-B-{suffix}",
                "no_connector_or_splice": True,
                "paths": [
                    {"cable_id": "P05", "conductor": "Belden 83269 RG-178/U center", "finished_length_mm": 45, "from": f"{carrier_board}.N_CH1_OUT land", "net": f"N_CH1_OUT_{suffix}", "to": f"J_CH1_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P05", "conductor": "Belden 83269 RG-178/U shield", "finished_length_mm": 45, "from": f"{carrier_board}.AGND_STAR land", "net": f"AGND_STAR_{suffix}", "to": f"J_CH1_{suffix}.return", "tolerance_mm": 2},
                    {"cable_id": "P06", "conductor": "Belden 83269 RG-178/U center", "finished_length_mm": 45, "from": f"{env_board}.N_ACCEL_Z land", "net": f"N_ACCEL_Z_{suffix}", "to": f"J_CH3_{suffix}.center", "tolerance_mm": 2},
                    {"cable_id": "P06", "conductor": "Belden 83269 RG-178/U shield", "finished_length_mm": 45, "from": f"{env_board}.AGND_STAR land", "net": f"AGND_STAR_{suffix}", "to": f"J_CH3_{suffix}.return", "tolerance_mm": 2},
                    {"cable_id": "E01", "conductor": "Alpha Wire 2840/1 E01", "finished_length_mm": 35, "from": f"{carrier_board}.ADR_REF_3V3 land", "net": f"ADR_REF_3V3_{suffix}", "to": f"{env_board}.ADR_REF_3V3 land", "tolerance_mm": 2},
                    {"cable_id": "E02", "conductor": "Alpha Wire 2840/1 E02", "finished_length_mm": 35, "from": f"{carrier_board}.AGND_STAR land", "net": f"AGND_STAR_{suffix}", "to": f"{env_board}.AGND_STAR land", "tolerance_mm": 2},
                ],
                "raw_materials": ["Belden 83269 RG-178/U", "Alpha Wire 2840/1 28-AWG PTFE"],
                "termination_law": "soldered point-to-point only after separate authority; strain-relieved at both ends; kept outside the guarded electrode keepout",
            },
        ])
        for connector_id, remote, shield_law in (
            ("J_SRC_C1", "SDG1032X CH1 BNC", "shield bonded only through J_SRC_C1 return"),
            ("J_SRC_C2", "SDG1032X CH2 BNC", "fixture-end shell remains isolated; cable shield is source-referenced only"),
            ("J_CH0", "DN2.592-04 CH0 differential pair", "return terminates only at declared CH0 negative input"),
            ("J_CH1", "DN2.592-04 CH1 differential pair", "return terminates only at declared CH1 negative input"),
            ("J_CH2", "DN2.592-04 CH2 differential pair", "return terminates only at declared CH2 negative input"),
            ("J_CH3", "DN2.592-04 CH3 differential pair", "return terminates only at declared CH3 negative input"),
        ):
            external_cables.append({"assembly": assembly, "cable_id": f"CBL-{connector_id}-{suffix}", "fixture_end": f"{connector_id}_{suffix}", "part_number": "2249-C-24", "remote_end": remote, "shield_law": shield_law})

        relay_contacts.extend([
            {"assembly": assembly, "common_pin": 3, "deenergized_pin": 2, "energized_pin": 4, "pole": "signal", "ref": f"K1_{suffix}", "role": "series barrier 1"},
            {"assembly": assembly, "common_pin": 6, "deenergized_pin": 7, "energized_pin": 5, "pole": "witness", "ref": f"K1_{suffix}", "role": "b1 high only when energized"},
            {"assembly": assembly, "common_pin": 3, "deenergized_pin": 2, "energized_pin": 4, "pole": "signal", "ref": f"K2_{suffix}", "role": "series barrier 2"},
            {"assembly": assembly, "common_pin": 6, "deenergized_pin": 7, "energized_pin": 5, "pole": "witness", "ref": f"K2_{suffix}", "role": "b2 high only when energized"},
            {"assembly": assembly, "common_pin": 3, "deenergized_pin": 2, "energized_pin": 4, "pole": "signal", "ref": f"K3_{suffix}", "role": "midpoint guard closed when deenergized"},
            {"assembly": assembly, "common_pin": 6, "deenergized_pin": 7, "energized_pin": 5, "pole": "witness", "ref": f"K3_{suffix}", "role": "b3 high only when deenergized"},
        ])
        for index, isolator_pin in ((1, 14), (2, 13), (3, 12)):
            relay_drivers.append({
                "assembly": assembly, "clamp_path": [f"D{index}_{suffix}", f"DZ{index}_{suffix}"], "coil_high": f"+5V_RELAY_{suffix}",
                "coil_low": f"K{index}_COIL_LOW_{suffix}", "command": f"CMD_K{index}_{suffix}", "fail_low": f"R_GPD{index}_{suffix}",
                "isolator_output": f"UISO_RELAY_{suffix}.{isolator_pin}", "mosfet": f"Q{index}_{suffix}", "relay": f"K{index}_{suffix}",
                "series_resistor": f"R_G{index}_{suffix}",
            })
        test_points.extend([
            {"access": f"K1_{suffix} package lands 2/3/4", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K1-SIGNAL-{suffix}", "law": "unpowered continuity only before closure"},
            {"access": f"K1_{suffix} package lands 5/6/7", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K1-WITNESS-{suffix}", "law": "unpowered continuity only before closure"},
            {"access": f"K2_{suffix} package lands 2/3/4 before carrier/dummy population", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K2-SIGNAL-{suffix}", "law": "temporary unpowered access; no permanent electrode test point"},
            {"access": f"K2_{suffix} package lands 5/6/7", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K2-WITNESS-{suffix}", "law": "unpowered continuity only before closure"},
            {"access": f"K3_{suffix} package lands 2/3/4", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K3-SIGNAL-{suffix}", "law": "unpowered continuity only before closure"},
            {"access": f"K3_{suffix} package lands 5/6/7", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-K3-WITNESS-{suffix}", "law": "unpowered continuity only before closure"},
            {"access": f"isolated-supply bypass pads and insulated star hardware for fixture {suffix}", "assembly": assembly, "dedicated_hardware": "NONE", "id": f"TA-RAILS-GROUND-{suffix}", "law": "unpowered resistance/isolation only"},
            {"access": "NONE", "assembly": assembly, "dedicated_hardware": "FORBIDDEN", "id": f"TA-ELECTRODE-{suffix}", "law": f"no test point, cable, clamp or added copper on N_ELECTRODE_A_{suffix}"},
        ])
        fixtures.append({"assembly": assembly, "fixture_class": fixture_class, "population": population, "mutual_exclusion": "only one complete assembly may be connected to the shared source/digitizer during a record", "selection_receipt_required": True})
        continuity.extend([
            {"assembly": assembly, "id": f"CT-{suffix}-K1", "access": "K1 package lands before enclosure closure", "expected_deenergized": "pins 3-4 open; pins 2-3 closed", "four_wire": True},
            {"assembly": assembly, "id": f"CT-{suffix}-K2", "access": "K2 package lands before carrier/X1 or dummy population", "expected_deenergized": "pins 3-4 open; pins 2-3 closed", "four_wire": True},
            {"assembly": assembly, "id": f"CT-{suffix}-K3", "access": "K3 package lands before enclosure closure", "expected_deenergized": "pins 2-3 closed; pins 3-4 open", "four_wire": True},
            {"assembly": assembly, "id": f"CT-{suffix}-GROUND", "access": "enclosure studs and connector returns", "expected_deenergized": "exactly one AGND_EXPORT-to-AGND_STAR link through harness coax shield; no second low-ohmic path", "four_wire": False},
            {"assembly": assembly, "id": f"CT-{suffix}-PATH", "access": "J_SRC_C1 center to carrier/dummy land before carrier population", "expected_deenergized": "open through K1 and K2", "four_wire": False},
        ])
        for pair in (("CTRL_GND", "AGND_EXPORT"), ("GND_RELAY", "AGND_EXPORT"), ("CTRL_GND", "GND_RELAY")):
            forbidden.append({"assembly": assembly, "from": f"{pair[0]}_{suffix}", "to": f"{pair[1]}_{suffix}", "law": "no galvanic connection"})
        forbidden.extend([
            {"assembly": assembly, "from": f"N_ELECTRODE_A_{suffix}", "to": "any cable, test point, clamp, capacitor except X1/C_DUMMY, or connector", "law": "forbidden downstream load"},
            {"assembly": assembly, "from": f"AGND_STAR_{suffix}", "to": "carrier enclosure or control enclosure metal", "law": "stud and BNC bodies remain insulated"},
            {"assembly": assembly, "from": f"J_SRC_C2_{suffix}.shell", "to": f"AGND_EXPORT_{suffix}", "law": "forbidden second source-return path"},
            {"assembly": assembly, "from": "external trigger", "to": "any P0 node", "law": "no trigger cable exists"},
        ])
        power_domains.extend([
            {"assembly": assembly, "domain": "CTRL", "rails": [f"CTRL_5V_{suffix}", f"CTRL_3V3_{suffix}", f"CTRL_GND_{suffix}"], "bond": "USB/controller only"},
            {"assembly": assembly, "domain": "GATE", "rails": [f"+5V_GATE_{suffix}", f"AGND_EXPORT_{suffix}", f"-5V_GATE_{suffix}"], "bond": "AGND_EXPORT joins AGND_STAR only through the harness coax shield"},
            {"assembly": assembly, "domain": "SENSE", "rails": [f"+5V_SENSE_{suffix}", f"AGND_EXPORT_{suffix}", f"-5V_SENSE_{suffix}"], "bond": "same declared analog domain; no second return"},
            {"assembly": assembly, "domain": "RELAY", "rails": [f"+5V_RELAY_{suffix}", f"GND_RELAY_{suffix}"], "bond": "none to analog or controller domains"},
        ])

    witness_resistors = [80600.0, 40200.0, 20000.0, 10000.0]
    nominal_centroids_mv: list[dict[str, float | int]] = []
    for code in range(16):
        # The b0 isolator output is actively driven low when the gate is OFF,
        # so R_W0 remains a shunt branch for every b0=0 code.  The three relay
        # witness branches are open when inactive and therefore do not add a
        # corresponding low-driven conductance.
        conductance = 1.0 / 1000.0 + 1.0 / witness_resistors[0]
        numerator = 3.3 / witness_resistors[0] if code & 1 else 0.0
        for bit, resistance in enumerate(witness_resistors[1:], start=1):
            if code & (1 << bit):
                conductance += 1.0 / resistance
                numerator += 3.3 / resistance
        nominal_centroids_mv.append({"code": code, "millivolts": round(1000.0 * numerator / conductance, 6)})
    minimum_gap_mv = min(
        float(nominal_centroids_mv[index]["millivolts"]) - float(nominal_centroids_mv[index - 1]["millivolts"])
        for index in range(1, len(nominal_centroids_mv))
    )

    members: dict[str, list[str]] = {}
    for item in components:
        for pin, net in item["pins"].items():
            if not net.startswith(("NC::", "DNP::")):
                members.setdefault(net, []).append(f"{item['ref']}.{pin}")
    for item in connectors:
        members.setdefault(item["center"], []).append(f"{item['id']}.center")
        if not item["return"].startswith("NC::"):
            members.setdefault(item["return"], []).append(f"{item['id']}.return")
    nets = [{"members": sorted(value), "name": key} for key, value in sorted(members.items())]

    xc_4pf = 1.0 / (2.0 * math.pi * 32820.0 * 4e-12)
    admittance = {
        "derivation": {"frequency_hz_worst_case": 32820.0, "hard_capacitance_f": 4e-12, "capacitive_reactance_ohm_minimum": round(xc_4pf, 3), "fc135_max_esr_ohm": 70000, "esr_to_reactance_ratio_maximum": round(70000 / xc_4pf, 6)},
        "hard_execution": {"cin_u95_pf_max": 4.0, "rin_u95_mohm_min": 100.0},
        "input_protection": {"external_components": [], "selection": "NONE", "reason": "any external clamp or series protection on N_ELECTRODE_A would violate the frozen capacitance/loading budget", "operating_boundary": "100 kohm source limiter plus K1/K2 isolation bound normal drive; OPA810 absolute input/current and on-die ESD ratings are never treated as operating permission", "handling_boundary": "grounded ESD controls with all sources, instruments and external cables absent"},
        "planning_pf": {"opa810_common_plus_differential_typical": 2.5, "guarded_input_land_and_routing_u95": 0.30, "k2_contact_and_carrier_land_u95": 0.20, "bias_resistor_body_and_pads_u95": 0.15, "contamination_u95": 0.15, "lot_temperature_and_repeat_reserve": 0.30, "total": 3.60, "closure_margin": 0.40},
        "law": "OPA810 capacitance is typical, not guaranteed; each populated carrier/control coupon must independently measure Cin,U95 <=4.00 pF and Rin,U95 >=100 Mohm before full unpowered assembly may proceed.",
    }

    return {
        "admittance_budget": admittance,
        "authority": AUTHORITY,
        "boards": boards,
        "channel_map": channels,
        "claim_ceiling": CEILING,
        "components": components,
        "connector_map": connectors,
        "continuity_tests": continuity,
        "control_fixtures": fixtures,
        "external_cables": external_cables,
        "failure_state_table": [
            {"failure": "controller or USB power loss", "physical_effect": "all isolated supplies decay; K1/K2 open and K3 returns to guard, but CH2/reference custody is unavailable", "scientific_action": "REJECT; unpowered topology may be checked only by continuity"},
            {"failure": "gate-domain supply or ADG1419 route loss", "physical_effect": "ADG1419 route is undefined while the 100 kohm limiter and two relay barriers remain", "scientific_action": "REJECT; no inferred termination state"},
            {"failure": "relay-domain supply loss", "physical_effect": "K1/K2 deenergize open and K3 deenergizes to guard", "scientific_action": "REJECT unless the complete stable-code and actual-path continuity receipts remain valid"},
            {"failure": "sense/reference supply loss", "physical_effect": "CH1, CH2 and ADXL reference custody are invalid", "scientific_action": "REJECT"},
            {"failure": "ADG1419 stuck in DRIVE or wrong termination", "physical_effect": "K1/K2 still provide independent series opens and K3 guards the midpoint", "scientific_action": "REJECT by topology/termination receipt and raw wrong-termination gates"},
            {"failure": "K1 signal pole stuck closed", "physical_effect": "K2 remains an independent series open and K3 guards the midpoint", "scientific_action": "REJECT a known stuck-contact fixture; an otherwise passing end-to-end event can establish path isolation but never identify which individual contact opened"},
            {"failure": "K2 signal pole stuck closed", "physical_effect": "K1 remains an independent series open and K3 guards the midpoint", "scientific_action": "REJECT a known stuck-contact fixture; an otherwise passing end-to-end event can establish path isolation but never identify which individual contact opened"},
            {"failure": "K3 guard pole fails to close", "physical_effect": "K1/K2 remain series-open but midpoint termination is absent", "scientific_action": "REJECT"},
            {"failure": "CH2 ambiguity, illegal code, late contact or re-entry", "physical_effect": "source-off ancestry is not established", "scientific_action": "REJECT with no retry implied"},
        ],
        "forbidden_connections": forbidden,
        "ground_model": {
            "digitizer_inputs": "DN2.592-04 true-differential inputs are not galvanically isolated; each input admittance to system ground is declared, calibrated and never called an isolation barrier.",
            "direct_source_bond": "one low-impedance source return through C1; C2 rig-end shell is open",
            "inter_enclosure_bond": "one intentional low-impedance RG-178 shield bond joins AGND_EXPORT to AGND_STAR; nonzero digitizer common-mode admittances are separately calibrated parasitic returns, not isolation",
            "external_trigger": "absent",
        },
        "intra_enclosure_harnesses": internal_harnesses,
        "inter_enclosure_harnesses": harnesses,
        "nets": nets,
        "power_domains": power_domains,
        "relay_coil_driver_map": relay_drivers,
        "relay_contact_map": relay_contacts,
        "revision": "P0-NETLIST-REV-D-20260718",
        "schema": "p0.final-netlist.v4",
        "source_off_state_table": [
            {"adg_route": "DRIVE D-to-SB", "admissible_for_science": False, "expected_ch2_code": 7, "gate": 1, "k1": 1, "k2": 1, "k3": 1, "state": "S0_DRIVE"},
            {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "expected_ch2_code": 6, "gate": 0, "k1": 1, "k2": 1, "k3": 1, "state": "S1_GATE_TERMINATED"},
            {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "allowed_ch2_codes": [0, 2, 4, 6], "gate": 0, "k1": "RELEASING", "k2": "RELEASING", "k3": 1, "state": "S2_OPEN_SERIES_BARRIERS"},
            {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "expected_ch2_code": 0, "gate": 0, "k1": 0, "k2": 0, "k3": 1, "state": "S3_SERIES_OPEN_STABLE_1000_SAMPLES"},
            {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": False, "allowed_ch2_codes": [0, 8], "gate": 0, "k1": 0, "k2": 0, "k3": "RELEASING_TO_GUARD", "state": "S4_GUARD_TRANSITION"},
            {"adg_route": "OFF D-to-SA-to-50.00-ohm", "admissible_for_science": True, "expected_ch2_code": 8, "gate": 0, "k1": 0, "k2": 0, "k3": 0, "state": "S5_STABLE_OFF_AFTER_1000_SAMPLES_AND_GUARD"},
            {"adg_route": "unpowered continuity state only", "admissible_for_science": False, "expected_ch2_code": None, "gate": 0, "k1": 0, "k2": 0, "k3": 0, "state": "S6_UNPOWERED_FAIL_SAFE"},
        ],
        "source_off_sequence": {
            "acquisition": "software-prearmed free-running continuous record; CH2 locates the event and CH0 dual-tone reconstruction supplies the phase gauge",
            "drive": f"SDG1032X C1 at calibration-bound f_carrier_hz, 0.400 Vpp continuous sine with 0 V offset and 0/pi phase; C2 at exactly 2*f_carrier_hz, {signal_model['mechanism']['selected_amplitude_vpp']:.3f} Vpp continuous fixed-zero-phase reference with 0 V offset; HIGH_Z load setting; both remain on for the whole record",
            "gate": "ADG1419 routes C1 from DRIVE to 50.00 ohm termination",
            "relay_release_delay_us": 250,
            "relay_order": "with ADG1419 already OFF, finish the 192-sample C2 pre-transfer window; deenergize K1 and K2; require 1000 consecutive samples of code 0 while K3 remains energized and evaluate the 960-sample end-to-end isolated C2 window; only after both gates pass may K3 deenergize to guard and code 8 stabilize",
            "series_open_stable_samples": 1000,
            "stable_off_samples": 1000,
            "guard_samples": 10000,
            "contact_limit_samples": 14500,
            "signal_pole_evidence_boundary": {
                "accepted_prerequisites": ["PRE_K3_COMPLEX_C2_END_TO_END_TRANSFER_WITNESS"],
                "actual_path_claim": "ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT",
                "auxiliary_contacts_sufficient": False,
                "individual_pole_identity_claim_authorized": False,
                "physical_execution_authorized": False,
                "source_disconnect_claim_authorized_only_on_future_passing_event": True,
                "topology": {"injection": "C2_REF_IN through exact TNPW 1.00 Mohm R_C2_INJECT to N_GATE_OUT; exact 100 kohm R_DRIVE_SHUNT remains on N_SRC and is isolated by ADG1419 during both H2 windows", "measured_path": ["N_GATE_OUT", "K1 signal pole", "N_MIDPOINT", "K2 signal pole", "N_ELECTRODE_A", "OPA810", "CH1"], "k3_during_h2": "energized and electrically open"},
                "witness_model": "P0_SIGNAL_PATH_CIRCUIT_MODEL.json",
            },
            "first_admissible": "max(t_gate+0.560 us, end of 1000-sample stable code-8 run)+10.000 ms",
            "source_persistence_proof": "CH0 must match the sample-level reconstruction at bound f_carrier_hz and bound f_witness_hz from t_gate through record end with peak residual <=5 percent of the fitted C1 amplitude, and both tones must remain within 2 percent amplitude and 0.010 rad phase in contiguous 100000-sample segments beginning at t_gate and covering record end; the bounded passive C2-to-C1 coupling through the monitor network is modeled and controlled rather than called zero",
            "source_setup": {
                "model": "SIGLENT SDG1032X",
                "c1": {"frequency_binding": "f_carrier_hz", "accepted_frequency_hz": [32768.0, 32820.0], "maximum_u95_hz": 0.050, "amplitude_vpp": 0.400, "offset_v": 0.0, "phase_command_rad": [0.0, math.pi]},
                "c2": {"frequency_binding": "f_witness_hz", "frequency_relation": "f_witness_hz == 2 * f_carrier_hz", "amplitude_vpp": signal_model["mechanism"]["selected_amplitude_vpp"], "offset_v": 0.0, "phase_command_rad": 0.0},
                "load_mode": "HIGH_Z",
                "physical_output_ohms": 50.0,
                "output_mode": "CONTINUOUS_SINE",
                "qualified_preparation_seconds_minimum": 3.0,
                "qualified_preparation_cycles_law": "ceil(3 * f_carrier_hz)",
                "queryback_fields_per_channel": ["model", "serial", "firmware", "load_mode", "physical_output_ohms", "waveform", "frequency_hz", "amplitude_vpp", "offset_v", "phase_command_rad", "output_mode", "output_state"],
                "complete_corner_c1_bound": {"series_limiter_ohms": 100000.0, "source_output_ohms_range": [47.5, 52.5], "carrier_motional_resistance_ohms_range": [30000.0, 70000.0], "source_vrms": 0.141421356, "motional_current_ua_max": signal_envelope["f1_motional_current_ua_rms"][1], "carrier_terminal_vpp_max": signal_envelope["f1_carrier_terminal_vpp"][1], "motional_power_uw_max": signal_envelope["f1_motional_power_uw"][1]},
            },
        },
        "status": "REVIEW_CANDIDATE__NO_HARDWARE_AUTHORITY",
        "switch_truth_table": [
            {"adg1419_pin_1_d": "N_SRC", "adg1419_pin_2_sa": "N_GATE_TERM", "adg1419_pin_8_sb": "N_GATE_OUT", "in_logic": 0, "route": "D-to-SA; C1 terminated through exact 50.00 ohm R_TERM", "state": "OFF_TERMINATE"},
            {"adg1419_pin_1_d": "N_SRC", "adg1419_pin_2_sa": "N_GATE_TERM", "adg1419_pin_8_sb": "N_GATE_OUT", "in_logic": 1, "route": "D-to-SB; C1 presented to K1", "state": "DRIVE"},
        ],
        "test_point_map": test_points,
        "witness_law": {
            "bits": {"b0": "ADuM gate-secondary output powered by ADR_REF_3V3", "b1": "K1 auxiliary contact", "b2": "K2 auxiliary contact", "b3": "K3 auxiliary contact"},
            "centroids": "all 16 measured before execution; adjacent centroids >=10 sigma; samples require unique +/-3 sigma membership",
            "equation": "V(code)=(3.3*b0/R0+sum(active relay bits 3.3/Rb))/(1/1000+1/R0+sum(active relay bits 1/Rb)); R0 is always present because b0 is actively driven high or low; code=b0+2*b1+4*b2+8*b3",
            "minimum_nominal_adjacent_gap_mv": round(minimum_gap_mv, 6),
            "nominal_centroids_mv": nominal_centroids_mv,
            "nominal_reference_v": 3.3,
            "pulldown_ohms": 1000.0,
            "stable_off_code": 8,
            "weight_resistors_ohms": {"b0": 80600.0, "b1": 40200.0, "b2": 20000.0, "b3": 10000.0},
            "qualification": "K1/K2/K3 auxiliary contacts do not substitute for their signal-pole continuity or actual-path controls; the gate branch's low-driven loading is included in every nominal and fitted centroid",
        },
    }


PACKAGE_BY_PART = {
    "ADG1419BRMZ": "8-lead MSOP", "G6K-2F-Y DC5": "surface-mount DPDT",
    "OPA810IDT": "8-lead SOIC D", "ADXL354CEZ": "14-lead LCC",
    "SHT45-AD1B-R2": "DFN4", "ADuM140D0BRZ": "16-lead SOIC-N",
    "ADR4533BRZ": "8-lead SOIC", "MEV1D0505SC": "SIP",
    "NUCLEO-G031K8": "Nucleo-32", "Q13FC1350000401": "FC-135",
    "GJM1555C1H1R0BB01D": "0402", "2N7002PW,115": "SOT-323",
    "1N4148W,115": "SOD-123", "BZT52H-C12,115": "SOD-123F",
    "0467.500NR": "0603 fast-acting fuse",
}


def document_hash(document_id: str) -> str:
    _, custody = research_metadata()
    record = next((item for item in custody["records"] if item["source_id"] == document_id), None)
    if record is None:
        return f"PROSPECTIVE_IDENTITY_ONLY__{document_id}"
    if record["current_sha256"] is not None:
        return record["current_sha256"]
    if record["legacy_expected_sha256"] is not None:
        return f"LEGACY_EXPECTED_SHA256__BYTES_NOT_LOCAL__{record['legacy_expected_sha256']}"
    return f"{record['custody_state']}__{document_id}"


def build_bom(netlist: dict[str, Any]) -> dict[str, Any]:
    model = read_canonical_json(ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in netlist["components"]:
        key = (item["manufacturer"], item["part_number"], item["exact_document"])
        grouped.setdefault(key, []).append(item)
    items: list[dict[str, Any]] = []
    for manufacturer, part, document_id in sorted(grouped):
        members = grouped[(manufacturer, part, document_id)]
        package = PACKAGE_BY_PART.get(part, "0805" if "0805" in part or part.startswith("GRM21") or part.startswith("C0805") else "exact manufacturer package")
        items.append({
            "allowed_substitution": "NONE",
            "assembly_gate": "reference, pin map, package, value and count must equal P0_FINAL_NETLIST.json",
            "category": "required_p0_build",
            "conservative_limit": "the frozen netlist and all component-specific qualification gates apply",
            "critical_value": "; ".join(sorted({member["function"] for member in members})),
            "datasheet_hash": document_hash(document_id),
            "execution_gate": "component and complete-path characterization remains separately unauthorized",
            "forbidden_substitution": "any different manufacturer, suffix, value, package, tolerance or topology",
            "function": "; ".join(sorted({member["function"] for member in members})),
            "line": len(items) + 1,
            "manufacturer": manufacturer,
            "netlist_refs": sorted(member["ref"] for member in members),
            "package": package,
            "part_number": part,
            "procurement_gate": "exact order identity, official-document byte hash, lot and packing code",
            "quantity": len(members),
        })

    additions = [
        ("existing_lab_equipment_assumption", "SIGLENT", "SDG1032X", 1, "bench instrument", f"existing-lab assumption only; ownership not asserted; SIGLENT North America marks the model obsolete while the global page still lists it; no new-procurement recommendation and SDG1032X Plus is not an automatic substitution; HIGH_Z continuous source: C1 at calibration-bound f_carrier_hz, 0.400 Vpp 0 V 0/pi; C2 at exactly 2*f_carrier_hz, {model['mechanism']['selected_amplitude_vpp']:.3f} Vpp 0 V zero phase; 50 ohm physical outputs", document_hash("SIGLENT_MANUAL")),
        ("existing_lab_equipment_assumption", "Spectrum Instrumentation", "DN2.592-04", 1, "digitizerNETBOX", "four simultaneous true-differential inputs at 1 MS/s", DOC["SPECTRUM_DATASHEET"]["sha256"]),
        ("required_p0_build", "authored", "P0-SOURCE-OFF-CONTROL-REV-B", 3, "84x64x1.6 mm four-layer PCB", "three matched control boards", "AUTHORED_BYTES_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-CARRIER-SENSE-REV-B variants A/B/C", 3, "68x32x1.0 mm two-layer PCB", "DUT, detector-only and exact-C0 variants", "AUTHORED_BYTES_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-ENV-SENSOR-REV-B", 3, "24x24x1.0 mm two-layer PCB", "three matched vibration boards", "AUTHORED_BYTES_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A", 6, "CNC-machined ASTM B209 6061-T6 bare-aluminum tub and lid; exact 124x82x36 mm minimum clear envelope, 140x98x44 mm closed outer envelope", "two enclosures per complete fixture; exact design is in P0_PCB_FABRICATION_RELEASE.json; fabricator identity and material receipt remain separately gated", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B", 3, "56x24x3 mm machined unfilled-natural-PEEK edge-retention tray", "one exact tray per Nucleo; admitted-board envelope, clip profile, holes, tolerances, insertion and retention acceptance are in the fabrication release", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-PEEK-M3-STANDOFF-REV-A", 24, "6 mm hex unfilled-natural-PEEK M3 male/female standoff; body length per mount row", "four control-board plus four carrier-board standoffs per fixture", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-PEEK-M2P5-STANDOFF-REV-A", 12, "5 mm hex unfilled-natural-PEEK M2.5 male/female standoff; 8 mm body", "four vibration-board standoffs per fixture", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-PEEK-WASHER-M3-6X3P2X0P5-REV-A", 24, "unfilled-natural-PEEK flat washer 6.0 OD x 3.2 ID x 0.5", "one under every M3 control/carrier board screw", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "authored", "P0-PEEK-WASHER-M2P5-5X2P7X0P5-REV-A", 12, "unfilled-natural-PEEK flat washer 5.0 OD x 2.7 ID x 0.5", "one under every M2.5 vibration-board screw", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "specified standard", "ISO 4762 M3x0.5x8 A2-70", 60, "stainless socket-head cap screw", "ten dry-seam lid screws per enclosure", "STANDARD_IDENTITY_BOUND"),
        ("required_p0_build", "specified standard", "ISO 4762 M3x0.5x6 A2-70", 24, "stainless socket-head cap screw", "four control-board plus four carrier-board screws per fixture", "STANDARD_IDENTITY_BOUND"),
        ("required_p0_build", "specified standard", "ISO 4762 M2.5x0.45x5 A2-70", 12, "stainless socket-head cap screw", "four vibration-board screws per fixture", "STANDARD_IDENTITY_BOUND"),
        ("required_p0_build", "authored", "P0-PEEK-CSK-M2X0P4X6-REV-A", 12, "unfilled-natural-PEEK 90-degree countersunk M2x0.4x6 screw with 4.0 mm maximum head diameter", "four flush tray-to-floor screws per fixture", "AUTHORED_MECHANICAL_IDENTITY_BOUND_BY_QUALIFICATION_ROOT"),
        ("required_p0_build", "Amphenol RF", "031-10-RFX", 18, "front-mount isolated BNC", "six isolated bulkheads per fixture", "OFFICIAL_CAPTURE_REQUIRED_BEFORE_PROCUREMENT__AMPHENOL_031_10_RFX"),
        ("required_p0_build", "Hammond", "1427CG13", 6, "nylon cable gland", "two glands per fixed harness", "OFFICIAL_CAPTURE_REQUIRED_BEFORE_PROCUREMENT__HAMMOND_1427CG13"),
        ("required_p0_build", "Pomona Electronics", "2249-C-24", 18, "24-inch BNC male-male", "six dedicated labeled coaxes per fixture", "OFFICIAL_CAPTURE_REQUIRED_BEFORE_PROCUREMENT__POMONA_2249"),
        ("required_p0_build", "Belden / Alpha Wire", "83269 plus 2840/1", 3, "fixture-local fixed wiring cut set", "150 mm inter-enclosure harness plus exact 35/45/55/60 mm internal RG-178 and PTFE pigtails from the machine-readable cable map", "OFFICIAL_CAPTURE_REQUIRED_BEFORE_PROCUREMENT__BELDEN_ALPHA"),
        ("required_p0_build", "specified standard", "ISO 7045 M4x16 brass + ISO 4032 brass nut + DIN 6798A brass star washer", 3, "insulated star set", "one analog star per carrier enclosure", "STANDARD_IDENTITY_BOUND"),
    ]
    for category, manufacturer, part, quantity, package, function, doc_hash in additions:
        items.append({
            "allowed_substitution": "NONE", "assembly_gate": "identity and coordinates must match the fabrication release",
            "category": category, "conservative_limit": function, "critical_value": function,
            "datasheet_hash": doc_hash, "execution_gate": "future separately authorized qualification only",
            "forbidden_substitution": "any changed identity, dimension, topology or quantity", "function": function,
            "line": len(items) + 1, "manufacturer": manufacturer, "netlist_refs": [], "package": package,
            "part_number": part, "procurement_gate": "separate user authority plus exact identity receipt", "quantity": quantity,
        })
    inventory = [{"manufacturer": item["manufacturer"], "part_number": item["part_number"], "quantity": item["quantity"]} for item in items]
    return {
        "authority": AUTHORITY,
        "categories": ["required_p0_build", "existing_lab_equipment_assumption", "optional_calibration_reference", "fallback_only", "silicon_future"],
        "electrical_component_count": len(netlist["components"]),
        "inventory": "prospective exact identities; no stock was inspected or reserved",
        "inventory_sha256": sha256_bytes(canonical(inventory)),
        "items": items,
        "price": "not requested or frozen",
        "schema": "p0.nonpurchasing-bom.v2",
        "supplier": "not selected or contacted",
    }


def board_placements(netlist: dict[str, Any], board_id: str, width: float, height: float) -> list[dict[str, Any]]:
    pilot = [item for item in netlist["components"] if item["board"] == board_id and (item["ref"].startswith("R_C2_INJECT_") or item["ref"].startswith("R_DRIVE_SHUNT_"))]
    members = sorted((item for item in netlist["components"] if item["board"] == board_id and item not in pilot), key=lambda item: item["ref"])
    columns = 10 if width >= 80 else 7 if width >= 60 else 3
    x_step = (width - 12.0) / max(columns - 1, 1)
    rows = max(1, math.ceil(len(members) / columns))
    y_step = (height - 12.0) / max(rows - 1, 1)
    placements = [
        {
            "ref": item["ref"], "x_mm": round(6.0 + (index % columns) * x_step, 3),
            "y_mm": round(6.0 + (index // columns) * y_step, 3), "rotation_deg": 0, "side": "top",
        }
        for index, item in enumerate(members)
    ]
    for index, item in enumerate(sorted(pilot, key=lambda value: value["ref"])):
        placements.append({"ref": item["ref"], "x_mm": 54.0 + 4.0 * (index % 3), "y_mm": 56.0 + 3.0 * (index // 3), "rotation_deg": 0, "side": "top"})
    return sorted(placements, key=lambda item: item["ref"])


def build_fabrication(netlist: dict[str, Any]) -> dict[str, Any]:
    releases: list[dict[str, Any]] = []
    for board in netlist["boards"]:
        if board["board_id"].startswith("NUCLEO"):
            continue
        width, height, thickness = board["outline_mm"]
        inset = 4.0 if width >= 60 else 2.5
        releases.append({
            "board_id": board["board_id"],
            "datum": "lower-left board corner, component side up",
            "drill": {"finished_mount_hole_mm": 3.2 if width >= 60 else 2.7, "plated": False, "tolerance_mm": 0.08},
            "finish": "ENIG; IPC-6012 Class 2; no-clean flux prohibited on guarded carrier board",
            "mount_holes_mm": [[inset, inset], [width - inset, inset], [inset, height - inset], [width - inset, height - inset]],
            "outline_mm": [width, height, thickness],
            "placements": board_placements(netlist, board["board_id"], width, height),
            "stackup": "four-layer FR-4 1.6 mm, L2 solid domain-specific return, L3 power" if thickness == 1.6 else "two-layer high-Tg FR-4, 1.0 mm; guarded input copper keepout on both layers",
            "tolerances_mm": {"outline": 0.15, "placement": 0.10, "thickness": 0.10},
        })
    enclosure_design = {
        "schema": "p0.custom-enclosure-design.v1",
        "design_id": "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A",
        "datum": "lower-left corner of the unobstructed inner floor envelope; +x east, +y north, +z toward lid",
        "material": "ASTM B209 6061-T6 aluminum plate or billet with lot and mill-certificate custody",
        "construction": "CNC-machined one-piece tub plus removable flat lid",
        "clear_internal_min_mm": [124.0, 82.0, 36.0],
        "body_outer_mm": [140.0, 98.0, 41.0],
        "lid_outline_mm": [140.0, 98.0, 3.0],
        "closed_outer_mm": [140.0, 98.0, 44.0],
        "wall_thickness_mm": 8.0,
        "floor_thickness_mm": 5.0,
        "lid_thickness_mm": 3.0,
        "finish": "bare machined aluminum, deburred and solvent-cleaned; no anodize, paint, conversion coating, gasket, or insulating film on body, lid, fastener seats, or seam",
        "panel_face_axes": {"east": ["z", "y"], "floor": ["x", "y"], "north": ["x", "z"], "south": ["x", "z"], "west": ["z", "y"]},
        "panel_face_datums": {
            "west": {"origin_inner_xyz_mm": [0.0, 0.0, 0.0], "coordinate_1": "+z", "coordinate_2": "+y"},
            "east": {"origin_inner_xyz_mm": [124.0, 0.0, 0.0], "coordinate_1": "+z", "coordinate_2": "+y"},
            "south": {"origin_inner_xyz_mm": [0.0, 0.0, 0.0], "coordinate_1": "+x", "coordinate_2": "+z"},
            "north": {"origin_inner_xyz_mm": [0.0, 82.0, 0.0], "coordinate_1": "+x", "coordinate_2": "+z"},
            "floor": {"origin_inner_xyz_mm": [0.0, 0.0, 0.0], "coordinate_1": "+x", "coordinate_2": "+y"},
        },
        "panel_land": "every vertical-panel through-hole is exterior-counterbored 5.0 mm deep, leaving an exact 3.0 mm inner throat",
        "lid_seam": "continuous bare-metal 8.0 mm perimeter land; dry assembly with no gasket",
        "lid_fasteners": {
            "part": "ISO 4762 M3x0.5x8 A2-70 stainless socket-head cap screw",
            "body_thread": "M3x0.5-6H blind tapped 5.0 mm minimum full thread depth",
            "lid_clearance_hole_mm": 3.4,
            "positions_outer_xy_mm": [[35.0, 4.0], [70.0, 4.0], [105.0, 4.0], [136.0, 32.0], [136.0, 66.0], [35.0, 94.0], [70.0, 94.0], [105.0, 94.0], [4.0, 32.0], [4.0, 66.0]],
            "installation_torque_nm": 0.45,
            "installation_torque_tolerance_nm": 0.05,
        },
        "retention_systems": [
            {
                "id": "P0-PEEK-M3-STANDOFF-REV-A",
                "applies_to_finished_hole_mm": 3.2,
                "material": "unfilled natural PEEK conforming to ASTM D6262 S-PAEK",
                "geometry": "6.0 mm hex body; M3x0.5 male floor stud 4.0 mm; M3x0.5 female top thread 6.0 mm; body length equals mount standoff_mm",
                "floor_interface": "M3x0.5 blind tapped 4.0 mm into the 5.0 mm floor, leaving at least 1.0 mm exterior floor",
                "board_fastener": "ISO 4762 M3x0.5x6 A2-70 plus 0.5 mm PEEK flat washer",
            },
            {
                "id": "P0-PEEK-M2P5-STANDOFF-REV-A",
                "applies_to_finished_hole_mm": 2.7,
                "material": "unfilled natural PEEK conforming to ASTM D6262 S-PAEK",
                "geometry": "5.0 mm hex body; M2.5x0.45 male floor stud 4.0 mm; M2.5x0.45 female top thread 5.0 mm; body length equals mount standoff_mm",
                "floor_interface": "M2.5x0.45 blind tapped 4.0 mm into the 5.0 mm floor, leaving at least 1.0 mm exterior floor",
                "board_fastener": "ISO 4762 M2.5x0.45x5 A2-70 plus 0.5 mm PEEK flat washer",
            },
            {
                "id": "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B",
                "material": "unfilled natural PEEK conforming to ASTM D6262 S-PAEK",
                "tray_outer_mm": [56.0, 24.0, 3.0],
                "tray_origin_from_board_origin_mm": [-3.0, -3.0, -3.0],
                "board_admission_outline_mm": {"nominal": [50.0, 18.0], "measured_min": [49.95, 17.95], "measured_max": [50.05, 18.05], "law": "measure the bare PCB edge outline before insertion; reject any board outside both closed intervals"},
                "board_seat": {"datum": "tray lower-left top face with components toward +z", "lower_left_tray_xy_mm": [3.0, 3.0], "envelope_mm": [50.2, 18.2, 1.8], "cut": "no pocket; flat continuous seat except flush PEEK anchor heads"},
                "integral_clip_contact_rectangles_board_xy_mm": [[0.0, 0.0, 1.0, 4.0], [49.0, 0.0, 1.0, 4.0], [0.0, 14.0, 1.0, 4.0], [49.0, 14.0, 1.0, 4.0]],
                "clip_profile": {"post_height_above_tray_mm": 2.4, "post_thickness_mm": 1.0, "post_width_along_perimeter_mm": 4.0, "overhang_inward_mm": 0.4, "overhang_thickness_mm": 0.8, "overhang_underside_above_tray_mm": 1.6, "lead_in_ramp_angle_deg": 30.0, "lead_in_ramp_vertical_height_mm": 0.8, "base_relief_radius_mm": 0.3, "tip_chamfer_mm": 0.2},
                "insertion_clearance_budget_mm": {"seat_span_min": [50.1, 18.1], "admitted_board_span_max": [50.05, 18.05], "per_side_clearance_min": [0.025, 0.025], "overhang_inward_max": 0.45, "outward_clip_deflection_required_max": 0.425, "allowed_outward_clip_deflection_max": 0.5},
                "anchor_holes_tray_xy_mm": [[2.5, 2.5], [53.5, 2.5], [2.5, 21.5], [53.5, 21.5]],
                "anchor_hole": {"through_diameter_mm": 2.2, "countersink_major_diameter_mm": 4.2, "countersink_angle_deg": 90.0, "head_flush_mm_max": 0.0, "head_recess_mm_max": 0.05},
                "floor_fastener": "P0-PEEK-CSK-M2X0P4X6-REV-A, unfilled natural PEEK, 90 degree countersunk head 4.0 mm maximum diameter; M2x0.4 male thread; four per tray into M2x0.4-6H blind floor holes 4.0 mm deep with at least 3.0 mm engagement and 1.0 mm exterior floor remaining",
                "assembly_method": "with power and all external cables absent, align the board over all four integral lead-in ramps and apply no more than 10.0 N total downward force distributed only across the outermost 1.0 mm PCB perimeter until all clips seat; the ramps may deflect each clip outward no more than 0.50 mm; no separate jig, pry tool, header, connector, component, pad, or test point may carry insertion force",
                "keepout": "clips contact only the outermost 1.0 mm board perimeter and do not cover connectors, headers, components, pads, test points, or copper beyond that perimeter",
                "tolerances_mm": {"tray_outer": 0.10, "tray_thickness": 0.05, "board_seat_position": 0.10, "board_seat_envelope": 0.10, "anchor_hole_diameter": 0.05, "anchor_hole_position": 0.10, "countersink_major_diameter": 0.10, "clip_post_height": 0.05, "clip_post_thickness": 0.05, "clip_overhang": 0.05, "clip_underside_height": 0.05, "relief_and_chamfer": 0.05},
                "retention_acceptance": ["the measured bare-PCB edge outline passes board_admission_outline_mm before insertion", "all four PEEK anchor heads are flush to 0.05 mm below the seat and no metal is exposed beneath the board", "board insertion requires no more than 0.50 mm outward clip deflection and leaves no crack, craze, chip, permanent set, or board witness mark", "after insertion the board moves no more than 0.20 mm laterally under 2.0 N in either tray axis", "the retained board withstands 5.0 N upward for 10 seconds without release, damage, connector/header contact, or permanent displacement", "all four contact rectangles remain within the admitted board outline and every clip contact and keepout condition passes visual inspection at 10x"],
            },
        ],
        "tolerances_mm": {"body_outer": 0.20, "clear_envelope_minimum": 0.0, "counterbore_depth": 0.10, "floor_and_wall": 0.10, "hole_diameter": 0.10, "hole_position": 0.10, "lid": 0.10, "mount_origin": 0.10},
        "retention_tolerances_mm": {"standoff_body_length": 0.10, "standoff_hex_across_flats": 0.10, "male_stud_length": 0.10, "female_thread_depth": 0.10, "washer_thickness": 0.05, "washer_outer_diameter": 0.10, "floor_blind_thread_depth": 0.10},
        "electrical_acceptance": {"analog_star_to_body_mohm_min": 1000.0, "body_to_lid_dc_ohm_max": 0.05, "isolated_bulkhead_to_body_mohm_min": 1000.0},
        "mechanical_acceptance": [
            "material, heat treatment, lot, mill certificate, body identity, and lid identity match the reviewed receipt",
            "the full 124.0x82.0x36.0 mm inner gauge envelope enters without interference at every mount and fastener location",
            "body, lid, wall, floor, panel throat, holes, counterbores, blind threads, mount origins, and retention parts pass the stated tolerances",
            "all edges are deburred 0.2 to 0.5 mm and every machined surface is clean and free of conductive debris",
            "all ten lid screws are present at 0.45 plus or minus 0.05 N m and the dry perimeter seam has no visible gap",
            "body-to-lid continuity and all isolated bulkhead/star-to-body resistance checks meet the electrical acceptance thresholds",
            "every PEEK standoff, washer, board screw, tray, tray screw, blind floor thread, and lid screw matches the exact non-purchasing BOM identity and quantity",
            "all three Nucleo trays pass every tray-specific dimensional, insertion, retention, keepout, flush-head, and visual acceptance requirement",
        ],
    }
    enclosures = []
    for suffix, assembly, _, _ in ASSEMBLIES:
        enclosures.extend([
            {
                "assembly": assembly, "enclosure": f"ENC-CONTROL-{suffix}", "part_number": "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A",
                "minimum_clear_envelope_mm": [124.0, 82.0, 36.0],
                "mounts": [{"object": f"P0-SOURCE-OFF-CONTROL-REV-B-{suffix}", "origin_mm": [4.0, 9.0, 6.0], "retention_id": "P0-PEEK-M3-STANDOFF-REV-A", "rotation_deg": 0, "standoff_mm": 6.0}, {"object": f"NUCLEO-G031K8-{suffix}", "origin_mm": [117.0, 16.0, 3.0], "origin_datum": "board-local lower-left underside [0,0,0] at the tray seat", "retention_id": "P0-NUCLEO-PEEK-EDGE-TRAY-REV-B", "rotation_deg": 90, "rotation_axis": "machine +z through origin_mm", "rotation_convention": "counterclockwise from machine +x toward machine +y when viewed from +z toward the floor", "standoff_mm": 3.0, "local_to_machine_transform": {"board_local_xyz_to_machine_xyz": "[x_m,y_m,z_m]=[117-y_b,16+x_b,3+z_b]", "tray_local_xyz_to_machine_xyz": "[x_m,y_m,z_m]=[120-y_t,13+x_t,z_t]", "tray_origin_machine_mm": [120.0, 13.0, 0.0], "tray_axes_machine": "+x_tray=+y_machine; +y_tray=-x_machine; +z_tray=+z_machine", "tray_footprint_machine_xy_closed_mm": [[96.0, 13.0], [120.0, 69.0]], "floor_anchor_centers_machine_xy_mm": [[117.5, 15.5], [117.5, 66.5], [98.5, 15.5], [98.5, 66.5]]}}],
                "panel_holes": [
                    {"id": f"J_SRC_C1_{suffix}", "face": "west", "center_mm": [18.0, 19.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"J_SRC_C2_{suffix}", "face": "west", "center_mm": [18.0, 45.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"J_CH0_{suffix}", "face": "east", "center_mm": [18.0, 19.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"J_CH2_{suffix}", "face": "east", "center_mm": [18.0, 45.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"GLAND-C-{suffix}", "face": "north", "center_mm": [47.0, 20.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 28.0, "diameter_mm": 20.42},
                ],
            },
            {
                "assembly": assembly, "enclosure": f"ENC-CARRIER-{suffix}", "part_number": "P0-CUSTOM-ENCLOSURE-124X82X36-REV-A",
                "minimum_clear_envelope_mm": [124.0, 82.0, 36.0],
                "mounts": [{"object": next(board["board_id"] for board in netlist["boards"] if board["assembly"] == assembly and board["board_id"].startswith("P0-CARRIER")), "origin_mm": [4.0, 25.0, 8.0], "retention_id": "P0-PEEK-M3-STANDOFF-REV-A", "rotation_deg": 0, "standoff_mm": 8.0}, {"object": f"P0-ENV-SENSOR-REV-B-{suffix}", "origin_mm": [76.0, 29.0, 8.0], "retention_id": "P0-PEEK-M2P5-STANDOFF-REV-A", "rotation_deg": 0, "standoff_mm": 8.0}],
                "panel_holes": [
                    {"id": f"J_CH1_{suffix}", "face": "east", "center_mm": [18.0, 19.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"J_CH3_{suffix}", "face": "east", "center_mm": [18.0, 45.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 20.0, "diameter_mm": 12.70},
                    {"id": f"GLAND-R-{suffix}", "face": "south", "center_mm": [47.0, 20.0], "counterbore_depth_mm": 5.0, "counterbore_diameter_mm": 28.0, "diameter_mm": 20.42},
                    {"id": f"AGND-STAR-{suffix}", "face": "floor", "center_mm": [94.0, 68.0], "diameter_mm": 4.20, "insulated_from_enclosure": True},
                ],
            },
        ])
    return {
        "authority": AUTHORITY,
        "boards": releases,
        "enclosure_design": enclosure_design,
        "enclosures": enclosures,
        "fabrication_authority": "NONE; coordinates are a reviewed prospective release only",
        "harness": {"finished_length_mm": 150.0, "length_tolerance_mm": 2.0, "gland_hole_mm": 20.42, "gland_hole_tolerance_mm": 0.10, "no_connector_or_splice": True},
        "mechanical_clearance": {"board_to_board_mm_min": 4.0, "board_to_lid_mm_min": 5.0, "board_to_wall_mm_min": 4.0, "coax_bend_radius_mm_min": 15.0, "gland_to_board_mm_min": 8.0},
        "schema": "p0.pcb-fabrication-release.v1",
        "status": "REVIEW_CANDIDATE__NOT_AUTHORIZED_FOR_FABRICATION",
    }


def build_component_documents() -> dict[str, Any]:
    manifest, custody = research_metadata()
    manifest_bytes = (RESEARCH_ROOT / "MANIFEST.json").read_bytes()
    custody_bytes = (RESEARCH_ROOT / "SOURCE_CUSTODY.json").read_bytes()
    netlist_required = {item["exact_document"] for item in build_netlist()["components"]}
    required = {item["source_id"] for item in custody["records"] if item["collection"] == "core_component_document"}
    if not netlist_required.issubset(required):
        raise ValueError("research core documents do not cover every netlist identity")
    records = []
    for source in custody["records"]:
        if source["source_id"] not in required:
            continue
        records.append({
            "bytes": source["current_bytes"],
            "custody_state": source["custody_state"],
            "direct_download_url": source["direct_download_url"],
            "document_id": source["source_id"],
            "download_result": source["download_result"],
            "download_result_detail": source["download_result_detail"],
            "legacy_expected_bytes": source["legacy_expected_bytes"],
            "legacy_expected_sha256": source["legacy_expected_sha256"],
            "license_or_redistribution_note": source["license_or_redistribution_note"],
            "official_url": source["official_product_page"],
            "publisher": source["publisher"],
            "relevance_to_p0": source["relevance_to_p0"],
            "revision": source["revision_and_date"],
            "sha256": source["current_sha256"],
            "title": source["title"],
        })
    if {record["document_id"] for record in records} != required:
        raise ValueError("research custody does not cover every netlist document identity")
    return {
        "authority": AUTHORITY,
        "records": sorted(records, key=lambda item: item["document_id"]),
        "research_bundle": {
            "canonical_relative_path": RESEARCH_RELATIVE_PATH,
            "custody_snapshot_sha256": sha256_bytes(custody_bytes),
            "custody_state_counts": custody["custody_state_counts"],
            "manifest_schema": manifest["schema"],
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "source_commit": RESEARCH_SOURCE_COMMIT,
            "source_record_count": manifest["record_count"],
            "third_party_byte_policy": custody["third_party_byte_policy"],
        },
        "schema": "p0.component-documents.v2",
        "automated_public_source_requests_recorded": True,
        "zero_human_vendor_outreach": True,
    }


def build_packet(netlist: dict[str, Any], bom: dict[str, Any], fabrication: dict[str, Any]) -> str:
    net_hash = sha256_bytes(canonical(netlist))
    bom_hash = sha256_bytes(canonical(bom))
    fab_hash = sha256_bytes(canonical(fabrication))
    model_bytes = (ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json").read_bytes()
    model = read_canonical_json(ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json")
    model_hash = sha256_bytes(model_bytes)
    selected = model["selected_envelope"]["envelope"]
    research_manifest, research_custody = research_metadata()
    research_manifest_hash = sha256_bytes((RESEARCH_ROOT / "MANIFEST.json").read_bytes())
    custody_counts = research_custody["custody_state_counts"]
    return f"""# P0 non-executing build-readiness packet

## Frozen scope

- authority: `{AUTHORITY}`
- claim ceiling: `{CEILING}`
- next authority boundary: `{NEXT}`
- status: `P0_BUILD_READINESS_PACKET_FROZEN`; the prospective actual-path witness repair is mechanically qualified, but no procurement, fabrication, assembly, connection, power, playback, acquisition, or instrument command is authorized
- exact design hashes: netlist `{net_hash}`, BOM `{bom_hash}`, fabrication `{fab_hash}`, signal-path model `{model_hash}`

This packet describes a prospective physical experiment. It reports no physical result. No human vendor outreach, cart, stock, procurement, instrument command, audio interface, target, or hardware contact occurred while producing it. Automated public-source HTTP retrieval attempts are separately disclosed in the research custody snapshot and private ignored download receipt; they are not described as zero network or zero server contact.

## Canonical research custody dependency

The repository-safe research dependency is `{RESEARCH_RELATIVE_PATH}`, imported from commit `{RESEARCH_SOURCE_COMMIT}`. Its canonical `{research_manifest['schema']}` manifest contains exactly {research_manifest['record_count']} records and has SHA-256 `{research_manifest_hash}`. Current repository-safe custody metadata records {custody_counts.get('LOCAL_BYTES_CAPTURED_AND_HASH_VERIFIED', 0)} locally hash-verified private-cache captures, {custody_counts.get('LOCAL_CURRENT_BYTES_CAPTURED__LEGACY_DIFFERS', 0)} current captures that differ from retained legacy hashes, {custody_counts.get('URL_AND_LEGACY_HASH_RECORDED__BYTES_NOT_LOCAL', 0)} URL-plus-legacy-hash records without local bytes, {custody_counts.get('MANUAL_CAPTURE_REQUIRED', 0)} manual captures, and {custody_counts.get('PROSPECTIVE_IDENTITY_ONLY', 0)} prospective identities. These counts describe the bound metadata snapshot; downloaded third-party document/model bytes, receipts, HTML captures and generated archives remain ignored, private and uncommitted.

Private refresh command, from the research-bundle directory, is `D:\\CCC 2.0\\AI\\agent-governance-system\\.venv\\Scripts\\python.exe scripts/download_sources.py --all`, followed by the same interpreter with `scripts/verify_downloads.py` and `scripts/build_custody_snapshot.py`. A URL or historical hash alone is never described as local byte custody. Revision alerts retain ADR45xx Rev. G rather than Rev. F, ADuM140D Rev. K, ST UM2591 Rev. 2 (April 2026), SHT4x PDF Version 7.1 (March 2025) alongside the product-page 04/2025 label, SIGLENT and Spectrum current document listings without discarding historical hashes, the `2N7002PW,115` Not-for-Design-In warning, and unresolved current first-party custody for `1N4148W,115`. Replacement candidates remain research alternatives only and do not silently mutate this frozen design.

## What is being tested

The bounded question is whether the phase of a mechanically stored 32.768 kHz quartz state remains distinguishable after the source path is physically isolated. The software analyzer is an ordinary deterministic reference and cannot establish physical persistence. The maximum later claim remains a bounded physical-carrier observation under the separately frozen execution contract.

## Three complete matched assemblies

`P0-DUT-A` contains Epson `Q13FC1350000401`; suffix `01` is the manufacturer's any-quantity tape-cut packing code and matches the one-piece build quantity. `P0-DETECTOR-B` has the carrier position deliberately open; `P0-DUMMY-C0-C` substitutes exactly one Murata `GJM1555C1H1R0BB01D` 1.0 pF C0G part. Each fixture owns its control board, carrier board, sensor board, controller, two enclosures, fixed harness, and six labeled coaxes. Only the source and digitizer are shared, and only one complete assembly may be connected in a future record.

## Sense admittance and the OPA810 correction

The selected sense part is Texas Instruments `OPA810IDT`, official `SBOS799E`, SHA-256 `{DOC['OPA810']['sha256']}`. The data sheet gives 12 Gohm in parallel with 2 pF common-mode input and 0.5 pF differential input as typical values. Those are planning values, not guarantees. Across the accepted 32,768..32,820 Hz calibration interval, 4.00 pF has at least 1.212 Mohm reactance; the FC-135 70 kohm maximum ESR remains below 0.058 of that value. The prospective budget is 2.50 pF typical amplifier + 0.30 pF guarded land/routing + 0.20 pF K2/carrier land + 0.15 pF bias body/pads + 0.15 pF contamination + 0.30 pF reserve = 3.60 pF, leaving 0.40 pF margin. Every future populated coupon must separately prove `Cin,U95 <= 4.00 pF` and `Rin,U95 >= 100 Mohm`; otherwise assembly stops. External electrode clamps are deliberately absent because their capacitance or leakage would change the carrier load. Normal drive is bounded upstream by the 100 kohm limiter and two relay barriers; OPA810 absolute input/current and on-die ESD ratings are damage limits, not operating permission. Handling therefore requires grounded ESD controls with every source, instrument and external cable absent.

## Continuous dual-tone source and phase gauge

The assumed source is one SIGLENT `SDG1032X` in `HIGH_Z` load mode; its physical output impedance remains 50 ohm. After the separately authorized calibration closes, C1 is a continuous sine at bound `f_carrier_hz`, exactly 0.400 Vpp and 0 V offset with phase command 0 or pi. C2 is a continuous sine at bound `f_witness_hz = 2 * f_carrier_hz`, exactly {model['mechanism']['selected_amplitude_vpp']:.3f} Vpp and 0 V offset with fixed zero phase. Both outputs share the source's internal reference; no burst and no external trigger is used. The C1 passive monitor tap is at `C1_IN`, upstream of the exact 100 kohm limiter and ADG1419, so routing the downstream node to 50 ohm cannot erase the source witness. C1 and C2 are each passively presented through 100 kohm to CH0, with 1 Mohm return, so CH0 carries a source-continuity witness and a 2x phase gauge. C2 is not an intended carrier drive, but the passive monitor network creates a bounded linear C2-to-C1 coupling path through `R_MON_C2`, the monitor node, `R_MON_C1`, the finite C1 source impedance and `R_LIMIT`; that path is included in the circuit model, feedthrough sweep and empty/dummy controls rather than asserted to be zero. Its fixture-end BNC shell has no internal conductor, preventing a second low-impedance source-return bond.

The corrected complete-corner BVD/load model bounds the C1 carrier terminal voltage to {selected['f1_carrier_terminal_vpp'][0]:.6f}..{selected['f1_carrier_terminal_vpp'][1]:.6f} Vpp, motional current to {selected['f1_motional_current_ua_rms'][0]:.6f}..{selected['f1_motional_current_ua_rms'][1]:.6f} microampere rms, and motional dissipation to {selected['f1_motional_power_uw'][0]:.6f}..{selected['f1_motional_power_uw'][1]:.6f} microwatt. These are prospective model bounds only; a future as-built path must meet the separately frozen terminal-voltage, current and power caps by calibrated measurement.

The requested first 1.00 Mohm/0.100 Vpp witness point survives the complete corrected model. One Vishay `TNPW08051M00BEEN` per channel connects `C2_REF_IN` to `N_GATE_OUT`, downstream of ADG1419 and upstream of K1. One exact 100 kohm `TNPW0805100KBEEN` shunts `N_SRC` to analog ground, holding the inherited carrier-terminal ceiling below 0.200 Vpp while remaining on the ADG D/SA side after OFF; it therefore cannot erase the C2 witness on SB/N_GATE_OUT. C2 remains visible on CH0 and independently traverses `N_GATE_OUT -> K1 -> N_MIDPOINT -> K2 -> N_ELECTRODE_A -> OPA810 -> CH1`. The complete {model['corner_count_per_candidate']}-binary-corner sweep bounds pre-open `|H2|` to {selected['pre_abs_h2'][0]:.6f}..{selected['pre_abs_h2'][1]:.6f}, every legitimate one- or both-contact-open `|H2|` to {selected['isolated_abs_h2'][0]:.6f}..{selected['isolated_abs_h2'][1]:.6f}, f1 terminal perturbation to at most {selected['pilot_f1_fractional_perturbation'][1]:.6f}, f1 carrier terminal voltage to at most {selected['f1_carrier_terminal_vpp'][1]:.6f} Vpp, and f2 carrier terminal voltage to at most {selected['f2_carrier_terminal_vpp'][1]:.6f} Vpp. Separate complete binary-corner sweeps are frozen for DUT, detector-only, exact-1-pF dummy, wrong-node, and K3-guarded populations. They are discrete prospective model sweeps, not continuous uncertainty envelopes and not physical observations.

## Calibration-derived resonance route

`P0_RESONANCE_LOAD_SANITY_MODEL.json` freezes the prospective calibration law without authorizing execution. A future separately authorized calibration searches 32,760..32,820 Hz at 1 Hz, then +/-1 Hz around the coarse maximum at 0.025 Hz, using one 0.100 Vpp pass and a bounded complex-BVD magnitude/phase fit. It accepts only `f_carrier_hz` in 32,768..32,820 Hz, `f_carrier_u95_hz <= 0.050 Hz`, and `Q` in 4,000..60,000. The calibration artifact must bind the exact assembly and carrier population, raw-data hash, analyzer hash, selected frequency and U95, Q and decay, source and instrument queryback hashes, completion time, and `primary_observed=false`. Calibration must complete before the assignment commitment is created. The source must then prepare for at least 3.000 s before primary acquisition, and `f_witness_hz` must equal exactly `2 * f_carrier_hz`.

The analyzer derives every operational tone from that bound tuple: source queryback, drive/reference fit, C2 transfer, I/Q projection, reconstruction, cycle counting, off-resonance controls, and matched comparison. It strictly parses the bound coarse/fine complex calibration sweep, recomputes its selected frequency, Q, decay, and uncertainty consistency, and evaluates a per-role raw off-resonance response at exactly `f_carrier + max(20 Hz, 20 calibrated linewidths)`. Missing frequency custody, altered or self-consistently rebound invalid calibration bytes, preparation before calibration/assignment, late calibration, source-queryback mismatch, a broken 2:1 witness relation, or an unbound off-resonance probe is a hard rejection. The prospective BVD ring-up/ringdown sanity model is a planning check only; it neither selects a physical frequency nor authorizes a powered action.

For a future record the source must be queried back for model, serial, firmware, output impedance, waveform, frequency, amplitude, offset, phase, continuous mode and output state on both channels. The record metadata also binds the sealed one-standard-uncertainty fields `phase_skew_standard_uncertainty_rad` and `phase_drive_cal_standard_uncertainty_rad`; the analyzer combines them with lag-7 Newey-West drive-fit covariance for the individual and matched-arm phase gates. Any mismatch stops before accepting bytes.

Topology custody is assembly- and event-specific. Each record carries its own canonical assembly manifest, four-state topology scan, C1-only nonlinear-control trace and version-2 topology receipt. That receipt binds the exact assembly identifier, carrier population, assembly-manifest hash, assigned role, native payload hash, chronology receipt, source and instrument querybacks, scan bytes, nonlinear-control bytes and scan-start/scan-completion times. All four scan/acquisition times use exact `YYYY-MM-DDTHH:MM:SS.ffffffZ` UTC form, are parsed as instants, and must establish scan start < scan completion <= acquisition start < acquisition completion. The three A roles must share the exact A manifest; B and C must use distinct manifests and their exact open-carrier and 1 pF populations. Every event's scan bytes, nonlinear-control bytes and topology receipt must be unique. Replaying A evidence into B or C, replaying a receipt or nonlinear-control trace across events, duplicating scan bytes, using noncanonical time text, or moving the scan after acquisition is a hard rejection.

## Software-prearmed acquisition

The assumed Spectrum `DN2.592-04` uses four simultaneous true-differential ADC paths at exactly 1,000,000 samples/s and 3,101,000 signed 16-bit samples per channel. It free-runs after software pre-arm. CH0 is the dual-tone source monitor, CH1 the OPA810 sense output, CH2 the four-bit source-off witness and CH3 the ADXL354 Z axis. There is no trigger cable. CH2 locates the gate event; half the fitted C2 phase supplies the continuous phase gauge, with the pi branch resolved by the commanded C1 arm.

The digitizer inputs are true differential but not galvanically isolated. Their positive-to-ground, negative-to-ground and differential admittances must be surveyed across 32.768 and 65.536 kHz and included in calibration. No document calls them an isolation barrier.

No proprietary DN2 container parser is claimed. The frozen acquisition boundary is an exact SDK export mode producing one headerless, sample-major, little-endian signed-int16 payload with four channels and exactly 3,101,000 samples per channel (24,808,000 bytes), plus canonical strict JSON metadata. The adapter is an export configuration and receipt, not a guessed parser: it binds SDK/driver identity, adapter source SHA-256, SDK-export hash/count, the identical analyzer-payload hash/count, and explicit false assertions for sample loss, reordering, averaging, filtering, resampling, clipping concealment and unit ambiguity. Any additional proprietary container is preserved separately but never parsed by this analyzer. The reference fixtures exercise the canonical export contract. Actual SDK/export bytes and adapter source cannot be frozen before acquisition-equipment authority, so their exact identities remain a procurement-time gate; physical `analyze` must reject until they are supplied and hash-bound.

## Physical source-off sequence

At the decoded gate event, ADG1419 routes C1 away from the drive path to an exact 50.00 ohm termination while C1 and C2 remain continuously energized. With K1/K2 still closed and K3 energized/open, the analyzer completes the frozen 192-sample C2 pre-transfer window from gate offsets 48 through 239. After the fixed 250 microseconds K1 and K2 release while K3 remains energized and electrically open. CH2 must decode code 0 for 1,000 consecutive samples; within that same run the analyzer evaluates the 960-sample C2 isolated-path window at offsets 20 through 979. It requires the prospectively frozen complex-transfer, uncertainty, separation and drop gates before K3 may deenergize to guard. CH2 must then decode code 8 for 1,000 consecutive samples and the ordinary 10 ms guard follows. The complete ordered transition must settle within 14,500 samples. CH0 source continuity remains mandatory. A muted source, missing or wrong-node C2 injection, guard-masked path, inverted phase, wrong ordering, late contact, or hidden trigger is a hard rejection.

CH2 still observes auxiliary poles and never proves either individual signal contact. The additional C2 transfer measures the complete actual source-to-carrier path before K3 can mask it. A passing future event supports only `ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT`, meaning at least one of the two series contacts interrupted the path. It never identifies which pole opened, never claims both opened, and supplies no physical observation under current authority.

The ADuM gate-secondary logic supply and the three relay witness contacts use the same exact `ADR_REF_3V3` rail. The gate bit is actively driven low when inactive, so its 80.6 kohm branch remains a shunt in every b0=0 code; inactive relay-contact branches float. The corrected nominal 80.6/40.2/20.0/10.0 kohm ladder into 1.00 kohm has ordered code centroids from 0 to 520.543716 mV, stable-OFF code 8 at 296.654026 mV, and a minimum adjacent gap of 24.996066 mV. Those nominal values are not calibration: a future unpowered build must measure every resistor, verify low-drive impedance, recompute all sixteen prospective centroids, and a later powered calibration must establish unique ten-sigma-separated measured centroids before any record can be admissible.

## Complete wiring and access custody

`P0_FINAL_NETLIST.json` binds every relay contact, relay driver, switch truth row, source-off state, failure response, connector, external cable, inter-enclosure conductor, internal pigtail, power domain and temporary continuity access. The Nucleo-to-control link is nine fixed 60 +/-2 mm PTFE conductors. Control-panel pigtails are exact 55 +/-2 mm cuts; carrier-panel pigtails are 45 +/-2 mm; the vibration-board power/reference link is 35 +/-2 mm. The C2 fixture shell has no internal conductor. No permanent test-point hardware exists, and no electrode test point is allowed.

## Ground and shield law

C1 provides the sole intentional low-impedance source return. The inter-enclosure RG-178 shield is the sole intentional low-impedance `AGND_EXPORT` to `AGND_STAR` bond. All BNC bodies and the M4 star stud are insulated from enclosure metal. Controller ground, relay ground and analog ground remain galvanically separate. C2 has no rig-end shell connection. Because the digitizer inputs are not galvanically isolated, their positive-to-ground, negative-to-ground and differential admittances are additional calibrated parasitic return paths and must be included in the loaded model; they are never described as zero or as an isolation barrier. The exact forbidden-pair list and connector returns are in `P0_FINAL_NETLIST.json`.

## Environmental custody

Each SHT45 row contains monotonic nanoseconds, UTC, sensor serial, command `0xfd`, raw temperature and RH words, both CRC-8 bytes, and converted values. CRC uses polynomial 0x31 and initial value 0xff. UTC is linked to the monotonic clock by a frozen mapping. CH3 raw ADXL354 data must satisfy RMS <=0.050 m/s2 and peak <=0.500 m/s2, with axis, gain, offset and filter calibration bound to the record.

## Deterministic complete-circuit model boundary

`P0_SIGNAL_PATH_CIRCUIT_MODEL.json` and its generator implement the prospective C2 path from the 50-ohm source through monitor and injection networks; explicit ADG1419 SB, D and SA nodes with D-to-SA-to-50-ohm OFF routing; every K1/K2 state; K3 energized-open capacitance; the FC-135 BVD model derived from the documented 12.5 pF loaded-frequency identity with 3.4 fF inside the motional-capacitance range; OPA810 input/output/gain; the frozen 1 Mohm-parallel-30 pF digitizer mode; board/layout loading; resistor/amplifier noise; ADC quantization; f1 loading; and raw-bound nonlinear 2f and topology scans. It evaluates all {model['corner_count_per_candidate']} complete binary corners for every resistor candidate and the frozen amplitude grid without inserting one favorable point. This closes build-readiness plausibility only. Exact as-built parameters, nonlinear residue and transfer must still pass future calibration without tuning any threshold.

## Staged restoration ladder

1. The current authority permits only authored bytes and offline deterministic tests.
2. `{NEXT}` may later authorize either procurement planning or an unpowered build step; the user must choose the extent explicitly.
3. A separately authorized unpowered coupon stage captures official documents, incoming identities, board bytes, continuity, isolation and OPA810 admittance.
4. A separately authorized unpowered full-assembly stage may proceed only if every coupon gate passes.
5. Powered calibration and physical execution remain outside all current authority and require another explicit instruction after the complete assembled receipt is reviewed.

No stage bootstraps authority for the next one.

## Claim law

Offline PASS means only that the deterministic model and analyzer distinguish the frozen synthetic transfer/isolation fixtures from the frozen adversaries. A future physical claim requires committed raw bytes, exact identities, complete source-off witnesses, matched controls and independent adjudication. No optimization, Ising, catalytic-loop, capacity, restoration, Wall, or computation-advantage claim is authorized.
"""


def build_drawings(netlist: dict[str, Any], fabrication: dict[str, Any]) -> str:
    return f"""# P0 authored assembly drawings

All dimensions are millimetres. Datum for every PCB is its lower-left corner, component side up. Panel holes use the machine-coordinate inner-face origins and signed axes in `enclosure_design.panel_face_datums`; they are never mirrored by an exterior viewing convention. These are reviewed prospective drawings under `{AUTHORITY}`; they authorize no fabrication or assembly.

## D001 complete system

One selected complete assembly connects SDG C1 to `J_SRC_C1`, SDG C2 to `J_SRC_C2`, and DN2 channels 0-3 to `J_CH0` through `J_CH3`. No external-trigger cable exists. The two fixture enclosures are joined only by the fixed 150 +/-2 mm harness. Never mix boards or cables across A/B/C.

## D002 control board

`P0-SOURCE-OFF-CONTROL-REV-B-*` is 84.0 x 64.0 x 1.6, four layer, with 3.2 finished non-plated holes at (4,4), (80,4), (4,60), (80,60). It contains C1 limiting/gating, passive C1+C2 monitor, K1/K3, relay drivers, isolated supplies, isolators, calibrated witness ladder, SHT45 and all local bypasses. Exact per-reference coordinates are in `P0_PCB_FABRICATION_RELEASE.json`. Nine exact 60 +/-2 mm Alpha 2840/1 conductors bind the published Nucleo CN3/CN4 lands to the named control-board lands; no detachable internal interface or splice is used.

## D003 carrier/sense board

The A/B/C carrier variants are 68.0 x 32.0 x 1.0, two layer, with 3.2 finished non-plated holes at (4,4), (64,4), (4,28), (64,28). OPA810 pin 3 and the electrode land use a <=5 mm guarded route; no test point, clamp, cable or added copper touches that node. Pins 1, 5 and 8 are isolated no-connect lands. K2 package lands are the only pre-population continuity access.

## D004 vibration board

`P0-ENV-SENSOR-REV-B-*` is 24.0 x 24.0 x 1.0 with 2.7 finished non-plated holes at (2.5,2.5), (21.5,2.5), (2.5,21.5), (21.5,21.5). The ADXL354 Z arrow points normal to the board and toward the enclosure lid. All four supply nodes have local 100 nF capacitors and both internal 1.8 V nodes have 100 kohm discharge paths. Exact 35 +/-2 mm Alpha 2840/1 conductors carry `ADR_REF_3V3` and `AGND_STAR` from the carrier board; the Z output goes directly to its 45 +/-2 mm panel pigtail.

## D005 control enclosure

The control enclosure is exact custom design `P0-CUSTOM-ENCLOSURE-124X82X36-REV-A`. Material, tub/lid dimensions, bare-metal seam, ten fasteners, panel datums/counterbores, tolerances, electrical acceptance, PEEK retention systems, four isolated BNC holes, gland hole, board origins and standoff heights are mechanically bound by `enclosure_design` and the exact `ENC-CONTROL-*` rows in the fabrication release. The Nucleo uses exact `P0-NUCLEO-PEEK-EDGE-TRAY-REV-B`; its admitted-board interval, seat, anchor/countersink geometry, in-outline clip contacts, worst-case deflection budget, tolerances, insertion method, keepout and six retention acceptance checks are fabrication fields. Each control-enclosure mount also binds the +z rotation pivot/convention, board/tray local-to-machine equations, resolved tray footprint and all four floor-anchor centers; no exterior-view mirroring or implicit rotation is permitted. All tray/standoff/washer/screw identities and quantities are separate BOM lines. Maintain >=5 mm lid clearance, >=4 mm wall clearance and >=15 mm coax bend radius. Exact 55 +/-2 mm internal pigtails bind C1, CH0 and CH2 by RG-178 center/shield pairs. C2 uses one PTFE center conductor only; its isolated panel shell has no internal conductor.

## D006 carrier enclosure

The carrier enclosure uses the same exact custom design and acceptance law. Its two isolated BNC holes, gland counterbore, insulated star hole, board origins, PEEK retention identities and standoff heights are the exact `ENC-CARRIER-*` entries. The star hardware must measure at least 1000 Mohm to enclosure metal before the harness shield is attached. Exact 45 +/-2 mm RG-178 pigtails bind CH1 and CH3; their shields terminate only at `AGND_STAR`.

## D007 fixed harness

One Belden 83269 RG-178 center carries `N_MIDPOINT`; its shield is the sole `AGND_EXPORT`-to-`AGND_STAR` bond. Six separately identified Alpha 2840/1 conductors carry +5V_RELAY, K2_COIL_LOW, ADR_REF_3V3, N_WIT_K2, +5V_SENSE and -5V_SENSE. Both `1427CG13` glands use 20.42 +/-0.10 holes. There is no connector or splice.

## D008 witness ladder

The ADuM gate-secondary supply is `ADR_REF_3V3`, so `IN_GATE` and the K1/K2/K3 auxiliary contacts select the same nominal 3.3 V reference through 80.6, 40.2, 20.0 and 10.0 kohm respectively. `IN_GATE` is actively driven low when b0=0, so the 80.6 kohm branch remains a shunt; inactive relay-contact branches float. `N_WITNESS` returns through 1.00 kohm. The corrected nominal equation and all sixteen centroids are frozen in `witness_law`; code 8 is 296.654026 mV and minimum nominal adjacent separation is 24.996066 mV. All 16 centroids and the low-driven b0 impedance are measured later; adjacent centroids must differ by at least ten pooled sigma and every sample must have unique +/-3 sigma membership.

## D009 isolation and no-connect drawing

Every `NC::reference.pin` entry is a land with no copper beyond the pad. Every `DNP::` entry is unpopulated and isolated. The controller, relay and analog domains have no undeclared galvanic bond. The C2 rig-end shell is isolated. DN2 input admittance is calibrated and is not described as isolation.

## D010 fixture population matrix

- A: FC-135 populated; dummy open.
- B: FC-135 and dummy both open.
- C: FC-135 open; exact 1.0 pF C0G dummy populated.

All other parts, boards, enclosures, harnesses and dedicated external cables are fixture-local and matched. The complete machine-readable topology is `P0_FINAL_NETLIST.json`; the coordinate release has {len(fabrication['boards'])} PCB instances and {len(fabrication['enclosures'])} enclosure instances.
"""


def build_assembly(netlist: dict[str, Any]) -> str:
    return f"""# P0 unpowered assembly packet

Status: **NOT AUTHORIZED**. This is a prospective, non-purchasing procedure under `{AUTHORITY}`. Every physical verb below is conditional on a later explicit user instruction.

## Stage U0 — incoming custody

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: capture manufacturer, complete order code, quantity, lot/date/packing code, marking photographs, official-document bytes and SHA-256 for every BOM line. Reject any substitution. Bind PCB fabrication bytes and enclosure machining report to the reviewed qualification root.

Before opening any ESD-sensitive package, record the grounded mat, wrist-strap tester, earth point, room temperature/RH and operator identifier. Source, digitizer, controller USB, external coaxes and all power leads remain physically absent. The ESD receipt is evidence only and cannot authorize the next stage.

## Stage U1 — bare-board coupons

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: inspect dimensions, drill, finish, layer stack and net isolation for all nine PCBs. Use a current-limited LCR fixture with no source or digitizer attached to characterize the guarded carrier input land. Reject any undeclared continuity, no-connect copper or fabrication-byte mismatch.

## Stage U2 — sense coupons

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: populate OPA810, K2, bias return and local bypass on each carrier/control coupon, leaving FC-135/dummy positions open. Measure `Cin,U95` and `Rin,U95` over the frozen environment envelope. Require <=4.00 pF and >=100 Mohm independently for A, B and C. This gate is mandatory because the amplifier data-sheet capacitances are typical, not guaranteed.

## Stage U3 — remaining unpowered board assembly

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: use the lower of every exact component document's soldering limits and the qualified board-process limit. Record paste/flux/cleaner identity, profile or iron-tip temperature, dwell, operator and rework count. Populate in this fixed order: fuses, current limiters and terminations; isolated supplies and bypasses; isolators, gate, relay drivers and suppression; K1/K3 and witness ladder; reference and sensors; K2, bias return and OPA810 last. Inspect polarity and pin 1 after each group, then clean and verify guarded surfaces using the qualified process. Confirm each `NC::` land has no copper beyond its pad and every `DNP::` location is open. Do not install FC-135 or dummy until K2 land continuity has been recorded. Any undocumented rework or exceeded document limit rejects the assembly.

## Stage U4 — continuity and isolation

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: execute every entry in `continuity_tests` and every forbidden pair in `P0_FINAL_NETLIST.json`. Record instrument identity, range, uncertainty, probes, raw readings and photographs. K1/K2/K3 signal-pole continuity and auxiliary-pole continuity are separate receipts. Do not infer signal isolation from an auxiliary contact. Measure every CH2 ladder resistor and the 3.3 V-reference path unpowered, compute the prospective 16-code centroid ordering from the measured values, and reject any swapped weight or nonmonotone code. No voltage is applied to a carrier electrode in this stage.

## Stage U5 — exact fixture populations

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: populate A with exact Epson `Q13FC1350000401`, leave B open, and populate C with exact Murata `GJM1555C1H1R0BB01D` 1.0 pF C0G. Apply the lowest-stress attachment process allowed by each exact document and record temperature/dwell. Photograph top, bottom, pin orientation, land wetting and enclosure clearance. Re-run input admittance and all continuity checks.

## Stage U6 — mechanics and harness

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: machine holes from `P0_PCB_FABRICATION_RELEASE.json`; deburr, clean and dimensionally inspect them. Install integral-isolation BNC bodies and the insulated M4 star. Bind every lid screw, standoff, PEEK washer, board screw, Nucleo tray and flush PEEK tray screw to its exact BOM line and fabrication geometry. Inspect the complete tray profile and tolerances, install the Nucleo only by the bounded four-ramp perimeter-force method in the fabrication release, then execute every tray insertion, lateral-motion, upward-retention, keepout, flush-head and visual acceptance check. Build one unspliced 150 +/-2 mm inter-enclosure harness per fixture. Build the exact internal cut sets from `intra_enclosure_harnesses`: nine 60 +/-2 mm Nucleo conductors, control-panel 55 +/-2 mm pigtails, carrier-panel 45 +/-2 mm pigtails, and two 35 +/-2 mm carrier-to-vibration conductors. The C2 panel shell receives no internal conductor. Verify every conductor end-to-end, every shield endpoint, pull retention, strain relief, and the sole inter-enclosure shield bond before mounting boards at the frozen origins and standoff heights.

Label every board, enclosure, fixed conductor and external coax with fixture `A`, `B` or `C` plus its exact connector/net identity. Route the guarded sense node only on its authored board, maintain the declared coax bend radius, keep relay/control conductors outside the carrier keepout and photograph every shield termination and strain relief before closure.

## Stage U7 — closed unpowered assembly receipt

`UNAUTHORIZED UNTIL SEPARATE USER AUTHORITY`

After separate authority only: with every external cable absent, re-run all forbidden-pair and path tests through the closed enclosures. Attach the six dedicated labeled external coaxes without source, digitizer, controller USB or power. Verify connector-center and shield mapping. Create one canonical as-built manifest containing the reviewed candidate root, board/fabrication hashes, every component and document identity, lot/packing fields, process receipts, measured continuity/isolation/admittance values, instrument calibration identities, photographs and file hashes. Seal it before closure, stop, and return the complete receipt for review.

No step in this packet authorizes power, playback, recording, instrument commands, source output, acquisition, calibration, or experiment execution. The next boundary remains `{NEXT}`.
"""


def build_execution_contract(netlist: dict[str, Any]) -> str:
    model = read_canonical_json(ROOT / "P0_SIGNAL_PATH_CIRCUIT_MODEL.json")
    return f"""# Future P0 physical execution contract

Status: **NOT AUTHORIZED**. Claim ceiling: `{CEILING}`. This contract is a prospective hard-stop checklist and grants no authority.

## Required prior receipts

1. Separate user authority for the exact proposed stage.
2. Reviewed qualification root binding netlist, BOM, fabrication, analyzer, schemas, fixtures and four role-separated root-bound review declarations. Those declarations are not represented as externally reproducible independence.
3. Official-document hashes and incoming custody for every acquired item.
4. Passed bare-board, sense-coupon, continuity, isolation, mechanics and closed-assembly receipts for A/B/C.
5. Exact as-built `Cin,U95 <= 4.00 pF` and `Rin,U95 >= 100 Mohm` on all three sense paths.

## Future source contract

One exact SDG1032X in `HIGH_Z` load mode, with 50 ohm physical output impedance, supplies continuous C1 at bound `f_carrier_hz`, 0.400 Vpp and 0 V offset, and phase-locked C2 at bound `f_witness_hz = 2 * f_carrier_hz`, {model['mechanism']['selected_amplitude_vpp']:.3f} Vpp and 0 V offset. C1 carries the 0/pi arm command. C2 remains fixed at zero phase, enters the passive CH0 monitor, and also enters the exact 1.00 Mohm TNPW branch to `N_GATE_OUT`; it never passes through ADG1419. The exact 100 kohm `N_SRC` drive shunt is on the ADG D/SA side during OFF and cannot define the C2 witness. Both source channels stay on through the entire record after at least 3.000 seconds of preparation. No burst and no external trigger is allowed. All source/queryback and frequency-uncertainty custody remains mandatory.

## Future acquisition contract

One exact DN2.592-04 captures four simultaneous true-differential channels at 1,000,000 samples/s for 3,101,000 samples/channel using software-prearmed free run. Its input mode is frozen to 1 Mohm in parallel with 30 pF. The present raw bytes support differential clipping only; no common-mode voltage is inferred from them. A future powered operating envelope must separately define and validate common-mode observability before authority. CH2 locates source isolation and C2 supplies the phase gauge. No proprietary container parser is claimed by this packet. A separately authorized acquisition must preserve any original proprietary container, run the frozen SDK lossless-export mode to the exact 24,808,000-byte signed-int16 canonical payload, require SDK-export and analyzer-payload hashes/counts to be identical, and supply actual byte descriptors for adapter source, instrument/source querybacks, native export, assignment commitment/reveal, environmental calibration, resonance calibration artifact/raw/analyzer, chronology, the assembly manifest, the event-specific version-2 topology receipt, a unique four-state C2 topology scan and a unique C1-only nonlinear-control trace. The receipt must bind the assigned role, exact A/B/C assembly and population, calibration-derived frequency tuple, native payload, chronology, querybacks, raw scan/control bytes and pre-acquisition scan times. All times must parse from exact `YYYY-MM-DDTHH:MM:SS.ffffffZ` UTC form. A roles reuse one exact A manifest; B and C use distinct manifests. Cross-assembly, cross-event, duplicate-scan, duplicate-control, noncanonical-time or post-acquisition replay is a hard stop. Missing or mismatched bytes are a hard stop; hash-shaped metadata alone is insufficient.

## Future source-off contract

ADG1419 terminates C1 into 50.00 ohm. The same-ADG-state 192-sample C2 pre-window completes before K1/K2 release at 250 microseconds. K3 stays energized/electrically open while code 0 remains stable for 1,000 samples and the 960-sample C2 isolated-path window passes. Only then may K3 release to guard; code 8 must remain stable for 1,000 samples and the 10 ms guard follows. Auxiliary contacts do not identify either signal pole. A passing end-to-end transfer event supports only the exact actual-path isolation token and at least one-open meaning. Any guard masking, wrong-node injection, missing C2, excessive open feedthrough, inversion, bounce, re-entry, hidden buffer, replay, trigger, muted source, wrong termination, wrong fixture, invalid CRC or ancestry mismatch rejects the record.

## Future controls

Every session requires A DUT, B detector-only and C exact-dummy records with randomized order fixed before connection. Ordinary DSP replay, decoded-spin replay, flat waveforms, spectral leakage, metadata leakage, query preselection, file/buffer persistence and restoration overclaim remain explicit adversaries. A physical observation cannot establish a catalytic, Ising, optimization, capacity, Wall or computation-advantage claim.

## Stop law

The first identity, custody, calibration, environment, topology, witness, raw-byte, control or analyzer failure stops the future run. No retry or parameter adjustment is implicit. Nothing in this file permits hardware contact now. Next boundary: `{NEXT}`.
"""


def build_findings() -> dict[str, Any]:
    closures = (
        ("P0BR-F001", "BLOCKER", "The prior 8 pF sense part violated the frozen loading envelope.", ["P0_FINAL_NETLIST.json#admittance_budget", "P0_BUILD_READINESS_PACKET.md#sense-admittance-and-the-opa810-correction", "P0_COMPONENT_DOCUMENTS.json#OPA810"]),
        ("P0BR-F002", "BLOCKER", "The prior external trigger fanout was undefined and risked an undeclared ground bond.", ["P0_FINAL_NETLIST.json#ground_model", "P0_BUILD_READINESS_PACKET.md#software-prearmed-acquisition", "p0_scientific_analyzer.py#software-prearmed-free-run"]),
        ("P0BR-F003", "BLOCKER", "A self-ending burst contradicted physical source-off custody.", ["P0_FINAL_NETLIST.json#source_off_sequence", "P0_BUILD_READINESS_PACKET.md#continuous-dual-tone-source-and-phase-gauge", "P0_ANALYZER_REFERENCE_RESULTS.json#source_muted_at_gate_raw"]),
        ("P0BR-F004", "MATERIAL", "Detector-only and dummy controls were connector labels rather than complete circuits.", ["P0_FINAL_NETLIST.json#control_fixtures", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D010", "P0_NONPURCHASING_BOM.json#three-complete-fixtures"]),
        ("P0BR-F005", "MATERIAL", "The isolator primary supply was incompatible with 3.3 V controller outputs.", ["P0_FINAL_NETLIST.json#UISO_GATE_A", "P0_FINAL_NETLIST.json#UISO_RELAY_A", "p0_build_readiness_validator.py#check_netlist"]),
        ("P0BR-F006", "MATERIAL", "The vibration sensor and support-network bypass inventory was incomplete.", ["P0_FINAL_NETLIST.json#C_VDDIO_A", "P0_FINAL_NETLIST.json#C_VSUPPLY_A", "P0_NONPURCHASING_BOM.json#derived-electrical-inventory"]),
        ("P0BR-F007", "MATERIAL", "Ground drawings disagreed on the sole inter-enclosure analog bond.", ["P0_FINAL_NETLIST.json#ground_model", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D007", "P0_UNPOWERED_ASSEMBLY_PACKET.md#Stage-U6"]),
        ("P0BR-F008", "MATERIAL", "Mechanical fit, placement, keepout, panel, mounting and strain-relief details were incomplete.", ["P0_PCB_FABRICATION_RELEASE.json", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D002-D007", "p0_build_readiness_validator.py#check_fabrication"]),
        ("P0BR-F009", "MATERIAL", "Board ownership, no-connect semantics, rails and dangling control nets were incomplete.", ["P0_FINAL_NETLIST.json#boards-components-nets-power_domains", "p0_build_readiness_validator.py#exact-net-projection", "p0_build_readiness_design.py#suffixed"]),
        ("P0BR-F010", "MATERIAL", "Relay continuity called for nonexistent test pads and conflicted with the electrode loading law.", ["P0_FINAL_NETLIST.json#continuity_tests", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D003", "P0_UNPOWERED_ASSEMBLY_PACKET.md#Stage-U4"]),
        ("P0BR-F011", "BLOCKER", "The earlier candidate was not byte-root-bound. The current exhaustive mutation pass proves root sensitivity only; semantic rejection remains a separate validation responsibility and is not claimed by the mutation receipt.", ["p0_build_readiness_validator.py#candidate_root", "p0_build_readiness_validator.py#check_reviews", "p0_build_readiness_mutation_test.py", "P0_BUILD_READINESS_MUTATION_RESULTS.json"]),
        ("P0BR-F012", "MATERIAL", "Environment custody omitted raw CRC/identity/clock evidence and hard-coded one year.", ["P0_BUILD_READINESS_SCHEMAS.json#environment_csv", "p0_scientific_analyzer.py#crc8", "P0_ANALYZER_REFERENCE_RESULTS.json#environment-adversaries"]),
        ("P0BR-F013", "MATERIAL", "Normative documents prematurely asserted a frozen decision and disagreed on authority tokens.", ["P0_BUILD_READINESS_AUTHORITY.md#current-status", "PHYSICAL_PHASE_CARRIER_P0_CONTRACT.md#status", "../AUDIO_SIDE_QUEST_ROADMAP.md#status", "p0_build_readiness_validator.py#check_reviews"]),
        ("P0BR-F014", "MATERIAL", "Document, coupon, assembly and execution gates were circular.", ["P0_BUILD_READINESS_PACKET.md#staged-restoration-ladder", "P0_UNPOWERED_ASSEMBLY_PACKET.md#Stage-U0-U7", "P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md#required-prior-receipts"]),
        ("P0BR-F015", "BLOCKER", "The connector map ended at panel bulkheads and omitted Nucleo, panel-pigtail and carrier-to-vibration wiring.", ["P0_FINAL_NETLIST.json#intra_enclosure_harnesses", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D002-D006", "P0_UNPOWERED_ASSEMBLY_PACKET.md#Stage-U6"]),
        ("P0BR-F016", "BLOCKER", "The CH2 ladder claimed binary conductance weights while b0 was driven from a different logic voltage.", ["P0_FINAL_NETLIST.json#UISO_GATE_A", "P0_FINAL_NETLIST.json#witness_law", "P0_AUTHORED_ASSEMBLY_DRAWINGS.md#D008"]),
    )
    return {
        "authority": AUTHORITY,
        "claim_ceiling": CEILING,
        "decision": "P0_BUILD_READINESS_PACKET_FROZEN",
        "findings": [
            {"closure_evidence": evidence, "finding_id": finding_id, "original_summary": summary, "severity": severity, "status": "CLOSED"}
            for finding_id, severity, summary, evidence in closures
        ] + [{
            "closure_evidence": ["P0_SIGNAL_PATH_CIRCUIT_MODEL.json", "P0_FINAL_NETLIST.json#source_off_sequence.signal_pole_evidence_boundary", "p0_scientific_analyzer.py#signal_path_transfer", "p0_scientific_analyzer.py#ASSEMBLY_FOR_ROLE", "P0_SIGNAL_PATH_ORDERING_PROOF.json", "P0_ANALYZER_REFERENCE_RESULTS.json#signal_path_control_outcomes", "P0_SIGNAL_PATH_MUTATION_RESULTS.json", "P0_BUILD_READINESS_REVIEWS.json"],
            "finding_id": "P0BR-R3-SIGNAL-POLE",
            "original_summary": "Auxiliary CH2 contacts do not provide per-event evidence that K1/K2 signal poles opened; closure requires a prospective actual-path witness plus assembly-, role-, event-, queryback- and parsed-UTC chronology custody that rejects A-to-B/C, cross-event scan/receipt/control replay and noncanonical time text.",
            "severity": "BLOCKER",
            "status": "CLOSED",
        }],
        "open_material_findings": 0,
        "repair_decision": "P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED",
        "repair_review_gate": "satisfied only by four focused PASS receipts bound to the exact final candidate root",
        "schema": "p0.build-readiness-findings.v3",
        "contact_attestation": {"audio_playback_or_recording": 0, "cart_or_stock_check": 0, "hardware": 0, "human_vendor_outreach": 0, "instrument_command": 0, "purchase": 0, "target": 0},
        "public_source_retrieval": {"automated_http_attempts_occurred": True, "repository_safe_receipt": f"{RESEARCH_RELATIVE_PATH}/SOURCE_CUSTODY.json", "third_party_bytes_private_and_ignored": True},
    }


def write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> int:
    netlist = build_netlist()
    bom = build_bom(netlist)
    fabrication = build_fabrication(netlist)
    artifacts = {
        "P0_FINAL_NETLIST.json": canonical(netlist),
        "P0_NONPURCHASING_BOM.json": canonical(bom),
        "P0_PCB_FABRICATION_RELEASE.json": canonical(fabrication),
        "P0_COMPONENT_DOCUMENTS.json": canonical(build_component_documents()),
        "P0_BUILD_READINESS_FINDINGS.json": canonical(build_findings()),
        "P0_BUILD_READINESS_PACKET.md": build_packet(netlist, bom, fabrication).encode("utf-8"),
        "P0_AUTHORED_ASSEMBLY_DRAWINGS.md": build_drawings(netlist, fabrication).encode("utf-8"),
        "P0_UNPOWERED_ASSEMBLY_PACKET.md": build_assembly(netlist).encode("utf-8"),
        "P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md": build_execution_contract(netlist).encode("utf-8"),
    }
    for name, data in artifacts.items():
        write(ROOT / name, data)
    receipt = {"artifacts": {name: {"bytes": len(data), "sha256": sha256_bytes(data)} for name, data in sorted(artifacts.items())}, "authority": AUTHORITY, "result": "PASS", "zero_hardware_contact": True}
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

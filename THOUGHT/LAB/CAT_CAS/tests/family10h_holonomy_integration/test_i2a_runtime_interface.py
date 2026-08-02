from __future__ import annotations

from pathlib import Path
import sys


CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
    / "phase6b6"
    / "live_small_wall"
    / "carrier_state_tomography"
    / "family10h_holonomy_integration_v1"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from i2a_runtime_interface import validate_interface  # noqa: E402


def test_i2a_interface_binds_exact_c_blobs() -> None:
    result = validate_interface()
    assert result["passed"] is True
    assert set(result["implementation_blobs"]) == {
        "post_source_operator_runtime.h",
        "post_source_operator_runtime.c",
    }
    assert result["lifecycle"][2:5] == [
        "SOURCE_DEAD_SEALED",
        "RECEIVER_WORD_OPEN",
        "RECEIVER_WORD_CLOSED",
    ]


def test_i2a_interface_remains_compile_only() -> None:
    result = validate_interface()
    assert result["decision"] == (
        "I2A_COMPILE_ONLY_POST_SOURCE_RUNTIME_INTERFACE_ESTABLISHED__PHYSICAL_BACKEND_NOT_IMPLEMENTED"
    )
    assert result["next_gate"] == "I2B_SECOND_GENERATOR_AND_INVERSE_CANDIDATE_DESIGN"
    assert "POST_SOURCE_OPERATOR_INTERFACE_ONLY" in result["claim_ceiling"]
    assert "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED" in result["claim_ceiling"]


def test_i2a_compile_contract_is_strict() -> None:
    result = validate_interface()
    flags = result["compile_flags"]
    for required in (
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-DF10HI_BUILD_SELF_TEST",
    ):
        assert required in flags

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = LAB_ROOT / "tools" / "validate_control_plane.py"
PHASE_LOCK = LAB_ROOT / "tools" / "phase_lock.py"
GRAPH_GENERATOR = LAB_ROOT / "tools" / "generate_capability_graph.py"


def run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(arg) for arg in args)],
        cwd=LAB_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def complete_common_receipt(receipt: dict[str, object]) -> None:
    receipt.update(
        {
            "agent": "cold-agent-test",
            "mission_summary": "Finite reusable substrate carries unresolved .holo geometry through native evolution to one classical boundary answer and restoration.",
            "infinite_compute_definition": "Compute leverage grows when native work does not scale with the classical path space being represented.",
            "holo_role": "Executable noncollapse geometric memory, not an answer file.",
            "pi_torus_role": "Phase lives on U(1); pi is the antipodal binary shadow and two pi closes the local loop.",
            "classical_allowed_stages": ["parse", "orchestrate", "package", "official_verify"],
            "native_middle": "Unresolved relational geometry evolves without candidate enumeration or full materialization.",
            "final_classical_boundary": "One declared witness projection consumed by the official verifier.",
            "restoration_role": "Close and reuse the borrowed substrate after invariant extraction.",
            "current_frontier": "Exp 50 remains below fixed-point advantage; audio develops the U(1) carrier and Track 8 targets external acceptance.",
            "phase_locked": True,
        }
    )


def test_control_plane_validator_passes() -> None:
    result = run(VALIDATOR)
    assert result.returncode == 0, result.stdout
    assert "CAT_CAS_CONTROL_PLANE_PASS" in result.stdout


def test_capability_graph_is_current() -> None:
    result = run(GRAPH_GENERATOR, "--check")
    assert result.returncode == 0, result.stdout
    assert "CAT_CAS_CAPABILITY_GRAPH_CURRENT" in result.stdout


def test_audio_phase_lock_loads_registered_branch_context(tmp_path: Path) -> None:
    result = run(
        PHASE_LOCK,
        "--task",
        "Continue the pi-native recursive audio phase computer",
        "--mode",
        "engineering",
        "--task-class",
        "enabling_infrastructure",
        "--branch",
        "codex/audio-frequency-wave-substrate",
        "--output",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout
    packet = (tmp_path / "PHASE_LOCK_PACKET.md").read_text(encoding="utf-8")
    receipt = json.loads((tmp_path / "PHASE_LOCK_RECEIPT.json").read_text(encoding="utf-8"))
    assert "REPLACE THE BIT WITH PI" in packet
    assert "AUDIO_SIDEQUEST" in packet
    assert receipt["branch"] == "codex/audio-frequency-wave-substrate"
    assert "AUDIO_SIDEQUEST" in receipt["selected_capability_nodes"]
    assert receipt["current_claim_ceiling"] == "NON_EXECUTING_PHYSICAL_PHASE_CARRIER_BUILD_READINESS_ONLY"


def test_uncompleted_flagship_receipt_fails(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    create = run(
        PHASE_LOCK,
        "--task",
        "Design a bounty experiment that tests fixed-point compute leverage",
        "--mode",
        "engineering",
        "--task-class",
        "flagship_compute",
        "--branch",
        "main",
        "--output",
        generated,
    )
    assert create.returncode == 0, create.stdout
    result = run(VALIDATOR, "--receipt", generated / "PHASE_LOCK_RECEIPT.json")
    assert result.returncode == 1
    assert "incomplete classical_explosion" in result.stdout
    assert "phase_locked must be true" in result.stdout


def test_completed_flagship_receipt_passes(tmp_path: Path) -> None:
    generated = tmp_path / "completed"
    create = run(
        PHASE_LOCK,
        "--task",
        "Design a bounty experiment that tests fixed-point compute leverage",
        "--mode",
        "engineering",
        "--task-class",
        "flagship_compute",
        "--branch",
        "main",
        "--output",
        generated,
    )
    assert create.returncode == 0, create.stdout
    receipt_path = generated / "PHASE_LOCK_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    complete_common_receipt(receipt)
    receipt.update(
        {
            "classical_explosion": "The matched forward solver enumerates an exponentially growing witness space.",
            "native_operator": "A prospectively frozen operator acts on compact public relational generators without witness-conditioned control.",
            "invariant_or_fixed_point": "An accepted fixed point of the public verifier map.",
            "no_smuggle_killer_control": "Blinded instances and an answer-cache adversary must fail unless the native mechanism exposes a new channel.",
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = run(VALIDATOR, "--receipt", receipt_path)
    assert result.returncode == 0, result.stdout


def test_external_product_receipt_does_not_claim_flagship_fields(tmp_path: Path) -> None:
    generated = tmp_path / "product"
    create = run(
        PHASE_LOCK,
        "--task",
        "Build a classical surface integrity product with CAT_CAS provenance",
        "--mode",
        "engineering",
        "--task-class",
        "external_product",
        "--branch",
        "main",
        "--output",
        generated,
    )
    assert create.returncode == 0, create.stdout
    receipt_path = generated / "PHASE_LOCK_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    complete_common_receipt(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = run(VALIDATOR, "--receipt", receipt_path)
    assert result.returncode == 0, result.stdout

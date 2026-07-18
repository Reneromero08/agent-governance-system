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


def complete_task_contract(contract: dict[str, object]) -> None:
    contract.update(
        {
            "experiment_id": "TEST_FRONTIER",
            "collapse_boundary": "One declared classical witness projection.",
            "restoration_law": "The borrowed carrier returns to its pre-run state or declared equivalence class.",
            "known_blockers": ["test-only"],
            "stop_conditions": ["official verifier rejection"],
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
        "--output",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout
    packet = (tmp_path / "PHASE_LOCK_PACKET.md").read_text(encoding="utf-8")
    receipt = json.loads((tmp_path / "PHASE_LOCK_RECEIPT.json").read_text(encoding="utf-8"))
    assert "REPLACE THE BIT WITH PI" in packet
    assert "AUDIO_SIDEQUEST" in packet
    assert receipt["branch"] == "codex/audio-frequency-wave-substrate"
    assert receipt["commit"] == "6c1875b2b4c39588ab5bdc4878a317671329b0f0"
    assert "AUDIO_SIDEQUEST" in receipt["selected_capability_nodes"]
    assert "HOLO_GEN5" in receipt["selected_capability_nodes"]
    assert receipt["selected_code_paths"]
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

    contract_path = generated / "TASK_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    complete_task_contract(contract)
    contract.update(
        {
            "classical_explosion": receipt["classical_explosion"],
            "holo_process_object": "A compact unresolved public relation whose classical witness space grows exponentially.",
            "native_operator": receipt["native_operator"],
            "invariant_or_fixed_point": receipt["invariant_or_fixed_point"],
            "no_smuggle_killer_control": receipt["no_smuggle_killer_control"],
            "official_boundary": "The frozen official bounty verifier.",
        }
    )
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run(
        VALIDATOR,
        "--receipt",
        receipt_path,
        "--task-contract",
        contract_path,
    )
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

    contract_path = generated / "TASK_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    complete_task_contract(contract)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run(
        VALIDATOR,
        "--receipt",
        receipt_path,
        "--task-contract",
        contract_path,
    )
    assert result.returncode == 0, result.stdout


def test_flagship_selection_always_includes_holo_runtime(tmp_path: Path) -> None:
    generated = tmp_path / "flagship-selection"
    result = run(
        PHASE_LOCK,
        "--task",
        "Design the first bounty experiment that directly tests compute leverage",
        "--mode",
        "engineering",
        "--task-class",
        "flagship_compute",
        "--branch",
        "main",
        "--output",
        generated,
    )
    assert result.returncode == 0, result.stdout
    receipt = json.loads((generated / "PHASE_LOCK_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["selected_capability_nodes"][0] == "TRACK8"
    assert "HOLO_GEN5" in receipt["selected_capability_nodes"]
    assert "EXP49" in receipt["selected_capability_nodes"]
    assert "EXP50" in receipt["selected_capability_nodes"]
    assert "TRACK8_DOMAIN_ADAPTERS_NOT_SUBMISSION_READY" in receipt["current_claim_ceiling"]
    assert "EXTERNAL_BOUNTY_ACCEPTANCE_NOT_ESTABLISHED" in receipt["current_claim_ceiling"]


def test_tampered_context_digest_fails(tmp_path: Path) -> None:
    generated = tmp_path / "tampered"
    create = run(
        PHASE_LOCK,
        "--task",
        "Build a classical external product",
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
    receipt["context_digest"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = run(VALIDATOR, "--receipt", receipt_path)
    assert result.returncode == 1
    assert "context_digest does not match current control inputs" in result.stdout


def test_unregistered_engineering_branch_is_blocked(tmp_path: Path) -> None:
    result = run(
        PHASE_LOCK,
        "--task",
        "Implement an experiment",
        "--mode",
        "engineering",
        "--task-class",
        "enabling_infrastructure",
        "--branch",
        "unregistered/research-branch",
        "--output",
        tmp_path / "blocked",
    )
    assert result.returncode != 0
    assert "Cannot resolve target branch commit" in result.stdout or "Branch context blocks" in result.stdout


def test_packet_uses_actual_validation_paths(tmp_path: Path) -> None:
    generated = tmp_path / "named-output"
    result = run(
        PHASE_LOCK,
        "--task",
        "Inspect CAT_CAS evidence",
        "--mode",
        "verification",
        "--task-class",
        "evidence_audit",
        "--branch",
        "main",
        "--output",
        generated,
    )
    assert result.returncode == 0, result.stdout
    packet = (generated / "PHASE_LOCK_PACKET.md").read_text(encoding="utf-8")
    assert (generated / "PHASE_LOCK_RECEIPT.json").as_posix() in packet
    assert (generated / "TASK_CONTRACT.json").as_posix() in packet

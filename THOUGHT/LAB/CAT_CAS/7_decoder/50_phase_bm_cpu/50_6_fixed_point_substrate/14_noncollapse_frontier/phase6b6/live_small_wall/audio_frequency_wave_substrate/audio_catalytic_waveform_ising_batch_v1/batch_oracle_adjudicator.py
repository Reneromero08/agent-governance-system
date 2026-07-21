from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
HELDOUT_SOURCE = (
    SUBSTRATE_DIR
    / "audio_catalytic_waveform_ising_heldout_v1"
    / "heldout_generalization_reference.py"
)
CUSTODY_FILE = PACKAGE_DIR / "BATCH_INSTANCE_CUSTODY.json"
AUTHORITY_FILE = PACKAGE_DIR / "BATCH_EXPERIMENT_AUTHORITY.txt"
PREORACLE_CONTRACT_FILE = PACKAGE_DIR / "BATCH_PREORACLE_CONTRACT.json"
PREORACLE_EVIDENCE_FILE = PACKAGE_DIR / "BATCH_PREORACLE_EVIDENCE.json"
PREORACLE_TRACE_FILE = PACKAGE_DIR / "BATCH_PREORACLE_TRACE.json"
PREORACLE_SEAL_FILE = PACKAGE_DIR / "BATCH_PREORACLE_SEAL.json"
RESULTS_FILE = PACKAGE_DIR / "BATCH_GENERALIZATION_RESULTS.json"
TRACE_FILE = PACKAGE_DIR / "BATCH_GENERALIZATION_ORACLE_TRACE.json"
REPORT_FILE = PACKAGE_DIR / "BATCH_GENERALIZATION_REPORT.md"
MANIFEST_FILE = PACKAGE_DIR / "BATCH_GENERALIZATION_MANIFEST.json"

FREEZE_COMMIT = "b6b53493722aeca5cc8cc38bb41f9e9be66afb68"
PREORACLE_COMMIT = "bb259b5a32ccfa9505d0fe7c61cbec0e39a57c3c"
ORDERED_BATCH_SHA256 = (
    "4109d430789b8fb3912ad606b78311855e89e40b422fb3ecec9b84f5818c0c12"
)
PREORACLE_EVIDENCE_SHA256 = (
    "1d05b0ab68751d34dd7199e6fcb94b12a9a340c2c3e36deebeb685747145b236"
)
PREORACLE_TRACE_SHA256 = (
    "cfabefe4a2b5dba035ea75b92358d7741351b2f4b78694a57fc9f493f3b32cae"
)
FROZEN_MACHINE_SHA256 = (
    "cf95d0cd364af38d47a2f2784aa489ab5a52dc8aea62131c1a8545ff4978203a"
)
AUTHORITY_SHA256 = (
    "ed537759d47cc69f0844a913113a2736a6ac8345550c06339be1bb253d3aee35"
)
AUTHORITY_BYTES = 11011

RESULT_SCHEMA = "catalytic_waveform_ising_batch_generalization_result_v1"
TRACE_SCHEMA = "catalytic_waveform_ising_batch_oracle_trace_v1"
MANIFEST_SCHEMA = "catalytic_waveform_ising_batch_generalization_manifest_v1"
VERIFIED = "CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_VERIFIED"
PARTIAL = "CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_PARTIAL"
NOT_ESTABLISHED = "CATALYTIC_WAVEFORM_ISING_BATCH_GENERALIZATION_NOT_ESTABLISHED"

# These null/adversarial baselines are executed for every frozen instance by the
# separately sealed pre-oracle runner; the adjudicator consumes their exact outcomes.
NULL_MODEL_CONTROLS = (
    "uniform_carrier_replacement",
    "flat_geometry",
    "scrambled_parent_child_geometry",
    "removed_waveform_transform",
    "removed_pair_operator",
    "no_lock",
    "wrong_query",
    "wrong_inverse",
    "omitted_inverse_step",
    "omitted_restoration",
    "samplewise_non_rank_one",
)


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


held = load_module(HELDOUT_SOURCE, "catcas_batch_adjudication_reference")
r4 = held.r4


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, canonical_bytes(value))


def verify_preoracle_root() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    custody = load_json(CUSTODY_FILE)
    contract = load_json(PREORACLE_CONTRACT_FILE)
    evidence = load_json(PREORACLE_EVIDENCE_FILE)
    trace = load_json(PREORACLE_TRACE_FILE)
    seal = load_json(PREORACLE_SEAL_FILE)
    if custody["ordered_batch_sha256"] != ORDERED_BATCH_SHA256:
        raise ValueError("custody ordered batch hash changed")
    if sha256_file(PREORACLE_EVIDENCE_FILE) != PREORACLE_EVIDENCE_SHA256:
        raise ValueError("pre-oracle evidence bytes changed")
    if sha256_file(PREORACLE_TRACE_FILE) != PREORACLE_TRACE_SHA256:
        raise ValueError("pre-oracle trace bytes changed")
    if seal["preoracle_evidence_sha256"] != PREORACLE_EVIDENCE_SHA256:
        raise ValueError("pre-oracle seal does not bind the evidence")
    if seal["preoracle_trace_sha256"] != PREORACLE_TRACE_SHA256:
        raise ValueError("pre-oracle seal does not bind the trace")
    if seal["oracle_call_count"] != 0:
        raise ValueError("pre-oracle runner consulted an oracle")
    if evidence["frozen_machine"]["machine_sha256"] != FROZEN_MACHINE_SHA256:
        raise ValueError("pre-oracle evidence used a different machine")
    if contract["frozen_machine"]["machine_sha256"] != FROZEN_MACHINE_SHA256:
        raise ValueError("pre-oracle contract used a different machine")
    if evidence["ordered_batch_sha256"] != ORDERED_BATCH_SHA256:
        raise ValueError("pre-oracle evidence used a different batch")
    if len(evidence["instances"]) != len(custody["ordered_instances"]):
        raise ValueError("pre-oracle evidence does not cover the complete batch")
    for frozen, observed in zip(custody["ordered_instances"], evidence["instances"]):
        if frozen["index"] != observed["index"]:
            raise ValueError("pre-oracle instance order changed")
        if frozen["instance_sha256"] != observed["instance_sha256"]:
            raise ValueError("pre-oracle instance identity changed")
    if trace["events"][-1]["name"] != "full_preoracle_batch_root_sealed":
        raise ValueError("pre-oracle trace did not end at the batch root seal")
    return custody, contract, evidence


def verify_experiment_authority() -> dict[str, Any]:
    payload = AUTHORITY_FILE.read_bytes()
    if len(payload) != AUTHORITY_BYTES:
        raise ValueError("prospective experiment authority byte count changed")
    if sha256_bytes(payload) != AUTHORITY_SHA256:
        raise ValueError("prospective experiment authority hash changed")
    text = payload.decode("utf-8")
    required_fragments = (
        VERIFIED,
        PARTIAL,
        NOT_ESTABLISHED,
        "multiple raw-correct-below-gate results",
        "coherence remaining the dominant limitation",
        "material history or complex-response changes",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(f"prospective experiment authority is incomplete: {missing}")
    return {
        "bytes": AUTHORITY_BYTES,
        "decision_law_frozen_before_execution": True,
        "required_fragment_count": len(required_fragments),
        "sha256": AUTHORITY_SHA256,
        "source": "exact_user_supplied_experiment_contract_bytes",
    }


@dataclass
class EventLedger:
    events: list[dict[str, Any]]
    previous_hash: str = "0" * 64

    def add(self, name: str, payload: Any) -> None:
        record = {
            "name": name,
            "payload_sha256": sha256_bytes(canonical_bytes(payload)),
            "previous_event_sha256": self.previous_hash,
            "sequence": len(self.events),
        }
        record["event_sha256"] = sha256_bytes(canonical_bytes(record))
        self.events.append(record)
        self.previous_hash = record["event_sha256"]

    def document(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),
            "events": self.events,
            "final_event_sha256": self.previous_hash,
            "schema": TRACE_SCHEMA,
        }


def exact_oracle_document(instance: dict[str, Any]) -> dict[str, Any]:
    coupling, field = r4.validate_problem(
        np.asarray(instance["coupling_matrix_J"], dtype=np.float64),
        np.asarray(instance["field_vector_h"], dtype=np.float64),
    )
    rows = r4.exact_oracle(coupling, field)
    optimum_energy = rows[0][0]
    optimum_rows = [
        spins for energy, spins in rows if abs(energy - optimum_energy) <= r4.ENERGY_TOL
    ]
    next_energies = [
        energy for energy, _ in rows if energy > optimum_energy + r4.ENERGY_TOL
    ]
    return {
        "gap": metric(min(next_energies) - optimum_energy) if next_energies else None,
        "optimum_count": len(optimum_rows),
        "optimum_energy": metric(optimum_energy),
        "optimum_spins": [list(spins) for spins in optimum_rows],
        "row_count": len(rows),
        "unique": len(optimum_rows) == 1,
    }


def classify(instance: dict[str, Any], oracle: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    latch = instance["raw_boundary"]["latch"]
    raw_spins = tuple(int(value) for value in latch["raw_spin_shadow"])
    optimums = {tuple(int(value) for value in row) for row in oracle["optimum_spins"]}
    raw_matches_any_optimum = raw_spins in optimums
    unique = bool(oracle["unique"])
    accepted = bool(latch["valid"])
    if not instance["interpretable_preoracle"]:
        outcome = "UNINTERPRETABLE"
    elif not unique:
        outcome = "NON_UNIQUE_OPTIMUM"
    elif accepted and raw_matches_any_optimum:
        outcome = "ACCEPTED_CORRECT"
    elif accepted:
        outcome = "ACCEPTED_INCORRECT"
    elif raw_matches_any_optimum:
        outcome = "RAW_CORRECT_BELOW_GATE"
    else:
        outcome = "RAW_INCORRECT"
    recomputed_raw_energy = r4.ising_energy(
        raw_spins,
        np.asarray(instance["coupling_matrix_J"], dtype=np.float64),
        np.asarray(instance["field_vector_h"], dtype=np.float64),
    )
    sealed_raw_energy = float(instance["raw_boundary"]["raw_spin_energy"])
    return outcome, {
        "accepted": accepted,
        "classification": outcome,
        "raw_energy_recomputed": metric(recomputed_raw_energy),
        "raw_energy_seal_matches": abs(recomputed_raw_energy - sealed_raw_energy)
        <= r4.ENERGY_TOL,
        "raw_matches_any_optimum": raw_matches_any_optimum,
        "raw_matches_unique_optimum": unique and raw_matches_any_optimum,
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "maximum": None, "mean": None, "median": None, "minimum": None, "values": []}
    if not np.all(np.isfinite(array)):
        raise ValueError("distribution contains a nonfinite value")
    return {
        "count": int(array.size),
        "maximum": metric(np.max(array)),
        "mean": metric(np.mean(array)),
        "median": metric(np.median(array)),
        "minimum": metric(np.min(array)),
        "values": [metric(value) for value in array.tolist()],
    }


def rate(count: int, denominator: int) -> float | None:
    return metric(count / denominator) if denominator else None


def control_aggregates(instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict_keys = {
        "carrier_content": "carrier_content_causal",
        "flat_geometry": "flat_geometry_changed_or_destroyed",
        "scrambled_geometry": "scrambled_geometry_changed_or_destroyed",
        "removed_transform": "no_transform_changed_or_destroyed",
        "removed_operator": "missing_phase_operator_changed_or_destroyed",
        "no_lock": "no_lock_changed_or_destroyed",
        "wrong_query": "wrong_query_changed_or_destroyed",
        "wrong_inverse": "wrong_inverse_failed",
        "omitted_step": "omitted_inverse_step_failed",
        "omitted_restoration": "omitted_restoration_failed",
        "samplewise_non_rank_one": "samplewise_non_rank_one",
    }
    count = len(instances)
    result: dict[str, Any] = {}
    for label, key in strict_keys.items():
        passed = sum(bool(item["controls"]["outcomes"][key]) for item in instances)
        result[label] = {"count": passed, "rate": rate(passed, count)}
    geometry_passed = sum(
        bool(item["controls"]["outcomes"]["flat_geometry_changed_or_destroyed"])
        and bool(item["controls"]["outcomes"]["scrambled_geometry_changed_or_destroyed"])
        for item in instances
    )
    strict_all = sum(bool(item["controls"]["all_pass"]) for item in instances)
    materiality_all = sum(
        bool(item["batch_law_control_materiality"]["all_pass"]) for item in instances
    )
    result["geometry_combined"] = {"count": geometry_passed, "rate": rate(geometry_passed, count)}
    result["strict_all_controls"] = {"count": strict_all, "rate": rate(strict_all, count)}
    result["batch_law_materiality_all_controls"] = {
        "count": materiality_all,
        "rate": rate(materiality_all, count),
    }
    return result


def aggregate_documents(instances: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        label: sum(item["classification"] == label for item in instances)
        for label in (
            "ACCEPTED_CORRECT",
            "ACCEPTED_INCORRECT",
            "RAW_CORRECT_BELOW_GATE",
            "RAW_INCORRECT",
            "UNINTERPRETABLE",
            "NON_UNIQUE_OPTIMUM",
        )
    }
    unique_count = sum(item["oracle"]["unique"] for item in instances)
    raw_unique_correct = sum(
        item["adjudication"]["raw_matches_unique_optimum"] for item in instances
    )
    non_unique_count = counts["NON_UNIQUE_OPTIMUM"]
    non_unique_raw_match = sum(
        (not item["oracle"]["unique"])
        and item["adjudication"]["raw_matches_any_optimum"]
        for item in instances
    )
    coherences = [
        value
        for item in instances
        for value in item["raw_boundary"]["latch"]["coherence"]
    ]
    per_site = {
        str(site): distribution(
            item["raw_boundary"]["latch"]["coherence"][site] for item in instances
        )
        for site in range(r4.SITE_COUNT)
    }
    lock_residuals = [
        item["raw_boundary"]["latch"]["residual_rad"]
        for item in instances
        if item["raw_boundary"]["latch"]["residual_rad"] is not None
    ]
    aggregate = {
        "batch_size": len(instances),
        "coherence": {
            "all_sites": distribution(coherences),
            "per_site": per_site,
            "per_instance_minimum": distribution(
                min(item["raw_boundary"]["latch"]["coherence"])
                for item in instances
            ),
        },
        "control_survival": control_aggregates(instances),
        "counts": counts,
        "lock_residual_rad_where_defined": distribution(lock_residuals),
        "non_unique_raw_optimum_agreement": {
            "count": non_unique_raw_match,
            "rate": rate(non_unique_raw_match, non_unique_count),
        },
        "outcome_rates_among_unique_optima": {
            "accepted_correct": rate(counts["ACCEPTED_CORRECT"], unique_count),
            "accepted_incorrect": rate(counts["ACCEPTED_INCORRECT"], unique_count),
            "raw_correct_below_gate": rate(counts["RAW_CORRECT_BELOW_GATE"], unique_count),
            "raw_incorrect": rate(counts["RAW_INCORRECT"], unique_count),
        },
        "overall_raw_optimum_agreement": {
            "count": raw_unique_correct,
            "rate": rate(raw_unique_correct, unique_count),
        },
        "restoration_error": distribution(
            item["measurements"]["restoration_max_abs_error"] for item in instances
        ),
        "reuse_input_error": distribution(
            item["measurements"]["reuse_input_max_abs_error"] for item in instances
        ),
        "reuse_restoration_error": distribution(
            item["measurements"]["reuse_restoration_max_abs_error"] for item in instances
        ),
        "samplewise_non_rank_one_residual": distribution(
            item["controls"]["measurements"]["samplewise_non_rank_one_residual"]
            for item in instances
        ),
        "unique_optimum_instance_count": unique_count,
    }
    return aggregate


def promotion_tests(
    criterion: dict[str, Any], aggregate: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, bool]:
    counts = aggregate["counts"]
    control_rates = aggregate["control_survival"]
    accepted_rate = aggregate["outcome_rates_among_unique_optima"]["accepted_correct"]
    return {
        "accepted_correct_count_min": counts["ACCEPTED_CORRECT"]
        >= criterion["accepted_correct_count_min"],
        "accepted_correct_rate_min": accepted_rate is not None
        and accepted_rate >= criterion["accepted_correct_rate_among_unique_min"],
        "accepted_incorrect_count_max": counts["ACCEPTED_INCORRECT"]
        <= criterion["accepted_incorrect_count_max"],
        "all_causality_restoration_reuse_and_controls_pass": (
            evidence["summary"]["all_instances_interpretable_preoracle"]
            and control_rates["strict_all_controls"]["count"] == aggregate["batch_size"]
        ),
        "batch_size_required": aggregate["batch_size"] == criterion["batch_size_required"],
        "unique_optimum_instance_count_min": aggregate["unique_optimum_instance_count"]
        >= criterion["unique_optimum_instance_count_min"],
        "uninterpretable_count_max": counts["UNINTERPRETABLE"]
        <= criterion["uninterpretable_count_max"],
    }


def adjudicate() -> tuple[dict[str, Any], dict[str, Any]]:
    authority = verify_experiment_authority()
    custody, contract, evidence = verify_preoracle_root()
    ledger = EventLedger([])
    ledger.add("prospective_external_experiment_authority_verified", authority)
    ledger.add("committed_preoracle_root_verified", {
        "preoracle_commit": PREORACLE_COMMIT,
        "preoracle_evidence_sha256": PREORACLE_EVIDENCE_SHA256,
        "preoracle_trace_sha256": PREORACLE_TRACE_SHA256,
    })
    ledger.add("oracle_opened_after_complete_batch_seal", {
        "batch_size": evidence["batch_size"],
        "ordered_batch_sha256": ORDERED_BATCH_SHA256,
    })

    adjudicated: list[dict[str, Any]] = []
    for preoracle in evidence["instances"]:
        oracle = exact_oracle_document(preoracle)
        outcome, adjudication = classify(preoracle, oracle)
        result = {**preoracle, "adjudication": adjudication, "classification": outcome, "oracle": oracle}
        adjudicated.append(result)
        ledger.add(f"instance_{preoracle['index']:02d}_oracle_adjudicated", {
            "classification": outcome,
            "instance_sha256": preoracle["instance_sha256"],
            "oracle": oracle,
        })

    aggregate = aggregate_documents(adjudicated)
    criterion = custody["promotion_criterion_frozen_before_execution"]
    tests = promotion_tests(criterion, aggregate, evidence)
    foundational = {
        "adapter_equivalence_exact": evidence["adapter_equivalence"]["pass"],
        "batch_law_materiality_controls_all_pass": (
            aggregate["control_survival"]["batch_law_materiality_all_controls"]["count"]
            == aggregate["batch_size"]
        ),
        "frozen_machine_exact": evidence["frozen_machine"]["machine_sha256"]
        == FROZEN_MACHINE_SHA256,
        "native_no_smuggle_pass": evidence["native_call_path_proof"]["pass"],
        "oracle_after_complete_preoracle_seal": True,
        "preoracle_oracle_call_count_zero": evidence["oracle_absence_proof"]["oracle_calls"] == 0,
        "prospective_decision_authority_exact": authority[
            "decision_law_frozen_before_execution"
        ],
        "raw_energy_seals_recompute": all(
            item["adjudication"]["raw_energy_seal_matches"] for item in adjudicated
        ),
        "restoration_success_all": evidence["summary"]["restoration_success_count"]
        == aggregate["batch_size"],
        "reuse_success_all": evidence["summary"]["reuse_success_count"]
        == aggregate["batch_size"],
    }
    promotion_pass = all(tests.values()) and all(foundational.values())
    credible_count = (
        aggregate["counts"]["ACCEPTED_CORRECT"]
        + aggregate["counts"]["RAW_CORRECT_BELOW_GATE"]
    )
    if promotion_pass:
        decision = VERIFIED
    elif all(foundational.values()) and credible_count > 0:
        decision = PARTIAL
    else:
        decision = NOT_ESTABLISHED

    ledger.add("aggregate_adjudication_sealed", {
        "aggregate": aggregate,
        "decision": decision,
        "experiment_authority": authority,
        "promotion_tests": tests,
    })
    trace = ledger.document()
    result = {
        "aggregate": aggregate,
        "claim_ceiling": held.CLAIM_CEILING,
        "decision": decision,
        "foundational_tests": foundational,
        "freeze_commit": FREEZE_COMMIT,
        "frozen_machine_sha256": FROZEN_MACHINE_SHA256,
        "instances": adjudicated,
        "oracle_order": {
            "oracle_open_event_sequence": 2,
            "preoracle_commit": PREORACLE_COMMIT,
            "preoracle_evidence_sha256": PREORACLE_EVIDENCE_SHA256,
            "preoracle_trace_sha256": PREORACLE_TRACE_SHA256,
        },
        "ordered_batch_sha256": ORDERED_BATCH_SHA256,
        "predecessor_results_preserved": [
            "CATALYTIC_WAVEFORM_ISING_COMPUTATION_VERIFIED",
            "CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_PARTIAL",
        ],
        "promotion_criterion": criterion,
        "promotion_pass": promotion_pass,
        "promotion_tests": tests,
        "schema": RESULT_SCHEMA,
        "source": {
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    return result, trace


def report_text(result: dict[str, Any], trace: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    counts = aggregate["counts"]
    coherence = aggregate["coherence"]["all_sites"]
    restoration = aggregate["restoration_error"]
    reuse_input = aggregate["reuse_input_error"]
    reuse_restore = aggregate["reuse_restoration_error"]
    controls = aggregate["control_survival"]
    tests_failed = [name for name, passed in result["promotion_tests"].items() if not passed]
    return f"""# Catalytic Waveform-Ising Frozen Batch Generalization

**Decision:** `{result['decision']}`

**Claim ceiling:** `{result['claim_ceiling']}`

## Custody and ordering

- Freeze commit: `{FREEZE_COMMIT}`
- Pre-oracle evidence commit: `{PREORACLE_COMMIT}`
- Ordered batch SHA-256: `{ORDERED_BATCH_SHA256}`
- Pre-oracle evidence SHA-256: `{PREORACLE_EVIDENCE_SHA256}`
- Pre-oracle trace SHA-256: `{PREORACLE_TRACE_SHA256}`
- Prospective authority SHA-256: `{AUTHORITY_SHA256}` ({AUTHORITY_BYTES} bytes)
- Oracle calls in the pre-oracle runner: `0`
- Final oracle trace hash: `{trace['final_event_sha256']}`

All sixteen waveform executions, raw projections, restorations, restored-carrier reuses,
and controls were sealed before the exact 32-state oracles were opened.

## Outcomes

```text
batch size                     {aggregate['batch_size']}
unique optima                  {aggregate['unique_optimum_instance_count']}
accepted correct               {counts['ACCEPTED_CORRECT']}
accepted incorrect             {counts['ACCEPTED_INCORRECT']}
raw correct below gate         {counts['RAW_CORRECT_BELOW_GATE']}
raw incorrect                  {counts['RAW_INCORRECT']}
non-unique optimum             {counts['NON_UNIQUE_OPTIMUM']}
non-unique raw matches         {aggregate['non_unique_raw_optimum_agreement']['count']} / {counts['NON_UNIQUE_OPTIMUM']}
uninterpretable                {counts['UNINTERPRETABLE']}
overall raw optimum agreement  {aggregate['overall_raw_optimum_agreement']['count']} / {aggregate['unique_optimum_instance_count']}
```

The prospectively frozen promotion gate did not pass. Failed promotion checks:
`{', '.join(tests_failed) if tests_failed else 'none'}`.

## Coherence and restoration

```text
coherence min/median/mean/max   {coherence['minimum']} / {coherence['median']} / {coherence['mean']} / {coherence['maximum']}
restoration error max           {restoration['maximum']}
reuse input error max           {reuse_input['maximum']}
reuse restoration error max     {reuse_restore['maximum']}
```

## Controls

```text
batch-law materiality all       {controls['batch_law_materiality_all_controls']['count']} / {aggregate['batch_size']}
strict all controls              {controls['strict_all_controls']['count']} / {aggregate['batch_size']}
carrier content                  {controls['carrier_content']['count']} / {aggregate['batch_size']}
geometry combined               {controls['geometry_combined']['count']} / {aggregate['batch_size']}
removed transform               {controls['removed_transform']['count']} / {aggregate['batch_size']}
removed operator                {controls['removed_operator']['count']} / {aggregate['batch_size']}
no lock                         {controls['no_lock']['count']} / {aggregate['batch_size']}
wrong query                     {controls['wrong_query']['count']} / {aggregate['batch_size']}
wrong inverse                   {controls['wrong_inverse']['count']} / {aggregate['batch_size']}
omitted inverse step            {controls['omitted_step']['count']} / {aggregate['batch_size']}
omitted restoration             {controls['omitted_restoration']['count']} / {aggregate['batch_size']}
```

The stricter predecessor control requires both a material history change and a material
complex-response change. Its failures are preserved. The separately frozen batch law
accepts a material history or response change and is reported independently; no machine
constant, query, threshold, instance, or result was altered.

## Concrete evidence repairs

- The first pre-oracle qualification incorrectly treated strict predecessor-control
  failure as uninterpretable even when the frozen batch contract's material-history-or-
  response law passed. The qualification-only distinction was repaired and every
  pre-oracle trajectory was rerun before any oracle opened. Strict failures remain.
- The exact user-supplied experiment contract, which prospectively defined the three-way
  decision lattice, is now preserved byte-for-byte as `BATCH_EXPERIMENT_AUTHORITY.txt`
  and bound by its byte count and SHA-256.
- Non-unique optima are explicitly separated from unique successes and failures.

## Interpretation

This batch measures bounded software generalization of the unchanged carrier-causal
waveform machine. It does not establish scale, computational advantage, physical
computation, hardware bit replacement, or a Wall crossing.
"""


def manifest_document() -> dict[str, Any]:
    paths = [
        PACKAGE_DIR / ".gitattributes",
        AUTHORITY_FILE,
        PACKAGE_DIR / "BATCH_INSTANCE_FREEZE.md",
        CUSTODY_FILE,
        PACKAGE_DIR / "batch_instance_freezer.py",
        PREORACLE_CONTRACT_FILE,
        PREORACLE_EVIDENCE_FILE,
        PREORACLE_TRACE_FILE,
        PREORACLE_SEAL_FILE,
        Path(__file__).resolve(),
        RESULTS_FILE,
        TRACE_FILE,
        REPORT_FILE,
    ]
    records = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(PACKAGE_DIR).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    root_payload = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n"
        for row in sorted(records, key=lambda item: item["path"])
    ).encode("utf-8")
    return {
        "file_count": len(records),
        "files": records,
        "package_root_sha256": sha256_bytes(root_payload),
        "schema": MANIFEST_SCHEMA,
        "total_bytes": sum(row["bytes"] for row in records),
    }


def build_package() -> dict[str, Any]:
    result, trace = adjudicate()
    write_json(TRACE_FILE, trace)
    result["oracle_trace_sha256"] = sha256_file(TRACE_FILE)
    write_json(RESULTS_FILE, result)
    write_atomic(REPORT_FILE, report_text(result, trace).encode("utf-8"))
    write_json(MANIFEST_FILE, manifest_document())
    return result


def verify_package() -> dict[str, Any]:
    result, trace = adjudicate()
    if TRACE_FILE.read_bytes() != canonical_bytes(trace):
        raise ValueError("committed oracle trace does not reproduce")
    result["oracle_trace_sha256"] = sha256_file(TRACE_FILE)
    if RESULTS_FILE.read_bytes() != canonical_bytes(result):
        raise ValueError("committed batch results do not reproduce")
    if REPORT_FILE.read_bytes() != report_text(result, trace).encode("utf-8"):
        raise ValueError("committed batch report does not reproduce")
    manifest = manifest_document()
    if MANIFEST_FILE.read_bytes() != canonical_bytes(manifest):
        raise ValueError("committed batch manifest does not reproduce")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    result = build_package() if args.command == "build" else verify_package()
    aggregate = result["aggregate"]
    print(json.dumps({
        "accepted_correct": aggregate["counts"]["ACCEPTED_CORRECT"],
        "decision": result["decision"],
        "raw_optimum_agreement": aggregate["overall_raw_optimum_agreement"],
        "strict_control_survival": aggregate["control_survival"]["strict_all_controls"],
        "unique_optima": aggregate["unique_optimum_instance_count"],
    }, sort_keys=True))
    return 0 if result["decision"] != NOT_ESTABLISHED else 1


if __name__ == "__main__":
    raise SystemExit(main())

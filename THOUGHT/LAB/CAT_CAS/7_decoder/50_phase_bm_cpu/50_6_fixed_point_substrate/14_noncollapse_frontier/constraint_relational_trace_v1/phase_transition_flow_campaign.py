from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from constraint_relational_trace_v1.adaptive_phase_logit_flow import (
        integrate_adaptive_phase_logit_flow,
    )
    from constraint_relational_trace_v1.phase_transition_corpus import (
        certify_phase_transition_case,
    )
else:
    from .adaptive_phase_logit_flow import integrate_adaptive_phase_logit_flow
    from .phase_transition_corpus import certify_phase_transition_case


@dataclass(frozen=True)
class PhaseTransitionFlowCaseResult:
    seed: int
    semantic_digest: str
    expected_status: str
    witness_count_reference_only: int
    terminal_boundary_status: str
    terminal_solution_verified: bool
    first_passage_time: float | None
    satisfiable_missed: bool
    false_positive: bool
    invalid_carrier: bool
    function_evaluations: int
    maximum_long_memory: float
    maximum_pair_log_ratio_magnitude: float
    phase_trajectory_length_lower_bound: float
    native_trajectory_length_lower_bound: float
    terminal_clause_satisfaction_margin: float


def build_phase_transition_flow_record(
    *,
    seed_count: int = 16,
    seed_start: int = 0,
    variable_count: int = 12,
    clause_count: int = 51,
    fixed_deadline: float = 3.0,
    gradient_mode: str = "exact_product",
) -> dict[str, object]:
    results: list[PhaseTransitionFlowCaseResult] = []
    satisfiable_cases = 0
    satisfiable_terminal_witnesses = 0
    satisfiable_misses = 0
    unsatisfiable_cases = 0
    unsat_false_positives = 0
    invalid_carriers = 0

    for seed in range(seed_start, seed_start + seed_count):
        case = certify_phase_transition_case(variable_count, clause_count, seed)
        run = integrate_adaptive_phase_logit_flow(
            case.holo,
            fixed_deadline=fixed_deadline,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-8,
            maximum_step=2.0e-2,
            solver_method="DOP853",
            gradient_mode=gradient_mode,
        )
        invalid = run.status.startswith("INVALID_CARRIER")
        false_positive = (
            case.expected_status == "UNSAT" and run.terminal_solution_verified
        )
        missed = (
            case.expected_status == "SAT" and not run.terminal_solution_verified
        )
        if case.expected_status == "SAT":
            satisfiable_cases += 1
            satisfiable_terminal_witnesses += int(run.terminal_solution_verified)
            satisfiable_misses += int(missed)
        else:
            unsatisfiable_cases += 1
            unsat_false_positives += int(false_positive)
        invalid_carriers += int(invalid)
        results.append(
            PhaseTransitionFlowCaseResult(
                seed=seed,
                semantic_digest=case.semantic_digest,
                expected_status=case.expected_status,
                witness_count_reference_only=case.witness_count_reference_only,
                terminal_boundary_status=run.status,
                terminal_solution_verified=run.terminal_solution_verified,
                first_passage_time=run.first_passage_time,
                satisfiable_missed=missed,
                false_positive=false_positive,
                invalid_carrier=invalid,
                function_evaluations=run.function_evaluations,
                maximum_long_memory=run.maximum_long_memory,
                maximum_pair_log_ratio_magnitude=(
                    run.maximum_pair_log_ratio_magnitude
                ),
                phase_trajectory_length_lower_bound=(
                    run.phase_trajectory_length_lower_bound
                ),
                native_trajectory_length_lower_bound=(
                    run.native_trajectory_length_lower_bound
                ),
                terminal_clause_satisfaction_margin=(
                    run.terminal_clause_satisfaction_margin
                ),
            )
        )

    passed = (
        satisfiable_misses == 0
        and unsat_false_positives == 0
        and invalid_carriers == 0
    )
    return {
        "schema": "PHASE_TRANSITION_FLOW_CAMPAIGN_V1",
        "status": (
            "PHASE_TRANSITION_FLOW_CAMPAIGN_ALL_CLASSIFIED_AT_FIXED_DEADLINE"
            if passed
            else "PHASE_TRANSITION_FLOW_CAMPAIGN_EXPOSED_MISSES_OR_INVALID_CARRIERS"
        ),
        "seed_count": seed_count,
        "seed_start": seed_start,
        "variable_count": variable_count,
        "clause_count": clause_count,
        "fixed_deadline": fixed_deadline,
        "gradient_mode": gradient_mode,
        "satisfiable_cases": satisfiable_cases,
        "satisfiable_terminal_witnesses": satisfiable_terminal_witnesses,
        "satisfiable_misses": satisfiable_misses,
        "unsatisfiable_cases": unsatisfiable_cases,
        "unsat_false_positives": unsat_false_positives,
        "invalid_carriers": invalid_carriers,
        "cases": [asdict(result) for result in results],
        "claim_boundary": {
            "uniform_polynomial_deadline": "NOT_ESTABLISHED",
            "polynomial_native_trajectory_length": "NOT_ESTABLISHED",
            "p_equals_np": "NOT_PROVEN",
        },
    }


def main() -> int:
    record = build_phase_transition_flow_record()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_transition_flow_campaign.json"
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "satisfiable_cases": record["satisfiable_cases"],
                "satisfiable_terminal_witnesses": record[
                    "satisfiable_terminal_witnesses"
                ],
                "satisfiable_misses": record["satisfiable_misses"],
                "unsat_false_positives": record["unsat_false_positives"],
                "invalid_carriers": record["invalid_carriers"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if record["unsat_false_positives"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
  exit 2
fi
build=$(realpath -e -- "$1")
case "$build" in
  /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
    echo "RAM-backed M255 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M255 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/grassmann_quartic_cumulant_hodge_rank_growth.py"
reference="$here/grassmann_quartic_cumulant_hodge_rank_growth_separate_reference.py"
qualifier="$here/qualify_grassmann_quartic_cumulant_hodge_rank_growth.sh"
sealed_raw="$here/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_RAW_RESULTS.json"
sealed_ref="$here/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_SEPARATE_REFERENCE.json"
sealed_result="$here/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_RESULTS.json"
raw="$build/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_RAW_RESULTS.json"
ref="$build/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_SEPARATE_REFERENCE.json"
result="$build/GRASSMANN_QUARTIC_CUMULANT_HODGE_RANK_GROWTH_RESULTS.json"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$production" >"$raw"
"${run_env[@]}" nice -n 10 ionice -c 3 python3 "$reference" >"$ref"

"${run_env[@]}" python3 - \
  "$raw" "$ref" "$result" "$production" "$reference" "$qualifier" <<'PY'
import ast,hashlib,json,sys
from pathlib import Path

raw_path,ref_path,result_path=map(Path,sys.argv[1:4])
sources=list(map(Path,sys.argv[4:7]))
raw=json.loads(raw_path.read_text())
ref=json.loads(ref_path.read_text())

if raw.get("milestone")!=255 or ref.get("milestone")!=255:
    raise SystemExit("M255 milestone mismatch")
if raw.get("classification")!="INDEPENDENTLY_VERIFIED_STRICT_SCOPE":
    raise SystemExit("M255 verification classification mismatch")
if raw.get("verification_level")!="SEPARATE_REFERENCE_PARITY":
    raise SystemExit("M255 verification level mismatch")
if raw.get("restoration_class")!="EXACT_ALGEBRAIC_RESTORATION":
    raise SystemExit("M255 restoration class mismatch")
if not all(raw.get("controls",{}).values()) or not all(ref.get("controls",{}).values()):
    raise SystemExit("M255 control failure")

raw_cases={case["run_kind"]:case for case in raw["cases"]}
ref_cases={case["run_kind"]:case for case in ref["cases"]}
expected_keys={"PRIMARY_M4","PRIMARY_M6","PRIMARY_M8","REUSE_M8","FRESH_M8"}
if set(raw_cases)!=set(ref_cases) or set(raw_cases)!=expected_keys:
    raise SystemExit("M255 case-key mismatch")
for key in expected_keys:
    production=raw_cases[key]; oracle=ref_cases[key]
    for field in (
        "port_count","generation","program_id","selected_connected_top_degree_cumulant",
        "hodge_normalization_scalar","same_pair_quartic_backings",
        "canonical_after_restoration","baseline_reload_used",
    ):
        if production[field]!=oracle[field]:
            raise SystemExit(f"M255 independent mismatch {key} {field}")
    if not production["same_pair_quartic_backings"] or not production["canonical_after_restoration"] or production["baseline_reload_used"]:
        raise SystemExit(f"M255 restoration/reuse failure {key}")

expected={
    "PRIMARY_M4":([1,4],[2,1]),
    "PRIMARY_M6":([-2,27],[-3,1]),
    "PRIMARY_M8":([14,625],[5,1]),
    "REUSE_M8":([15,2048],[16,1]),
    "FRESH_M8":([15,2048],[16,1]),
}
for key,(cumulant,scalar) in expected.items():
    if raw_cases[key]["selected_connected_top_degree_cumulant"]!=cumulant:
        raise SystemExit(f"M255 exact cumulant mismatch {key}")
    if raw_cases[key]["hodge_normalization_scalar"]!=scalar:
        raise SystemExit(f"M255 exact scalar mismatch {key}")
if raw_cases["PRIMARY_M6"]["rank1_quartic_plucker_dual_square_witness"]!=[2,1]:
    raise SystemExit("M255 exact rank-two witness mismatch")
if raw_cases["PRIMARY_M6"]["rank1_quartic_plucker_dual_square_support"]!=[0,1,4,5]:
    raise SystemExit("M255 exact rank-two witness support mismatch")
if raw_cases["PRIMARY_M4"]["rank1_quartic_plucker_dual_square_applicable"] or raw_cases["PRIMARY_M4"]["rank1_quartic_plucker_dual_square_witness"]!=[0,1]:
    raise SystemExit("M255 M4 Plucker applicability mismatch")
if not raw_cases["PRIMARY_M6"]["rank1_quartic_plucker_dual_square_applicable"]:
    raise SystemExit("M255 M6 Plucker applicability missing")
if ref["rank_obstruction"]!={
    "m6_connected_degree6":[-2,27],
    "m6_rank_certificate":{
        "decomposable_summand_count":2,
        "dual_square_witness":[2,1],
        "dual_square_witness_support":[0,1,4,5],
        "exact_rank":2,
        "rank_lower_bound_from_dual_plucker_square":2,
        "rank_upper_bound_from_displayed_sum":2,
    },
    "m6_quartic_sum_exact_rank":2,
    "m8_connected_degree8":[14,625],
}:
    raise SystemExit("M255 independent obstruction certificate mismatch")
if [ref_cases[f"PRIMARY_M{width}"]["maximum_nonzero_connected_degree"] for width in (4,6,8)] != [4,6,8]:
    raise SystemExit("M255 independent connected-degree law mismatch")
if raw_cases["REUSE_M8"]["generation"]!=2 or raw_cases["FRESH_M8"]["generation"]!=1:
    raise SystemExit("M255 reuse generation mismatch")
if raw_cases["REUSE_M8"]["selected_connected_top_degree_cumulant"]!=raw_cases["FRESH_M8"]["selected_connected_top_degree_cumulant"]:
    raise SystemExit("M255 restored/fresh boundary mismatch")

law=raw["resource_law"]
if law["accepted_path_full_even_signature_materialized"] or law["accepted_path_relation_table_or_assignment_expansion"]:
    raise SystemExit("M255 compact accepted-path law mismatch")
if law["accepted_carrier_pair_cells"]!={"4":6,"6":15,"8":28}:
    raise SystemExit("M255 pair-cell law mismatch")
if law["accepted_carrier_quartic_coefficient_cells"]!={"4":1,"6":2,"8":3}:
    raise SystemExit("M255 quartic-cell law mismatch")
if law["strongest_fixed_fixture_classical_baseline"]!="PUBLIC_O1_CLOSED_CERTIFICATES":
    raise SystemExit("M255 fixed comparator mismatch")
if law["strongest_transferable_normalization_scalar_baseline"]!="TWO_STATE_MONOMER_DIMER_CONTINUANT_O_M_WORK_O1_LIVE_FIELD_STATE":
    raise SystemExit("M255 transferable normalization comparator mismatch")
if law["strongest_implemented_descriptor_level_selected_cumulant_baseline"]!="STREAMED_PFAFFIAN_COEFFICIENT_AND_EVEN_SET_PARTITION_CUMULANT":
    raise SystemExit("M255 descriptor-level comparator mismatch")
if law["descriptor_level_selected_cumulant_optimality_claimed"]:
    raise SystemExit("M255 comparator optimality overclaim")
if not law["accepted_projection_is_not_the_strongest_classical_baseline"]:
    raise SystemExit("M255 comparator ceiling missing")
if ref["classical_baselines"]["continuant_state_field_cells"]!=2:
    raise SystemExit("M255 independent continuant resource mismatch")
if any(case["retained_final_boundary_field_cells_during_inverse"]!=2 for case in raw["cases"]):
    raise SystemExit("M255 retained final-boundary cell count mismatch")
if raw["control_applicability"]!={
    "reason":"NATIVE_CUMULANT_INTERSECTION_ADDITION_COMMUTES;_ONLY_SLOT_RECEIPT_ORDER_IS_CUSTODY_RELEVANT",
    "reordered_inverse_algebraic_failure_applicable":False,
}:
    raise SystemExit("M255 reordered-inverse applicability mismatch")
if raw["obstruction"]!={
    "connected_degree6_nonzero_at_declared_m6":True,
    "connected_degree8_nonzero_at_declared_m8":True,
    "first_declared_escape_port_count":6,
    "rank1_degree4_chart_fails_at_declared_m6":True,
    "rank1_degree4_chart_vacuously_sufficient_at_declared_m4":True,
    "route_disposition":"RETIRE_GAUSSIAN_PLUS_RANK1_QUARTIC_FIXED_DEGREE_CHART_AFTER_M8_CONFIRMATION",
}:
    raise SystemExit("M255 obstruction disposition mismatch")
if any(value is not False for key,value in raw["claim_limits"].items() if key.endswith("_established")):
    raise SystemExit("M255 claim ceiling promoted")

production_tree=ast.parse(sources[0].read_text())
reference_tree=ast.parse(sources[1].read_text())
reference_imports={node.module for node in ast.walk(reference_tree) if isinstance(node,ast.ImportFrom) and node.module}
if any("grassmann" in module or "catvm" in module for module in reference_imports):
    raise SystemExit("M255 standalone imported production")
production_text=sources[0].read_text()
for forbidden in ("itertools.product", "cartesian_product", "assignment_table", "truth_table"):
    if forbidden in production_text:
        raise SystemExit(f"M255 accepted source forbidden expansion marker {forbidden}")
if not any(isinstance(node,ast.FunctionDef) and node.name=="pfaffian_from_pairs" for node in ast.walk(production_tree)):
    raise SystemExit("M255 compact Pfaffian evaluator missing")

source_dependencies={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
result={
    "milestone":255,
    "claim":raw["claim"],
    "classification":raw["classification"],
    "verification_level":raw["verification_level"],
    "restoration_class":raw["restoration_class"],
    "verification":{
        "production_compact_projection_matches_standalone_full_exterior_oracle":True,
        "independent_reference_carrier_restoration_and_reuse":True,
        "exact_boundaries":expected,
        "rank_obstruction":ref["rank_obstruction"],
        "maximum_connected_degrees":{"4":4,"6":6,"8":8},
    },
    "cases":raw["cases"],
    "controls":{"production":raw["controls"],"independent":ref["controls"]},
    "control_applicability":raw["control_applicability"],
    "resource_law":raw["resource_law"],
    "classical_baselines":ref["classical_baselines"],
    "obstruction":raw["obstruction"],
    "claim_limits":raw["claim_limits"],
    "source_dependencies":source_dependencies,
}
result_path.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
PY

cmp -s "$raw" "$sealed_raw" || {
  echo "M255 raw seal mismatch" >&2
  exit 1
}
cmp -s "$ref" "$sealed_ref" || {
  echo "M255 reference seal mismatch" >&2
  exit 1
}
cmp -s "$result" "$sealed_result" || {
  echo "M255 result seal mismatch" >&2
  exit 1
}

echo "M255 GRASSMANN QUARTIC CUMULANT HODGE RANK GROWTH QUALIFIED"

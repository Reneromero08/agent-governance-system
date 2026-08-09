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
    echo "RAM-backed M257 build forbidden" >&2
    exit 2
    ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
  tmpfs|ramfs)
    echo "RAM-backed M257 filesystem forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/software_phase_forward_shadow_dominance.py"
reference="$here/software_phase_forward_shadow_dominance_separate_reference.py"
sealed_result="$here/SOFTWARE_PHASE_FORWARD_SHADOW_DOMINANCE_RESULTS.json"
sealed_reference="$here/SOFTWARE_PHASE_FORWARD_SHADOW_DOMINANCE_SEPARATE_REFERENCE.json"
result="$build/SOFTWARE_PHASE_FORWARD_SHADOW_DOMINANCE_RESULTS.json"
reference_result="$build/SOFTWARE_PHASE_FORWARD_SHADOW_DOMINANCE_SEPARATE_REFERENCE.json"

mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"
run_env=(
  env
  TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp"
  XDG_CACHE_HOME="$build/xdg-cache"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$build/pycache"
)

"${run_env[@]}" python3 "$production" "$here" "$result"
"${run_env[@]}" python3 "$reference" "$here" "$reference_result"

"${run_env[@]}" python3 - "$here" "$result" "$reference_result" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root=Path(sys.argv[1])
production=json.loads(Path(sys.argv[2]).read_text())
reference=json.loads(Path(sys.argv[3]).read_text())

assert production["milestone"] == reference["milestone"] == 257
assert production["result"] == "PASS_EXACT_SOFTWARE_PHASE_TRANSACTION_FORWARD_SHADOW_STRICT_SCOPE"
assert production["classification"] == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
assert production["verification_level"] == reference["verification_level"] == "SEPARATE_REFERENCE_PARITY"
assert production["restoration_classification"] == reference["restoration_classification"] == "NO_RESTORATION_CLAIM"

pp={item["package"]:item for item in production["packages"]}
rp={item["package"]:item for item in reference["packages"]}
assert set(pp) == set(rp)
assert len(pp) == 5
common=(
    "sealed_result_sha256", "sealed_service_sha256", "primary_run_kind",
    "boundary_sha256", "source_order", "accepted_forward_state_shape_upper_bound",
    "shadow_forward_state_shape_upper_bound",
    "reported_transaction_only_inverse_work_vector_omitted_by_shadow",
)
for name in sorted(pp):
    for key in common:
        assert pp[name][key] == rp[name][key], (name,key)
    order=pp[name]["source_order"]
    assert order["forward_line"] < order["projection_line"] < order["inverse_line"] < order["release_line"] < order["response_line"]
    assert pp[name]["accepted_forward_state_shape_upper_bound"] == pp[name]["shadow_forward_state_shape_upper_bound"]
    assert pp[name]["reported_forward_work_vector_reused_by_shadow"]
    assert all(value > 0 for value in pp[name]["reported_transaction_only_inverse_work_vector_omitted_by_shadow"].values())
    assert pp[name]["same_backing_restoration"] is True
    assert pp[name]["baseline_reload_used"] is False

aggregate=production["aggregate"]
assert aggregate["algebraically_distinct_packages"] == 5
assert all(value is True for key,value in aggregate.items() if key.startswith("all_"))
assert aggregate["same_domain_software_advantage_established"] is False
assert aggregate["distinct_phase_resource_established"] is False
assert aggregate["small_wall_crossed"] is False
assert production["theorem"]["does_not_require_summing_heterogeneous_operation_counters"] is True
assert any("AUXILIARY_STATE_SECRET_INPUT_OR_ORACLE_ACCESS_IS_GRANTED_EQUALLY" in item for item in production["theorem"]["assumptions"])
assert reference["theorem_certificate"]["full_atomic_transaction_is_a_different_task"] is True
assert reference["applicability_controls"]["private_input_denied_to_shadow"] == "INAPPLICABLE_ACCESS_MISMATCH"
assert reference["applicability_controls"]["restoration_and_reuse_required_as_output"] == "INAPPLICABLE_FULL_TRANSACTION_TASK"
assert reference["applicability_controls"]["dense_only_comparator_substitution"] == "REJECTED_STRONGER_COMPACT_BASELINE_REQUIRED"
assert all(value is False for value in production["claim_limits"].values())
assert all(value is False for value in reference["claim_limits"].values())
assert production["resource_law"]["comparison_is_componentwise_not_a_sum_of_heterogeneous_counters"] is True
assert production["resource_law"]["existing_stronger_package_specific_baselines_remain_authoritative"] is True
assert production["resource_law"]["whole_process_python_object_allocator_socket_hash_serialization_timing_and_rss_costs_complete"] is False

for document in (production,reference):
    for relative,expected in document["source_dependencies"].items():
        actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        assert actual == expected, relative
PY

if [[ ! -f "$sealed_result" || ! -f "$sealed_reference" ]]; then
  echo "M257 sealed outputs missing" >&2
  exit 2
fi
cmp --silent "$result" "$sealed_result"
cmp --silent "$reference_result" "$sealed_reference"

echo "M257 SOFTWARE PHASE FORWARD SHADOW DOMINANCE QUALIFIED"

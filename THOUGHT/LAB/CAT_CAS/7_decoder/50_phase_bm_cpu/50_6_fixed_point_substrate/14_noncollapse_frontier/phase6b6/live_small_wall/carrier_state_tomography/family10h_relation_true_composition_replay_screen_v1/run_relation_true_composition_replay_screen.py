#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CARRIER_ROOT = HERE.parent
REPLAY_RUNNER = (
    CARRIER_ROOT
    / "family10h_relation_composition_replay_exclusion_discovery_v1"
    / "run_relation_composition_replay_exclusion_discovery.py"
)

spec = importlib.util.spec_from_file_location("composition_replay_base", REPLAY_RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load replay runner: {REPLAY_RUNNER}")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

ORIGINAL_REPLAY_PATCH = replay.patch_runtime_for_composition_loop
REPLAY_TARGET_SCRIPT = replay.target_script

RUN_ID = "family10h_relation_true_composition_replay_screen_v1_0"
PACKAGE_ID = "family10h_relation_true_composition_replay_screen_v1"
SOURCE_ROOT = HERE / "generated_source"
SCHEDULE_DIR = SOURCE_ROOT / "TRUE_COMPOSITION_REPLAY_SCREEN_SCHEDULES"
ATTEMPT_LABEL = os.environ.get("FAMILY10H_DISCOVERY_ATTEMPT_LABEL", "attempt_1")
if any(sep in ATTEMPT_LABEL for sep in ("/", "\\", ":")) or not ATTEMPT_LABEL:
    raise RuntimeError(f"invalid attempt label: {ATTEMPT_LABEL!r}")
RUN_ROOT = CARRIER_ROOT / "runs" / RUN_ID / ATTEMPT_LABEL
LOCAL_PACKAGE = Path("C:/tmp") / f"{RUN_ID}_source_package.tar.gz"
LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{RUN_ID}_remote_root.tar.gz"
LOCAL_ARCHIVE = RUN_ROOT / "DISCOVERY_TARGET_ROOT.tar.gz"
SUMMARY_JSON = HERE / "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_SUMMARY.json"
SUMMARY_MD = HERE / "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_SUMMARY.md"


def configure_replay() -> None:
    replay.RUN_ID = RUN_ID
    replay.PACKAGE_ID = PACKAGE_ID
    replay.HERE = HERE
    replay.SOURCE_ROOT = SOURCE_ROOT
    replay.SCHEDULE_DIR = SCHEDULE_DIR
    replay.RUN_ROOT = RUN_ROOT
    replay.LOCAL_PACKAGE = LOCAL_PACKAGE
    replay.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    replay.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    replay.SUMMARY_JSON = SUMMARY_JSON
    replay.SUMMARY_MD = SUMMARY_MD
    replay.target_script = target_script
    replay.base.RUN_ID = RUN_ID
    replay.base.REMOTE_ROOT = f"{replay.base.REMOTE_BASE}/{RUN_ID}"
    replay.base.REMOTE_PACKAGE = f"{replay.base.REMOTE_BASE}/{RUN_ID}_source_package.tar.gz"
    replay.base.REMOTE_ARCHIVE = f"{replay.base.REMOTE_BASE}/{RUN_ID}_remote_root.tar.gz"
    replay.base.OWNER_MARKER = f".{RUN_ID}_owner"
    replay.base.HERE = HERE
    replay.base.CARRIER_ROOT = CARRIER_ROOT
    replay.base.SOURCE_ROOT = SOURCE_ROOT
    replay.base.SCHEDULE_DIR = SCHEDULE_DIR
    replay.base.RUN_ROOT = RUN_ROOT
    replay.base.LOCAL_PACKAGE = LOCAL_PACKAGE
    replay.base.LOCAL_TMP_ARCHIVE = LOCAL_TMP_ARCHIVE
    replay.base.LOCAL_ARCHIVE = LOCAL_ARCHIVE
    replay.base.SUMMARY_JSON = SUMMARY_JSON
    replay.base.SUMMARY_MD = SUMMARY_MD
    replay.base.target_script = target_script
    replay.base.make_rows = replay.make_rows


def target_script() -> str:
    return (
        REPLAY_TARGET_SCRIPT()
        .replace("COMPOSITION_REPLAY_EXCLUSION_SCHEDULES", "TRUE_COMPOSITION_REPLAY_SCREEN_SCHEDULES")
        .replace("RELATION_COMPOSITION_REPLAY_EXCLUSION", "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN")
    )


def patch_runtime_for_true_physical_composition() -> None:
    ORIGINAL_REPLAY_PATCH()
    runtime = SOURCE_ROOT / "relation_spatial_runtime.c"
    text = runtime.read_text(encoding="utf-8")
    physical_child_single_prepare = (
        "                prep.bank_a_work = row.bank_a_work;\n"
        "                prep.bank_b_work = row.bank_b_work;\n"
        "                prep.relation = row.r_prepare;\n"
        "                prep.source_order = row.source_order;\n"
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "                shared->source_cpu_after = current_cpu_checked();\n"
    )
    physical_child_composition_prepare = (
        "                prep.bank_a_work = row.bank_a_work;\n"
        "                prep.bank_b_work = row.bank_b_work;\n"
        "                prep.relation = row.r_prepare;\n"
        "                prep.source_order = row.source_order;\n"
        "                prep.cyclic_origin = row.cyclic_origin;\n"
        "                if (compose_r0_r1_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "                } else if (compose_r1_r0_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "                } else if (gapped_r0_r1_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_gapped(prep, &shared->state, RELATION_SPATIAL_R0, RELATION_SPATIAL_R1);\n"
        "                } else if (gapped_r1_r0_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_gapped(prep, &shared->state, RELATION_SPATIAL_R1, RELATION_SPATIAL_R0);\n"
        "                } else if (balanced_alt_a_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_balanced_alt(prep, &shared->state, 0u);\n"
        "                } else if (balanced_alt_b_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_balanced_alt(prep, &shared->state, 1u);\n"
        "                } else if (neutral_compose_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_neutral(prep, &shared->state);\n"
        "                } else if (random_compose_control(row.control)) {\n"
        "                    shared->preparation_ok = relation_spatial_prepare_composition_randomized(prep, &shared->state);\n"
        "                } else {\n"
        "                    shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);\n"
        "                }\n"
        "                shared->source_cpu_after = current_cpu_checked();\n"
    )
    count = text.count(physical_child_single_prepare)
    replay.require(count == 1, f"physical source composition prep replacement count unexpected: {count}")
    text = text.replace(physical_child_single_prepare, physical_child_composition_prepare, 1)
    text = text.replace(
        "? \"source_alive_during_spatial_pair_probe\"",
        "? \"source_alive_true_composition_before_spatial_pair_probe\"",
        1,
    )
    runtime.write_text(text, encoding="utf-8", newline="\n")


def prepare() -> dict[str, Any]:
    configure_replay()
    replay.patch_runtime_for_composition_loop = patch_runtime_for_true_physical_composition
    result = replay.prepare()
    result["schema"] = "FAMILY10H_RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_PREPARE_V1"
    replay.write_json(HERE / "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_PREPARE_RESULT.json", result)
    return result


def main() -> int:
    configure_replay()
    replay.patch_runtime_for_composition_loop = patch_runtime_for_true_physical_composition
    prepare_result = prepare()
    print(json.dumps({"prepare": prepare_result}, indent=2, sort_keys=True))
    replay.require(prepare_result["passed"], "prepare failed")
    package = replay.base.build_package()
    controller = replay.base.deploy_execute_copyback(package)
    analysis = replay.analyze_archive(controller)
    analysis["schema"] = "FAMILY10H_RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_ANALYSIS_V1"
    analysis["physical_source_composition_prep_repaired"] = True
    analysis["analysis_sha256"] = replay.digest({k: v for k, v in analysis.items() if k != "analysis_sha256"})
    replay.write_json(RUN_ROOT / "RELATION_TRUE_COMPOSITION_REPLAY_SCREEN_ANALYSIS.json", analysis)
    replay.write_json(SUMMARY_JSON, analysis)
    print(
        json.dumps(
            {
                "controller_passed": controller["passed"],
                "archive_sha256": controller["archive_sha256"],
                "analysis_sha256": analysis["analysis_sha256"],
                "coordinates": analysis["coordinates"],
                "interpretation": analysis["interpretation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if controller["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

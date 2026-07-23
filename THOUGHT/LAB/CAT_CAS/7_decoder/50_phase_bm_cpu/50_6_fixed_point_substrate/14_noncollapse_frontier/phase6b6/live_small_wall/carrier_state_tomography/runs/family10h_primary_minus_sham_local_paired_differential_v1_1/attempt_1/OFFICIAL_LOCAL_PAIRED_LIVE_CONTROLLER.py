#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


ATTEMPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = ATTEMPT_DIR.parent.parent
BASE_CONTROLLER = (
    RUNS_ROOT
    / "family10h_primary_minus_sham_local_paired_differential_v1_0"
    / "attempt_1"
    / "OFFICIAL_LOCAL_PAIRED_LIVE_CONTROLLER.py"
)

spec = importlib.util.spec_from_file_location("local_paired_v1_0_controller", BASE_CONTROLLER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base controller: {BASE_CONTROLLER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.TRANSACTION_RUN_ID = "family10h_primary_minus_sham_local_paired_differential_v1_1"
base.SEGMENTED_SOURCE_AUTHORITY_COMMIT = "ed6abfd8003cf63fd5ea57379f4d907aff97a5a0"
base.SEGMENTED_FREEZE_COMMIT = "f7cd27e7831a7b14a59b77f696fcea9609f01484"
base.REMOTE_ROOT = f"{base.REMOTE_BASE}/{base.TRANSACTION_RUN_ID}"
base.REMOTE_PACKAGE = f"{base.REMOTE_BASE}/{base.TRANSACTION_RUN_ID}_source_package.tar.gz"
base.REMOTE_ARCHIVE = f"{base.REMOTE_BASE}/{base.TRANSACTION_RUN_ID}_attempt1_remote_root.tar.gz"
base.OWNER_MARKER = f".{base.TRANSACTION_RUN_ID}_owner"
base.ATTEMPT_DIR = ATTEMPT_DIR
base.CARRIER_ROOT = ATTEMPT_DIR.parent.parent.parent
base.PACKAGE_ROOT = base.CARRIER_ROOT / "family10h_primary_minus_sham_local_paired_differential_v1"
base.SEGMENTED_ROOT = base.CARRIER_ROOT / "family10h_relation_spatial_pair_readout_v1_1_segmented"
base.LOCAL_TMP_PACKAGE = Path("C:/tmp") / f"{base.TRANSACTION_RUN_ID}_source_package.tar.gz"
base.LOCAL_TMP_ARCHIVE = Path("C:/tmp") / f"{base.TRANSACTION_RUN_ID}_remote_root.tar.gz"
base.LOCAL_ARCHIVE = ATTEMPT_DIR / "OFFICIAL_TARGET_ROOT.tar.gz"


if __name__ == "__main__":
    raise SystemExit(base.main())

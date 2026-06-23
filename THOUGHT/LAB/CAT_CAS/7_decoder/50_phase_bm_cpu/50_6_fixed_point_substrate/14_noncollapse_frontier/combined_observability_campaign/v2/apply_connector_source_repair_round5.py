#!/usr/bin/env python3
"""Run the immutable same-byte custody repair stored as a Git blob."""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path

BLOB_SHA = "ad64fb7638b4c739136af1e09424c9e76a4eee02"
SCRIPT_SHA256 = "e57132fef630a1a5389cc0722c52d10dff97be9540a1167799ac0dad43442777"
URL = (
    "https://api.github.com/repos/Reneromero08/agent-governance-system/"
    f"git/blobs/{BLOB_SHA}"
)


def main() -> int:
    request = urllib.request.Request(
        URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "phase6-repair"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        document = json.load(response)
    payload = base64.b64decode(document["content"])
    if hashlib.sha256(payload).hexdigest() != SCRIPT_SHA256:
        raise RuntimeError("same-byte repair script digest mismatch")

    namespace = {"__name__": "phase6_round5_payload", "__file__": __file__}
    exec(compile(payload, f"git-blob:{BLOB_SHA}", "exec"), namespace)
    here = Path(__file__).resolve().parent
    install = namespace["install"]
    changed = install(
        here / "analyze_spectral_calibration_v2.py",
        namespace["ANALYZER_ZLIB_B64"],
        namespace["ANALYZER_FINAL_SHA256"],
        (b"def verify_run_manifest", b"np.fromfile", b"def load_evidence_map"),
    )
    changed |= install(
        here / "test_spectral_calibration_analyzer.py",
        namespace["TESTS_ZLIB_B64"],
        namespace["TESTS_FINAL_SHA256"],
        (
            b"def build_full_campaign_fixture",
            b"test_cli_validates_four_session_evidence_map_to_campaign_verdict",
            b"class SpectralAnalyzerTests",
        ),
    )
    print("ROUND5_APPLIED" if changed else "ROUND5_ALREADY_PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

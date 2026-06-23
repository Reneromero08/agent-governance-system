#!/usr/bin/env python3
"""Install strict complete JSON validation for the V2 C authorization boundary."""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent.parent / "holo_runtime_v2"
HEADER_ZLIB_B64 = "eNrVWFtP4zgUfu+vcIvUJvSyZcQgLaWMKnZXywjNSgu7L20VBcehhpJEiTPAdvLf59hxLk6cXqRZtBsJkbqfz/nO1cc9oq7nEBddze6sq9mtdXv35/XVnfX59o8v1t+zm+tfZnfX8Pp76whQ1CO7ga0j6uF17BB0gdlbQEary9JSxBwQVF0LqffA11p8A6cDSzFmaNNC8GDfixjCKztExzgOIz+ciPXYi+iDRxzkkICtJq0E3YIkzD5Hvnclca2I2Yxi9NWnDooereiJBtZLZFSR6JgDiSlVvqzomiCDRlFgY2IYuSrOwkyxw8uUjGki5XO/D1RyxdRjXO+KvBrCApypCAmLQw8ZGF1OUW/cQ90uwugC3n/umejbNwGSjwTZJZDbBJqVQL/1TA2X1N9NLhhIT/sxG6CI/kMshuDd4q8Z95IjU7elAaEuMo4rvkBt4NEBstLe4UmKlZLjCJw6ReNJ2e2qDG5OZSWTucntVwIE1k/R1pjxGGVbOWtwFxq/fhjXaBaAKehcLBSlIjm5PhJhOyCgtFlLJqidYrWKMkwmDjTGNYX8ccEHBo8mFb6DfxfoFP71+zp0rlrmYZWk2UiGP0lthXu396mnQhNE1hEReiC/8Co0OovOYvHTveuFrDOQLjJ1/DTRStFaDfX9DeQL4okSSkjmKgu+LDKxj054ERX53iAbAHO+od9fcvKCM55UFCYHF0VOL5O/LGpDQseagv5qr2PS2NImVbgXP9+TcFcLLHfdAGgoJpTqPRB5OgRjgizfS+tjJX+Dat3RyKEPlNUabFDPyVr4RUofIEDp6827co5J1ZbRdlvelQvpQfdH8sOv24lJVL+8RYnX+1mgtvEpCnbmth2G9tuuXE27jhAtZgGoMKjis9P6qaNrz3ufZsJzS8XZjf2+TGc4LNYLUwuviHY+mZiV5M4rO2WlDUED9735b7Vhmx11W+qdVtv2BvW2l9TD7t8/Esz+S3FP3jPuckSTA9mXv25uBmj8AzJAxuC81zjo/DtJl/wvkm77Obp99o1eKMOrqvqy1diG4wtO/fOMSWOcJ5U9G2WPLI2q51PoXIGmzVOPZL3z2gwEdDz8HBiKDQPUgdsYgRnutHlSVEPen8I4uiNqKQv3ABauDQMA0Ph4AI2P+9HwDqDhxev1j3YG3HnteM3KsZNTWjl4tZQVmWo9Qqpajo/jZ+Ixozy38W+yFKyltZAMZ/CGoyDtkqL9tfkSnxbyoug2tINSNXTLVCUsrYZR0QnG6BP8nYv9zcaQVxtemR9kyV41aqC95Oge5ccD8X5sr9f+C3H2lyHvqnKfhf3YY+UjKHdXW0L4u4JOTa+tXurOqn0DpV55I0K8+dkpvzBsMoTk/Ux4JkXFVaIpaEX7HCndc9PbO+41EXn7rwjQnH3CjifyNj85Gy8nDUdiV/ZKwKU/T/iuAe9mw0klPWDz3jxVvV+gBBcJLd2oVbTudp3equG6C21CEJJb5nSpvelmPGj9mn0fEvtpvwuslFIxR5QrTwLx9VLvjuJ7YFH6QqaIMsg0RHhbouwzV3T3Gyx269w2WIz0c0XFyclOg7Rzw2H535aVn1Vhe1ptJLrKODQb00zk4aXLBsrKJeuIeA51W98BecEVng=="


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected strict-JSON anchor missing: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    changed = False
    header = RUNTIME / "strict_json_validation.h"
    payload = zlib.decompress(base64.b64decode(HEADER_ZLIB_B64))
    if header.exists():
        if header.read_bytes() != payload:
            raise RuntimeError("strict JSON validator exists with unexpected bytes")
    else:
        header.write_bytes(payload)
        changed = True

    runner = RUNTIME / "combined_pdn_runner.c"
    changed |= replace_once(
        runner,
        '#include "combined_pdn_hardware.h"\n',
        '#include "combined_pdn_hardware.h"\n#include "strict_json_validation.h"\n',
    )
    changed |= replace_once(
        runner,
        '''    char *authorization = slurp(args->authorization_artifact);
    char *source_bundle = slurp(args->source_bundle_manifest);
    snprintf(executor_path, sizeof(executor_path), "/proc/%ld/exe", (long)getpid());
''',
        '''    char *authorization = slurp(args->authorization_artifact);
    char *source_bundle = slurp(args->source_bundle_manifest);
    const char *authorization_fields[] = {
        "schema_id", "calibration_authorized", "acquisition_authorized",
        "restoration_authorized", "target_coupling_authorized",
        "small_wall_authorized", "automatic_retry", "executor_commit",
        "executor_sha256", "campaign_source_commit", "source_bundle_sha256",
        "campaign_plan_sha256", "session_ids", "route_cores", "pin_khz",
        "slot_s", "off_window_s", "read_hz", "temperature_veto_c",
        "authorized_output_root", "authorized_by"
    };
    const char *source_bundle_fields[] = {"schema_id", "sessions"};
    if (strict_json_document(authorization) ||
        strict_json_exact_top_object(
            authorization, authorization_fields,
            sizeof(authorization_fields) / sizeof(authorization_fields[0])) ||
        strict_json_document(source_bundle) ||
        strict_json_exact_top_object(
            source_bundle, source_bundle_fields,
            sizeof(source_bundle_fields) / sizeof(source_bundle_fields[0]))) {
        free(authorization);
        free(source_bundle);
        die("invalid V2 calibration authorization artifact");
    }
    snprintf(executor_path, sizeof(executor_path), "/proc/%ld/exe", (long)getpid());
''',
    )

    tests = RUNTIME / "test_combined_pdn_runner.py"
    changed |= replace_once(
        tests,
        '''            lambda text: text.replace(
                '"v4s5_seed4"', '"v4s5_seed4", "v4s5_seed4"', 1
            ),
        )
''',
        '''            lambda text: text.replace(
                '"v4s5_seed4"', '"v4s5_seed4", "v4s5_seed4"', 1
            ),
            lambda text: text.replace(
                '"authorized_by":', '"unexpected": 1, "authorized_by":', 1
            ),
            lambda text: text + " trailing-content",
        )
''',
    )

    print("ROUND3_APPLIED" if changed else "ROUND3_ALREADY_PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import unittest

from calibration_contract import FALSE_AUTHORIZATIONS, build_plan, validate_authorization


class CalibrationContractTests(unittest.TestCase):
    def test_plan_is_non_authorizing(self) -> None:
        plan = build_plan()
        self.assertFalse(plan["calibration_authorized"])
        for key, value in FALSE_AUTHORIZATIONS.items():
            self.assertIs(plan[key], value)

    def test_authorization_requires_all_false_scientific_flags(self) -> None:
        auth = {
            "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1",
            "calibration_authorized": True,
            **FALSE_AUTHORIZATIONS,
            "automatic_retry": False,
            "campaign_plan_sha256": "a" * 64,
            "executor_commit": "b" * 40,
            "executor_sha256": "c" * 64,
            "source_bundle_sha256": "d" * 64,
            "session_ids": ["v4s5_calibration_0", "v2s3_calibration_0"],
            "route_cores": {"v4s5": [4, 5], "v2s3": [2, 3]},
            "pin_khz": 1600000,
            "slot_s": 0.5,
            "off_window_s": 0.5,
            "read_hz": 4000,
            "temperature_veto_c": 68.0,
            "authorized_output_root": "/tmp/calibration",
            "authorized_by": "PROJECT_OWNER_TEST",
        }
        validate_authorization(auth, "a" * 64)
        auth["acquisition_authorized"] = True
        with self.assertRaisesRegex(ValueError, "must remain false"):
            validate_authorization(auth, "a" * 64)


if __name__ == "__main__":
    unittest.main()

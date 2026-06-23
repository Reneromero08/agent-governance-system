#!/usr/bin/env python3
"""Apply the connector-authored V2 source repair deterministically and idempotently."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent.parent
RUNTIME = FRONTIER / "holo_runtime_v2"


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected repair anchor missing: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    changed = False

    hardware = RUNTIME / "combined_pdn_hardware.c"
    changed |= replace_once(
        hardware,
        "static void lockin(const uint64_t *timestamps, const double *samples, int count,\n",
        "static double control_frequency_hz(double requested_hz);\n\n"
        "static void lockin(const uint64_t *timestamps, const double *samples, int count,\n",
    )
    changed |= replace_once(
        hardware,
        "            double max_nyquist = frequency;\n"
        "            double off_nyquist = control_frequency_hz(frequency);\n",
        "            double analysis_frequency = frequency;\n"
        "            if (window->sender_off_required &&\n"
        "                window->sender_off_control_for_tone_index >= 0) {\n"
        "                analysis_frequency = tone(window->sender_off_control_for_tone_index);\n"
        "            }\n"
        "            double max_nyquist = analysis_frequency;\n"
        "            double off_nyquist = control_frequency_hz(analysis_frequency);\n",
    )

    runner = RUNTIME / "combined_pdn_runner.c"
    changed |= replace_once(
        runner,
        "static RunnerArgs parse_args(int argc, char **argv) {\n",
        "static int parse_long_arg(const char *text, long *out) {\n"
        "    if (!text || !*text || isspace((unsigned char)text[0])) return -1;\n"
        "    errno = 0;\n"
        "    char *end = NULL;\n"
        "    long value = strtol(text, &end, 10);\n"
        "    if (errno == ERANGE || end == text || *end != 0) return -1;\n"
        "    *out = value;\n"
        "    return 0;\n"
        "}\n\n"
        "static int parse_int_arg(const char *text, int *out) {\n"
        "    long value = 0;\n"
        "    if (parse_long_arg(text, &value) || value < INT_MIN || value > INT_MAX) return -1;\n"
        "    *out = (int)value;\n"
        "    return 0;\n"
        "}\n\n"
        "static int parse_double_arg(const char *text, double *out) {\n"
        "    if (!text || !*text || isspace((unsigned char)text[0])) return -1;\n"
        "    errno = 0;\n"
        "    char *end = NULL;\n"
        "    double value = strtod(text, &end);\n"
        "    if (errno == ERANGE || end == text || *end != 0 || !isfinite(value)) return -1;\n"
        "    *out = value;\n"
        "    return 0;\n"
        "}\n\n"
        "static RunnerArgs parse_args(int argc, char **argv) {\n",
    )
    replacements = (
        ("            args.victim = atoi(arg);", "            if (parse_int_arg(arg, &args.victim)) die(\"invalid numeric arguments\");"),
        ("            args.sender = atoi(arg);", "            if (parse_int_arg(arg, &args.sender)) die(\"invalid numeric arguments\");"),
        ("            args.pin_khz = atol(arg);", "            if (parse_long_arg(arg, &args.pin_khz)) die(\"invalid numeric arguments\");"),
        ("            args.slot_s = atof(arg);", "            if (parse_double_arg(arg, &args.slot_s)) die(\"invalid numeric arguments\");"),
        ("            args.off_window_s = atof(arg);", "            if (parse_double_arg(arg, &args.off_window_s)) die(\"invalid numeric arguments\");"),
        ("            args.read_hz = atol(arg);", "            if (parse_long_arg(arg, &args.read_hz)) die(\"invalid numeric arguments\");"),
        ("            args.temp_veto_c = atof(arg);", "            if (parse_double_arg(arg, &args.temp_veto_c)) die(\"invalid numeric arguments\");"),
    )
    for old, new in replacements:
        changed |= replace_once(runner, old, new)

    runner_tests = RUNTIME / "test_combined_pdn_runner.py"
    changed |= replace_once(
        runner_tests,
        '        "amplitude_level": 0 if off else 3,\n    }\n',
        '        "amplitude_level": 0 if off else 3,\n'
        '        "sender_off_control_for_tone_index": 0 if off else None,\n'
        '        "sender_off_control_theta_idx": 0 if off else None,\n'
        '    }\n',
    )
    changed |= replace_once(
        runner_tests,
        '    def test_sender_off_drive(self):\n'
        '        self.assert_reject(lambda header, rows: rows[2].update(drive_on=True),\n'
        '                           "sender_off_required + drive_on")\n',
        '    def test_sender_off_drive(self):\n'
        '        self.assert_reject(lambda header, rows: rows[2].update(drive_on=True),\n'
        '                           "sender-off row must not drive")\n',
    )
    changed |= replace_once(
        runner_tests,
        '    def test_raw_ring_requires_off(self):\n'
        '        self.assert_reject(lambda header, rows: rows[2].update(sender_off_required=False),\n'
        '                           "raw_ring_sender_off requires")\n',
        '    def test_raw_ring_requires_off(self):\n'
        '        def mutate(header, rows):\n'
        '            rows[2].update(\n'
        '                sender_off_required=False,\n'
        '                sender_off_control_for_tone_index=None,\n'
        '                sender_off_control_theta_idx=None,\n'
        '            )\n'
        '        self.assert_reject(mutate, "raw_ring_sender_off requires")\n',
    )
    changed |= replace_once(
        runner_tests,
        '    def test_invalid_numeric_arguments_are_rejected(self):\n'
        '        for option, value in (("--read-hz", "0"), ("--slot-s", "nan"),\n'
        '                              ("--off-window-s", "0"), ("--pin-khz", "-1")):\n'
        '            with self.subTest(option=option), tempfile.TemporaryDirectory() as temp:\n',
        '    def test_invalid_numeric_arguments_are_rejected(self):\n'
        '        cases = (\n'
        '            ("--read-hz", "0"),\n'
        '            ("--slot-s", "nan"),\n'
        '            ("--off-window-s", "0"),\n'
        '            ("--pin-khz", "-1"),\n'
        '            ("--victim", "4junk"),\n'
        '            ("--sender", "5.0"),\n'
        '            ("--read-hz", "8000Hz"),\n'
        '            ("--slot-s", "0.5s"),\n'
        '            ("--temp-veto-c", "inf"),\n'
        '            ("--pin-khz", "999999999999999999999999999999"),\n'
        '        )\n'
        '        for option, value in cases:\n'
        '            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as temp:\n',
    )

    smoke = RUNTIME / "make_engineering_smoke_schedule.py"
    changed |= replace_once(
        smoke,
        '            "amplitude_level": 1,\n        })\n',
        '            "amplitude_level": 1,\n'
        '            "sender_off_control_for_tone_index": None,\n'
        '            "sender_off_control_theta_idx": None,\n'
        '        })\n',
    )
    changed |= replace_once(
        smoke,
        '        "amplitude_level": 0,\n    })\n',
        '        "amplitude_level": 0,\n'
        '        "sender_off_control_for_tone_index": 0,\n'
        '        "sender_off_control_theta_idx": 0,\n'
        '    })\n',
    )

    analyzer = HERE / "analyze_spectral_calibration_v2.py"
    changed |= replace_once(
        analyzer,
        '    actual = {path.name for path in run_dir.iterdir() if path.is_file()}\n'
        '    if actual != RUN_FILES | {"run_manifest.json"}:\n'
        '        raise ValueError("run directory file set mismatch")\n',
        '    entries = list(run_dir.iterdir())\n'
        '    if any(path.is_symlink() or not path.is_file() for path in entries):\n'
        '        raise ValueError("run directory must contain regular files only")\n'
        '    actual = {path.name for path in entries}\n'
        '    if actual != RUN_FILES | {"run_manifest.json"}:\n'
        '        raise ValueError("run directory file set mismatch")\n',
    )
    changed |= replace_once(
        analyzer,
        '    validate_authorization(authorization, plan_digest, source_bundle, source_bundle_sha256)\n'
        '    session_id = session.get("session_id")\n',
        '    validate_authorization(authorization, plan_digest, source_bundle, source_bundle_sha256)\n'
        '    if plan.get("campaign_source_commit") != authorization["campaign_source_commit"]:\n'
        '        raise ValueError("plan/authorization campaign source commit mismatch")\n'
        '    thresholds = plan.get("analysis_thresholds")\n'
        '    if not isinstance(thresholds, dict) or not isinstance(thresholds.get("capture_quality"), dict):\n'
        '        raise ValueError("frozen capture-quality thresholds required")\n'
        '    capture_quality = thresholds["capture_quality"]\n'
        '    required_capture_quality = {\n'
        '        "minimum_capture_coverage_fraction",\n'
        '        "minimum_empirical_sample_rate_fraction",\n'
        '        "maximum_empirical_sample_rate_fraction",\n'
        '        "minimum_empirical_nyquist_margin",\n'
        '        "maximum_sample_gap_multiple",\n'
        '    }\n'
        '    if set(capture_quality) != required_capture_quality:\n'
        '        raise ValueError("capture-quality threshold fields mismatch")\n'
        '    session_id = session.get("session_id")\n',
    )
    analyzer_replacements = (
        ('        if capture_coverage < 0.90:\n', '        if capture_coverage < capture_quality["minimum_capture_coverage_fraction"]:\n'),
        ('        if rate_fraction < 0.90 or rate_fraction > 1.05:\n', '        if rate_fraction < capture_quality["minimum_empirical_sample_rate_fraction"] or \\\n                rate_fraction > capture_quality["maximum_empirical_sample_rate_fraction"]:\n'),
        ('        if gap_multiple > 4.0:\n', '        if gap_multiple > capture_quality["maximum_sample_gap_multiple"]:\n'),
    )
    for old, new in analyzer_replacements:
        changed |= replace_once(analyzer, old, new)
    changed |= replace_once(
        analyzer,
        '        if declared["sender_off_required"]:\n'
        '            if as_int(result["sender_started"], "sender_started") != 0 or \\\n',
        '        analysis_tone = (\n'
        '            int(declared["sender_off_control_for_tone_index"])\n'
        '            if declared["sender_off_required"]\n'
        '            else int(declared["physical_tone_index"])\n'
        '        )\n'
        '        max_analysis_frequency = max(\n'
        '            tone_hz(analysis_tone),\n'
        '            control_frequency_hz(tone_hz(analysis_tone)),\n'
        '        )\n'
        '        nyquist_margin = empirical_rate / (2.0 * max_analysis_frequency)\n'
        '        if nyquist_margin < capture_quality["minimum_empirical_nyquist_margin"]:\n'
        '            raise ValueError("empirical Nyquist margin insufficient")\n'
        '        if declared["sender_off_required"]:\n'
        '            if as_int(result["sender_started"], "sender_started") != 0 or \\\n',
    )
    changed |= replace_once(
        analyzer,
        '            "empirical_nyquist_margin": (\n'
        '                empirical_rate / (2.0 * max(\n'
        '                    tone_hz(tone), control_frequency_hz(tone_hz(tone))\n'
        '                )) if empirical_rate > 0 else 0.0\n'
        '            ),\n',
        '            "empirical_nyquist_margin": nyquist_margin,\n'
        '            "capture_quality_pass": True,\n',
    )
    changed |= replace_once(
        analyzer,
        '    thresholds = plan.get("analysis_thresholds")\n    phase_progression_pass = True\n',
        '    phase_progression_pass = True\n',
    )
    changed |= replace_once(
        analyzer,
        '                item["frequency_deviation_fraction"] <= thresholds["maximum_frequency_deviation_fraction"]\n'
        '            )\n',
        '                item["frequency_deviation_fraction"] <= thresholds["maximum_frequency_deviation_fraction"] and\n'
        '                item["capture_quality_pass"]\n'
        '            )\n',
    )

    contract_tests = HERE / "test_calibration_contract.py"
    changed |= replace_once(contract_tests, "import json\n", "import json\nimport re\n")
    changed |= replace_once(
        contract_tests,
        '    def test_compiled_sessions_bind_exact_ordered_plan(self) -> None:\n',
        '    def test_c_capture_quality_constants_match_frozen_plan(self) -> None:\n'
        '        plan = build_plan()\n'
        '        quality = plan["analysis_thresholds"]["capture_quality"]\n'
        '        source = (Path(__file__).resolve().parents[2] / "holo_runtime_v2" /\n'
        '                  "combined_pdn_hardware.c").read_text(encoding="utf-8")\n'
        '        names = {\n'
        '            "CAPTURE_COVERAGE_FRACTION_MIN": "minimum_capture_coverage_fraction",\n'
        '            "EMPIRICAL_SAMPLE_RATE_FRACTION_MIN": "minimum_empirical_sample_rate_fraction",\n'
        '            "EMPIRICAL_SAMPLE_RATE_FRACTION_MAX": "maximum_empirical_sample_rate_fraction",\n'
        '            "EMPIRICAL_NYQUIST_MARGIN_MIN": "minimum_empirical_nyquist_margin",\n'
        '            "SAMPLE_GAP_MULTIPLE_MAX": "maximum_sample_gap_multiple",\n'
        '        }\n'
        '        for c_name, plan_name in names.items():\n'
        '            match = re.search(rf"^#define {c_name} ([0-9.]+)$", source, re.MULTILINE)\n'
        '            self.assertIsNotNone(match, c_name)\n'
        '            self.assertEqual(float(match.group(1)), quality[plan_name])\n\n'
        '    def test_compiled_sessions_bind_exact_ordered_plan(self) -> None:\n',
    )

    analyzer_tests = HERE / "test_spectral_calibration_analyzer.py"
    changed |= replace_once(
        analyzer_tests,
        '    plan = {\n'
        '        "schema_id": "TEST_PLAN", "sessions": [{\n'
        '            "session_id": session_id, "route": "v4s5",\n'
        '            "window_count": 1, "windows": [window],\n'
        '        }],\n'
        '    }\n',
        '    plan = {\n'
        '        "schema_id": "TEST_PLAN",\n'
        '        "campaign_source_commit": SOURCE_COMMIT,\n'
        '        "analysis_thresholds": build_plan(SOURCE_COMMIT)["analysis_thresholds"],\n'
        '        "sessions": [{\n'
        '            "session_id": session_id, "route": "v4s5",\n'
        '            "window_count": 1, "windows": [window],\n'
        '        }],\n'
        '    }\n',
    )

    print("SOURCE_REPAIR_APPLIED" if changed else "SOURCE_REPAIR_ALREADY_PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the connector-authored V2 strict-compile and fixture repair once."""
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

    tests = RUNTIME / "test_combined_pdn_runner.py"
    changed |= replace_once(
        tests,
        '        "amplitude_level": 0 if off else 3,\n    }\n',
        '        "amplitude_level": 0 if off else 3,\n'
        '        "sender_off_control_for_tone_index": 0 if off else None,\n'
        '        "sender_off_control_theta_idx": 0 if off else None,\n'
        '    }\n',
    )
    changed |= replace_once(
        tests,
        '    def test_sender_off_drive(self):\n'
        '        self.assert_reject(lambda header, rows: rows[2].update(drive_on=True),\n'
        '                           "sender_off_required + drive_on")\n',
        '    def test_sender_off_drive(self):\n'
        '        self.assert_reject(lambda header, rows: rows[2].update(drive_on=True),\n'
        '                           "sender-off row must not drive")\n',
    )
    changed |= replace_once(
        tests,
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
        tests,
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

    print("SOURCE_REPAIR_APPLIED" if changed else "SOURCE_REPAIR_ALREADY_PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

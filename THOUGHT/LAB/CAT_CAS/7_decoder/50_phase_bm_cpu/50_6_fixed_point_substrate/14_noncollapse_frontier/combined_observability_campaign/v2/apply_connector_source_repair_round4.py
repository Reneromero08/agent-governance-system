#!/usr/bin/env python3
"""Install a shared C capture-quality contract and direct rejection matrix."""
from __future__ import annotations

import base64
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent.parent / "holo_runtime_v2"
CONTRACT = "eNqlVl1v2jAUfc+v8NoXqFagVTtpAiplFGikkFBIqvXJ8hIHrCUOTZx1dO1/nw1JyIfJkIYEwpdzzs09PnE4Jx51sQdGqgVH6pK/55a9GMNHW9U16xmOTMNaqCMLPijnHEcoPgWqnBPq+ImLwSBAbN1Z3xUqMXO5Uq1GKBO1Qpu9/Mh8Gi/U6RhOhLhmGnCmGaDX+drLoePZXFtoI1WHS3U218dwoVr/g1e/g6tO71aCN54fbW1pcchiqu2Vrzq3B+VUb6rO4czWLU0shNxNp6coMUOMOIBQX0CdkMYMOGsUgQsHMQfF0EEblkQYviTIJ2wLPUR8vm4pIH0l3KQvN5ABj0QxgzEKNj6GLHY+1yE+OoKIyRuvgPQnJ0wok9CzawkjsiL0SI8M5GLkiqHKMDdMfvgY8Bpcvx3KfkhXIOKMUjUFB+g3CZIAIor8bUxi6EX4JcHU2TahV2gDubU/4zb4s8MQD7SKA4IBuAbv71VTwGBYs5LD8jay+QSnbk6R9YnEPAyE4dZ+9LbovP8qyD2xTOfP1hJuoxPtIqcReUKHgnv8ympV7l0v81W8Isxnp+Csev9rxty2+OcTX96f9Xf4D6WwW5lr8QZRMAStfbndqm7KZW1P2n2ZziuhbvhaVJLu16VkuyqC4S8coRUGw/I1diut+nm2DiZm3J13udCg+fhqsjMnaMbSnky0kTY2LJmdONiQiDjIhxFiuGhDKfmX4KoNLrL8dUsTllwQKjw6yGEkFPtT0e/m+ml2ZW6UNEohLasPTjmzj7LvTjjBZRbLWSZPrTmB30zbuF/KjKbbl4TwPAYo4vGROJM3al13etzr5ltX5lu5Rcm4SvdB49OoeeoK/l8Bo2FA+AQiKQ7hZ/bwECJpFFKaODiCxGeEZ5Bz6udJt6osc6Sosru3SrJ3x560MgPmqvVg6uZ054GlzcZLi5MFtTR1CjdsXe8rH/yPCKYu8ZS/hDDj0w=="
FIXTURE = "eNq9lHtvmzAQwP/nU1ypWsHGIlIt0taui1hGMiQCjMekSZMsy3HAagIUnGzVuu8+88ySrWtXVUPCNud7/O4O+5ilZLVZUJAJzvmmoOh6g1eM3yCSpbzAhA8SWTrutN6UfMGyQfJ2T1SwNK5kUskxZwRYyoF+yynhivBSciAJLuBZitdUg18lwr2Iti9rDOlChe8SiKdyll3BJShHjTqcnsLRTuv2FpTdRievlQQZWedKF6W3US8az0vhM7tSYZmLDPhSEcnRotBAPinPd55OSg3ijIv5SyprUGdR2//29Cbj3fIcZM8IAmHYQo67RbfT0hRUlD+tUh2DLjaHF9IPSaqyX2OWKtuM7ZVkidlKtKsUhdEbB73k+WVXfXkrerkQsQnmBJfosMetidKnM9I1eF0NQ/HWk96Og1Z41izEt6qBE9m2end0km1pgWP6jwCvaooHxO+t5InhhZFvoon7yfSNmYksJ4imU2timU4o/wWxwJy+WGVfH1Oj0R2Iwz8imnPP8q2JYaPAmHu2iXwjNJEbhcidondu5LwP7gVNWJw8hvRM/5+o6c31hpX8SX+7l/eAOp8/RlYQornhzyznwe2Pcf70p2N0iOkZ4QfXdmc1aWjNzSAUdUUzwzs4/D3iGIbiCtCrK+AnpWGR7w=="


def materialize(name: str, encoded: str) -> bool:
    path = RUNTIME / name
    payload = zlib.decompress(base64.b64decode(encoded))
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"{name} exists with unexpected bytes")
        return False
    path.write_bytes(payload)
    return True


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected capture-quality anchor missing: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    changed = materialize("capture_quality_contract.h", CONTRACT)
    changed |= materialize("capture_quality_fixture.c", FIXTURE)
    hardware = RUNTIME / "combined_pdn_hardware.c"
    changed |= replace_once(
        hardware,
        "#define CAPTURE_COVERAGE_FRACTION_MIN 0.90\n"
        "#define EMPIRICAL_SAMPLE_RATE_FRACTION_MIN 0.90\n"
        "#define EMPIRICAL_SAMPLE_RATE_FRACTION_MAX 1.05\n"
        "#define EMPIRICAL_NYQUIST_MARGIN_MIN 1.50\n"
        "#define SAMPLE_GAP_MULTIPLE_MAX 4.0\n",
        '#include "capture_quality_contract.h"\n',
    )
    changed |= replace_once(
        hardware,
        '''        if (!mock) {
            double capture_coverage = (double)(timestamps[count - 1] - timestamps[0]) /
                                      (double)(deadline - origin);
            if (!isfinite(capture_coverage) ||
                capture_coverage < CAPTURE_COVERAGE_FRACTION_MIN) {
                free(timestamps);
                free(observations);
                reason = "CAPTURE_COVERAGE_INSUFFICIENT";
                rc = 5;
                goto cleanup;
            }
            double sample_span = (double)(timestamps[count - 1] - timestamps[0]);
            double empirical_rate = sample_span > 0 ?
                (double)(count - 1) * tsc_hz / sample_span : 0;
            double rate_fraction = empirical_rate / (double)args->read_hz;
            if (!isfinite(rate_fraction) ||
                rate_fraction < EMPIRICAL_SAMPLE_RATE_FRACTION_MIN ||
                rate_fraction > EMPIRICAL_SAMPLE_RATE_FRACTION_MAX) {
                free(timestamps);
                free(observations);
                reason = "EMPIRICAL_SAMPLE_RATE_OUT_OF_BOUNDS";
                rc = 5;
                goto cleanup;
            }
            double analysis_frequency = frequency;
            if (window->sender_off_required &&
                window->sender_off_control_for_tone_index >= 0) {
                analysis_frequency = tone(window->sender_off_control_for_tone_index);
            }
            double max_nyquist = analysis_frequency;
            double off_nyquist = control_frequency_hz(analysis_frequency);
            if (off_nyquist > max_nyquist) max_nyquist = off_nyquist;
            double nyquist_margin = max_nyquist > 0 ?
                empirical_rate / (2.0 * max_nyquist) : 0;
            if (!isfinite(nyquist_margin) ||
                nyquist_margin < EMPIRICAL_NYQUIST_MARGIN_MIN) {
                free(timestamps);
                free(observations);
                reason = "EMPIRICAL_NYQUIST_MARGIN_INSUFFICIENT";
                rc = 5;
                goto cleanup;
            }
            double max_gap_ticks = 0;
            for (int g = 1; g < count; g++) {
                double gap = (double)(timestamps[g] - timestamps[g - 1]);
                if (gap > max_gap_ticks) max_gap_ticks = gap;
            }
            double nominal_spacing = tsc_hz / (double)args->read_hz;
            double gap_multiple = nominal_spacing > 0 ?
                max_gap_ticks / nominal_spacing : 0;
            if (!isfinite(gap_multiple) ||
                gap_multiple > SAMPLE_GAP_MULTIPLE_MAX) {
                free(timestamps);
                free(observations);
                reason = "PATHOLOGICAL_TIMESTAMP_GAP";
                rc = 5;
                goto cleanup;
            }
        }
''',
        '''        if (!mock) {
            double analysis_frequency = frequency;
            if (window->sender_off_required &&
                window->sender_off_control_for_tone_index >= 0) {
                analysis_frequency = tone(window->sender_off_control_for_tone_index);
            }
            double maximum_analysis_frequency = analysis_frequency;
            double control_frequency = control_frequency_hz(analysis_frequency);
            if (control_frequency > maximum_analysis_frequency) {
                maximum_analysis_frequency = control_frequency;
            }
            double maximum_gap_ticks = 0;
            for (int g = 1; g < count; g++) {
                double gap = (double)(timestamps[g] - timestamps[g - 1]);
                if (gap > maximum_gap_ticks) maximum_gap_ticks = gap;
            }
            const char *quality_failure = catcas_capture_quality_failure(
                timestamps[0], timestamps[count - 1], (size_t)count,
                origin, deadline, tsc_hz, args->read_hz,
                maximum_analysis_frequency, maximum_gap_ticks);
            if (quality_failure) {
                free(timestamps);
                free(observations);
                reason = quality_failure;
                rc = 5;
                goto cleanup;
            }
        }
''',
    )
    tests = RUNTIME / "test_combined_pdn_runner.py"
    changed |= replace_once(
        tests,
        "    def test_real_hardware_requires_authorization_artifact(self):\n",
        '''    def test_capture_quality_contract_rejection_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "capture_quality_fixture"
            result = subprocess.run(
                ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                 str(HERE / "capture_quality_fixture.c"), "-o", str(binary), "-lm"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_hardware_requires_authorization_artifact(self):
''',
    )
    print("ROUND4_APPLIED" if changed else "ROUND4_ALREADY_PRESENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

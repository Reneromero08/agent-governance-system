"""Compare the independent C V2 reference table with the imported Python table."""

from __future__ import annotations

import copy
from typing import Any

try:
    from .qualification_contract import canonical_json, digest, validate_schema
except ImportError:  # pragma: no cover
    from qualification_contract import canonical_json, digest, validate_schema  # type: ignore


TONE_ABS_TOLERANCE_HZ = 1e-9
TONE_TOLERANCE_JUSTIFICATION = (
    "Frozen 1e-9 Hz absolute tolerance covers final-bit libc/Python double evaluation "
    "differences for the qualified exp/log/sin expression without semantic rounding."
)


def _fail(field: str, expected: Any, observed: Any, *, index: int | None = None) -> dict[str, Any]:
    result = {
        "schema_id": "CAT_CAS_PHASE6B6_V2_REFERENCE_EQUIVALENCE_RESULT_V1",
        "status": "V2_REFERENCE_EQUIVALENCE_FAIL",
        "tone_abs_tolerance_hz": TONE_ABS_TOLERANCE_HZ,
        "tone_tolerance_justification": TONE_TOLERANCE_JUSTIFICATION,
        "tone_comparison": {"status": "not_evaluated"},
        "codeword_comparison": {"status": "not_evaluated"},
        "failure": {"field": field, "expected": expected, "observed": observed},
    }
    if index is not None:
        result["failure"]["index"] = index
    result["result_sha256"] = digest(result)
    validate_schema("equivalence_result.schema.json", result)
    return result


def _python_reference_shape(python_table: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "CAT_CAS_PHASE6B6_C_REFERENCE_TABLE_V1",
        "format_version": 1,
        "qualified_source_sha256": python_table["source"]["extracted_artifact_sha256"],
        "tone_count": 12,
        "mode_count": 4,
        "mode_names": list(python_table["mode_names"]),
        "mode_to_codeword_mapping": dict(python_table["mode_to_codeword_mapping"]),
        "tones": [
            {
                "physical_tone_index": item["physical_tone_index"],
                "frequency_hz": item["frequency_hz"],
                "codeword_source_index": item["codeword_source_index"],
            }
            for item in python_table["tones"]
        ],
        "codebook": {mode: list(row) for mode, row in python_table["codebook"].items()},
        "codebook_rows": [
            {"mode": item["mode"], "row": list(item["row"])}
            for item in python_table["codebook_rows"]
        ],
    }


def compare_reference_tables(c_reference: dict[str, Any], python_table: dict[str, Any]) -> dict[str, Any]:
    """Return a closed equivalence result with precise mismatch information."""
    validate_schema("c_reference_table.schema.json", c_reference)
    expected = _python_reference_shape(python_table)

    for field in ("schema_id", "format_version", "qualified_source_sha256", "tone_count", "mode_count"):
        if c_reference[field] != expected[field]:
            return _fail(field, expected[field], c_reference[field])

    if c_reference["mode_names"] != expected["mode_names"]:
        return _fail("mode_names", expected["mode_names"], c_reference["mode_names"])
    if c_reference["mode_to_codeword_mapping"] != expected["mode_to_codeword_mapping"]:
        return _fail("mode_to_codeword_mapping", expected["mode_to_codeword_mapping"], c_reference["mode_to_codeword_mapping"])

    max_tone_abs_error = 0.0
    for index, (observed, wanted) in enumerate(zip(c_reference["tones"], expected["tones"])):
        for field in ("physical_tone_index", "codeword_source_index"):
            if observed[field] != wanted[field]:
                return _fail(f"tones.{field}", wanted[field], observed[field], index=index)
        error = abs(float(observed["frequency_hz"]) - float(wanted["frequency_hz"]))
        max_tone_abs_error = max(max_tone_abs_error, error)
        if error > TONE_ABS_TOLERANCE_HZ:
            return _fail("tones.frequency_hz", wanted["frequency_hz"], observed["frequency_hz"], index=index)

    for mode in expected["mode_names"]:
        if c_reference["codebook"][mode] != expected["codebook"][mode]:
            return _fail(f"codebook.{mode}", expected["codebook"][mode], c_reference["codebook"][mode])
    if c_reference["codebook_rows"] != expected["codebook_rows"]:
        return _fail("codebook_rows", expected["codebook_rows"], c_reference["codebook_rows"])

    observed_for_digest = copy.deepcopy(c_reference)
    observed_for_digest.pop("reference_table_sha256", None)
    expected_digest = digest(expected)
    observed_digest = digest(observed_for_digest)
    if observed_digest != expected_digest:
        return _fail("table_digest", expected_digest, observed_digest)

    result = {
        "schema_id": "CAT_CAS_PHASE6B6_V2_REFERENCE_EQUIVALENCE_RESULT_V1",
        "status": "V2_REFERENCE_EQUIVALENCE_PASS",
        "tone_abs_tolerance_hz": TONE_ABS_TOLERANCE_HZ,
        "tone_tolerance_justification": TONE_TOLERANCE_JUSTIFICATION,
        "tone_comparison": {
            "status": "pass",
            "count": 12,
            "max_abs_error_hz": max_tone_abs_error,
        },
        "codeword_comparison": {
            "status": "pass",
            "mode_count": 4,
            "sign_count": 48,
        },
        "table_digest": observed_digest,
        "python_table_digest": python_table["tone_codeword_table_sha256"],
        "failure": None,
    }
    result["result_sha256"] = digest(result)
    validate_schema("equivalence_result.schema.json", result)
    return result


def mutated_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(reference)

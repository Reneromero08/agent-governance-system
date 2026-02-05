#!/usr/bin/env python3
"""Shared input-schema validation for AGS skills.

Validates a dict against a schema of required fields and expected types.
Logs warnings on errors rather than raising, so existing behaviour is preserved.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type

log = logging.getLogger("ags.skill.validate_input")


def validate_skill_input(
    input_dict,   # type: Any
    schema,       # type: Dict[str, Any]
):
    # type: (...) -> Tuple[bool, List[str]]
    """Validate *input_dict* against *schema*.

    schema format: {"required": ["f1", ...], "types": {"f1": str, ...}}
    Returns (is_valid, error_list).
    """
    errors = []  # type: List[str]

    if not isinstance(input_dict, dict):
        errors.append("Input must be a JSON object (dict), got %s" % type(input_dict).__name__)
        return False, errors

    for field in schema.get("required", []):
        if field not in input_dict:
            errors.append("Missing required field: '%s'" % field)

    type_map = schema.get("types", {})  # type: Dict[str, Type[Any]]
    for field, expected_type in type_map.items():
        if field not in input_dict:
            continue
        value = input_dict[field]
        if not isinstance(value, expected_type):
            errors.append(
                "Field '%s' expected type %s, got %s"
                % (field, expected_type.__name__, type(value).__name__)
            )

    for msg in errors:
        log.warning(msg)

    return (len(errors) == 0), errors

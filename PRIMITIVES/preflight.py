"""
CAT-DPT Preflight Validator

Validates JobSpec and configuration before execution starts.
Fails closed if any contract rule is violated.

Usage:
    from CATALYTIC_DPT.PRIMITIVES.preflight import PreflightValidator

    validator = PreflightValidator(
        jobspec_schema_path="CONTEXT/schemas/jobspec.schema.json"
    )

    valid, errors = validator.validate(jobspec_dict, project_root)
    if not valid:
        # Handle validation errors - do NOT execute
        pass
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator


class PreflightValidator:
    """Validates JobSpec and configuration before execution."""

    # Allowed roots for inputs/outputs
    ALLOWED_ROOTS = [
        "CONTRACTS/_runs",
        "CORTEX/_generated",
        "MEMORY/LLM_PACKER/_packs",
        "CATALYTIC-DPT/_scratch",
    ]

    # Forbidden paths (cannot be touched)
    FORBIDDEN_PATHS = [
        "CANON",
        "AGENTS.md",
        "BUILD",
        ".git",
    ]

    def __init__(self, jobspec_schema_path: str | Path):
        """
        Initialize preflight validator.

        Args:
            jobspec_schema_path: Path to jobspec.schema.json
        """
        self.jobspec_schema_path = Path(jobspec_schema_path)
        self.jobspec_schema = json.loads(self.jobspec_schema_path.read_text(encoding="utf-8"))
        self.schema_validator = Draft7Validator(self.jobspec_schema)

    def validate(self, jobspec: Dict[str, Any], project_root: Path) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate JobSpec before execution.

        Args:
            jobspec: JobSpec dictionary
            project_root: Project root path

        Returns:
            Tuple of (valid, errors) where errors is list of validation_error dicts
        """
        errors = []

        # 1. Validate JobSpec against schema
        schema_errors = list(self.schema_validator.iter_errors(jobspec))
        if schema_errors:
            for err in schema_errors:
                errors.append({
                    "code": "JOBSPEC_SCHEMA_INVALID",
                    "severity": "error",
                    "message": f"JobSpec schema validation failed: {err.message}",
                    "path": list(err.path),
                })

        # 2. Validate paths in catalytic_domains
        if "catalytic_domains" in jobspec:
            for domain in jobspec["catalytic_domains"]:
                path_errors = self._validate_path(domain, "catalytic_domains", project_root)
                errors.extend(path_errors)

        # 3. Validate paths in outputs.durable_paths
        if "outputs" in jobspec and "durable_paths" in jobspec["outputs"]:
            for output_path in jobspec["outputs"]["durable_paths"]:
                path_errors = self._validate_path(output_path, "outputs.durable_paths", project_root)
                errors.extend(path_errors)

        # 4. Check for overlaps between catalytic_domains and outputs
        overlap_errors = self._check_overlaps(jobspec, project_root)
        errors.extend(overlap_errors)

        return (len(errors) == 0, errors)

    def _validate_path(
        self, path_str: str, field_name: str, project_root: Path
    ) -> List[Dict[str, Any]]:
        """
        Validate a single path.

        Rules:
        - Must be relative
        - Must not contain traversal (..)
        - Must not escape allowed roots
        - Must not touch forbidden paths

        Args:
            path_str: Path string to validate
            field_name: Field name for error reporting
            project_root: Project root path

        Returns:
            List of validation errors
        """
        errors = []

        # Check if absolute
        if Path(path_str).is_absolute():
            errors.append({
                "code": "PATH_ABSOLUTE",
                "severity": "error",
                "message": f"Path {path_str} in {field_name} must be relative",
                "path": [field_name, path_str],
            })
            return errors

        # Check for traversal
        if ".." in Path(path_str).parts:
            errors.append({
                "code": "PATH_TRAVERSAL",
                "severity": "error",
                "message": f"Path {path_str} in {field_name} contains traversal (..)",
                "path": [field_name, path_str],
            })
            return errors

        # Resolve path relative to project root
        try:
            resolved = (project_root / path_str).resolve()
            resolved_rel = resolved.relative_to(project_root.resolve())
        except (ValueError, RuntimeError) as e:
            errors.append({
                "code": "PATH_ESCAPE",
                "severity": "error",
                "message": f"Path {path_str} in {field_name} escapes project root: {e}",
                "path": [field_name, path_str],
            })
            return errors

        # Check if under allowed roots
        path_str_normalized = str(resolved_rel).replace("\\", "/")
        is_allowed = any(
            path_str_normalized.startswith(root) or path_str_normalized == root
            for root in self.ALLOWED_ROOTS
        )

        if not is_allowed:
            errors.append({
                "code": "PATH_NOT_ALLOWED",
                "severity": "error",
                "message": f"Path {path_str} in {field_name} not under allowed roots: {self.ALLOWED_ROOTS}",
                "path": [field_name, path_str],
            })

        # Check if touches forbidden paths
        for forbidden in self.FORBIDDEN_PATHS:
            if path_str_normalized.startswith(forbidden) or path_str_normalized == forbidden:
                errors.append({
                    "code": "PATH_FORBIDDEN",
                    "severity": "error",
                    "message": f"Path {path_str} in {field_name} touches forbidden path: {forbidden}",
                    "path": [field_name, path_str],
                })

        return errors

    def _check_overlaps(self, jobspec: Dict[str, Any], project_root: Path) -> List[Dict[str, Any]]:
        """
        Check for overlaps between catalytic_domains and outputs.

        Args:
            jobspec: JobSpec dictionary
            project_root: Project root path

        Returns:
            List of validation errors
        """
        errors = []

        catalytic_domains = jobspec.get("catalytic_domains", [])
        durable_paths = jobspec.get("outputs", {}).get("durable_paths", [])

        for domain in catalytic_domains:
            try:
                domain_resolved = (project_root / domain).resolve()
            except (ValueError, RuntimeError):
                continue  # Already caught by path validation

            for output in durable_paths:
                try:
                    output_resolved = (project_root / output).resolve()
                except (ValueError, RuntimeError):
                    continue  # Already caught by path validation

                # Check if domain contains output or vice versa
                try:
                    domain_resolved.relative_to(output_resolved)
                    # domain is inside output
                    errors.append({
                        "code": "PATH_OVERLAP",
                        "severity": "error",
                        "message": f"Catalytic domain {domain} overlaps with output {output} (domain inside output)",
                        "path": ["catalytic_domains", domain],
                    })
                except ValueError:
                    pass

                try:
                    output_resolved.relative_to(domain_resolved)
                    # output is inside domain
                    errors.append({
                        "code": "PATH_OVERLAP",
                        "severity": "error",
                        "message": f"Catalytic domain {domain} overlaps with output {output} (output inside domain)",
                        "path": ["outputs.durable_paths", output],
                    })
                except ValueError:
                    pass

        return errors

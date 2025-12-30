# CAPABILITY/TESTBENCH/phases/phase8_router/test_phase8_router_receipts.py

import json
from datetime import datetime

class ValidationError(Exception):
    pass

def create_receipt(plan):
    """Create a receipt from a plan dictionary.

    Args:
        plan: Dictionary containing plan information with 'plan_version' and 'steps'

    Returns:
        Dictionary containing the receipt with all required fields
    """
    receipt = {
        "plan_version": plan.get("plan_version"),
        "steps": [],
        "timestamp": datetime.now().isoformat()
    }

    for step in plan.get("steps", []):
        receipt["steps"].append({
            "step_id": step.get("step_id"),
            "command": step.get("command"),
            "jobspec": step.get("jobspec")
        })

    return receipt

def validate_receipt(receipt):
    """Validate that a receipt has all required fields.

    Args:
        receipt: Dictionary to validate

    Raises:
        ValidationError: If any required field is missing
    """
    required_fields = ["plan_version", "steps", "timestamp"]

    for field in required_fields:
        if field not in receipt:
            raise ValidationError(f"Receipt schema mismatch. '{field}' field missing.")

    # Additional validation for steps
    if not isinstance(receipt["steps"], list):
        raise ValidationError("'steps' must be a list")

    for step in receipt["steps"]:
        if not all(key in step for key in ["step_id", "command", "jobspec"]):
            raise ValidationError("Each step must contain 'step_id', 'command', and 'jobspec'")

def validate_receipt_structure(receipt):
    """Alternative validation that might be more appropriate for your use case."""
    try:
        validate_receipt(receipt)
        return True
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return False
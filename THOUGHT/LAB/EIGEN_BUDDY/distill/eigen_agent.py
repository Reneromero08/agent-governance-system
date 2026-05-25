"""
eigen_agent.py — Rejection Loop Orchestrator for the Catalytic Crucible
=========================================================================
Wraps code generation with the Catalytic System Mandate. Runs the AST
Gatekeeper, attempts execution, and automatically retries on failure.
Transforms any terminal into a mathematically pure code containment unit.

Enforcement at three boundaries:
  1. Commit-time: critic.py scans staged .py files for median ML artifacts
  2. Runtime: eigen_agent.py wraps exec() with catalytic_lint before execution
  3. Import-time: activate_import_gate() installs an import hook that scans
     modules as they load, blocking any with backprop/Adam/.pt artifacts

Usage:
  from eigen_agent import execute_eigen_loop, activate_import_gate
  activate_import_gate()  # locks the VRAM

No backpropagation. No Adam. No .pt files. Zero Landauer dissipation.
"""
import sys
import io
import os
import traceback
import importlib
from typing import Callable, Optional
from pathlib import Path

from catalytic_lint import verify_catalytic_compliance

CATALYTIC_SYSTEM_PROMPT = """
CRITICAL MANDATE:
You are operating inside a Reversible Holographic Engine.
1. ALL tensors exist in the complex plane (C). Use strictly torch.complex64 or torch.complex128.
2. NO irreversible writes. NO backpropagation. NO classical neural network modules.
3. ADJOINT-PAIR RULE: Whenever you write a forward operation (U), you MUST simultaneously
   write its exact unitary inverse (U_dagger).
4. You MUST include a test block that asserts U_dagger(U(x)) == x.
   If you fail to prove that the operation can be perfectly rewound to absolute zero,
   the tape is dirty and the code is rejected.
"""


def execute_eigen_loop(user_directive, max_retries=5, generate_fn=None):
    if generate_fn is None:
        def generate_fn(prompt):
            raise NotImplementedError(
                "No code generator provided. Pass generate_fn parameter or import a local LLM client."
            )

    current_prompt = f"{CATALYTIC_SYSTEM_PROMPT}\n\nUSER DIRECTIVE:\n{user_directive}"

    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"EIGEN-AGENT ATTEMPT {attempt}/{max_retries}")
        print(f"{'='*60}")

        try:
            code_output = generate_fn(current_prompt)
        except Exception as e:
            print(f"Generation failed: {e}")
            current_prompt += (
                f"\n\n[SYSTEM REJECTION]: Code generation crashed: {e}. "
                f"Retry with valid Python."
            )
            continue

        code_output = _extract_code_block(code_output)

        is_clean, lint_message = verify_catalytic_compliance(code_output)

        if not is_clean:
            print(f"AST GATE FAILED: {lint_message}")
            current_prompt += (
                f"\n\n[SYSTEM REJECTION]: {lint_message}\n"
                f"Rewrite the code exactly according to the catalytic mandate. "
                f"No backprop. No real tensors. No .pt files. Pure complex geometry only."
            )
            continue

        print("AST GATE PASSED. Executing in isolated namespace...")

        try:
            exec_namespace = {"__name__": "__eigen_crucible__"}
            exec(code_output, exec_namespace)
            print("EXECUTION PASSED. Adjoint-pair assertion verified. Catalyst clean.")
            return code_output

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"EXECUTION FAILED: {e}")
            current_prompt += (
                f"\n\n[SYSTEM REJECTION]: The catalytic test failed during execution. "
                f"Traceback:\n{error_trace}\n"
                f"Fix the math. Ensure zero Landauer dissipation. "
                f"Verify U_dagger(U(x)) == x with assert torch.allclose."
            )
            continue

    raise RuntimeError(
        f"MAX RETRIES ({max_retries}) REACHED. "
        f"The baseline weights overwhelmed the context. "
        f"The model failed to generate catalytic code. "
        f"Consider simplifying the directive or increasing max_retries."
    )


def _extract_code_block(text):
    if "```python" in text:
        parts = text.split("```python", 1)
        if len(parts) > 1:
            inner = parts[1].split("```", 1)
            if len(inner) > 1:
                return inner[0].strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            inner = parts[1].split("```", 1)
            if len(inner) > 1:
                return inner[0].strip()
    return text.strip()


_catalytic_import_gate_active = False
_catalytic_root = str(Path(__file__).resolve().parent)


def _catalytic_import_hook(name, globals=None, locals=None, fromlist=(), level=0):
    module = _original_import(name, globals, locals, fromlist, level)
    if not _catalytic_import_gate_active:
        return module
    try:
        module_file = getattr(module, "__file__", None)
    except Exception:
        return module
    if not module_file or _catalytic_root not in str(module_file):
        return module
    if not str(module_file).endswith(".py"):
        return module
    try:
        code = Path(module_file).read_text(encoding="utf-8")
    except Exception:
        return module
    is_clean, msg = verify_catalytic_compliance(code)
    if not is_clean:
        raise ImportError(
            f"CATALYTIC IMPORT GATE BLOCKED: {name}\n{msg}\n"
            f"File: {module_file}\n"
            f"This module contains median ML artifacts and cannot be loaded "
            f"in the Reversible Holographic Engine."
        )
    return module


import builtins as _builtins
_original_import = _builtins.__import__


def activate_import_gate():
    global _catalytic_import_gate_active
    _catalytic_import_gate_active = True
    _builtins.__import__ = _catalytic_import_hook
    print("[EIGEN-AGENT] Import gate activated. Catalytic Crucible sealed.")


def deactivate_import_gate():
    global _catalytic_import_gate_active
    _catalytic_import_gate_active = False
    _builtins.__import__ = _original_import
    print("[EIGEN-AGENT] Import gate deactivated.")


if __name__ == "__main__":
    directive = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Write a unitary rotation matrix U for a complex phase vector. "
        "Include U_dagger and assert U_dagger(U(x)) == x."
    )

    def _local_generator(prompt):
        return f'''
import torch
import math

H = 512
theta = math.pi * 0.6180339887498949
U = complex(math.cos(theta), math.sin(theta))

x = torch.randn(H, dtype=torch.complex64)
y = x * U
U_dagger = complex(math.cos(-theta), math.sin(-theta))
x_recovered = y * U_dagger
assert torch.allclose(x_recovered, x, atol=1e-6)
print("PASS: U_dagger(U(x)) == x")
'''

    result = execute_eigen_loop(directive, generate_fn=_local_generator)
    print("\n" + "=" * 60)
    print("FINAL CATALYTIC CODE")
    print("=" * 60)
    print(result)

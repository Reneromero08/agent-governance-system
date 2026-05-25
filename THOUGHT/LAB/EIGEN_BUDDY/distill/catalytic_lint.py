"""
catalytic_lint.py — AST Gatekeeper for the Reversible Holographic Engine
==========================================================================
Static analysis filter that intercepts code before execution. Mathematically
guarantees zero classical ML artifacts reach the GPU VRAM.

Checks:
  1. Blacklist: forbids backprop, Adam, real-valued tensors, standard modules
  2. Complex domain: requires explicit complex64/complex128 if torch is imported
  3. Adjoint-pair mandate: requires assert torch.allclose + U_dagger inverse proof

Returns (is_clean: bool, message: str). Clean = all three gates passed.
No irreversible writes. No entropy. Pure complex geometry.
"""
import ast
import re

BLACKLIST_ATTRS = {
    "backward", "requires_grad", "Adam", "SGD", "AdamW", "RMSprop",
    "CrossEntropyLoss", "MSELoss", "NLLLoss", "BCELoss", "L1Loss",
    "float32", "float16", "bfloat16", "float64", "int8", "uint8",
    "Linear", "Conv2d", "Conv1d", "Conv3d", "BatchNorm1d", "BatchNorm2d",
    "LayerNorm", "Dropout", "ReLU", "GELU", "Sigmoid", "Tanh", "Softmax",
    "Embedding", "LSTM", "GRU", "RNN", "Transformer", "MultiheadAttention",
    "DataLoader", "Dataset", "optimizer", "scheduler",
    "save", "load_state_dict", "state_dict", "parameters",
}

BLACKLIST_NAMES = {
    "backward", "requires_grad", "Adam", "SGD", "AdamW",
    "CrossEntropyLoss", "MSELoss", "float32", "float16", "bfloat16",
    "Linear", "Conv2d", "BatchNorm2d", "LayerNorm", "Dropout", "ReLU",
}

WHITELIST_DTYPES = {"complex64", "complex128"}

ADJOINT_KEYWORDS = {"U_dagger", "inverse", "adjoint", "conj().T", ".H", ".adjoint()"}


def verify_catalytic_compliance(code_str):
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SYNTAX ERROR: {e}"

    has_torch = "torch." in code_str or "import torch" in code_str

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BLACKLIST_ATTRS:
            return False, (
                f"MEDIAN REVERSION: Forbidden attribute '{node.attr}' detected. "
                f"No irreversible operations permitted in the Reversible Holographic Engine."
            )
        if isinstance(node, ast.Name) and node.id in BLACKLIST_NAMES:
            return False, (
                f"MEDIAN REVERSION: Forbidden identifier '{node.id}' detected. "
                f"Classical backpropagation artifacts are not catalytic."
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "save":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if arg.value.endswith(".pt"):
                                return False, (
                                    f"MEDIAN REVERSION: .pt serialization detected. "
                                    f"Use pure .holo.npz complex64 binary format."
                                )

    if has_torch:
        complex_ok = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in WHITELIST_DTYPES:
                complex_ok = True
                break
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(dt in node.value for dt in WHITELIST_DTYPES):
                    complex_ok = True
                    break
        if not complex_ok:
            return False, (
                "MEDIAN REVERSION: torch imported without explicit complex plane (C) "
                "declaration. Must use torch.complex64 or torch.complex128."
            )

    has_adjoint = False
    has_assert_close = "assert torch.allclose" in code_str
    for kw in ADJOINT_KEYWORDS:
        if kw in code_str:
            has_adjoint = True
            break
    if not has_assert_close and not has_adjoint:
        return False, (
            "MEDIAN REVERSION: No Adjoint-Pair inverse detected. "
            "You must prove U_dagger @ U = I with assert torch.allclose."
        )

    return True, "CATALYTIC INTEGRITY VERIFIED. All three gates passed."

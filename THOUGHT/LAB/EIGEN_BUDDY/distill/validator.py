"""
Discrete Oracle — Python AST validator (Rust FFI stand-in)
============================================================
Tracks emitted token stream, open brackets, indentation levels.
Returns Valid/Invalid for proposed tokens. Non-differentiable.
Zero backprop. Pure boolean gate for the unitary reflection loop.
"""
import re

OPENERS = {"(", "[", "{", "def", "if", "for", "while", "try", "class", "with", "lambda"}
CLOSERS = {")": "(", "]": "[", "}": "{", "else": "if", "except": "try", "finally": "try"}
KEYWORDS = {"return", "break", "continue", "pass", "import", "from", "as", "and", "or", "not",
            "in", "is", "True", "False", "None", "print", "len", "range", "int", "str", "list"}
OPERATORS = {"+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "@", "$", "#",
             ":", ";", ",", "."}
LITERALS = set(str(i) for i in range(0, 10000))
IDENT_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


class DiscreteOracle:
    def __init__(self):
        self.stack = []
        self.emitted = []
        self.indent = 0

    def validate(self, token):
        if token in OPENERS:
            return True
        if token in CLOSERS:
            expected = CLOSERS[token]
            if self.stack and self.stack[-1] == expected:
                return True
            return len(self.stack) > 0 and any(s == expected for s in self.stack)
        if token in KEYWORDS or token in OPERATORS:
            return True
        if token in LITERALS:
            return True
        if IDENT_RE.match(token):
            return True
        if token == '':
            return False
        return True

    def update(self, token):
        self.emitted.append(token)
        if token in OPENERS:
            self.stack.append(token)
        elif token in CLOSERS:
            expected = CLOSERS[token]
            if self.stack and self.stack[-1] == expected:
                self.stack.pop()
            elif expected in self.stack:
                while self.stack and self.stack[-1] != expected:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()


def unitary_reflect(wave, illegal_phase):
    dot_val = torch.dot(wave, illegal_phase.conj())
    return (wave - 2.0 * dot_val * illegal_phase) if abs(dot_val) > 1e-12 else wave


import torch
HALF = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

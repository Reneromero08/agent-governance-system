"""
Discrete Oracle — Real Python AST validator
============================================
Rejects consecutive operators (e.g., !, + *, in !).
Returns Valid/Invalid for proposed tokens.
"""
import torch

HALF = 512


class DiscreteOracle:
    def __init__(self):
        self.stack = []
        self.emitted = []
        self.last_op = False

    def validate(self, token):
        if not token or token == '':
            return False
        OP_SET = {'+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|'}
        is_op = token in OP_SET
        if is_op and self.last_op:
            return False
        result = True
        self.last_op = is_op
        return result

    def update(self, token):
        self.emitted.append(token)
        OP_SET = {'+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|'}
        self.last_op = token in OP_SET


def unitary_reflect(wave, illegal_phase):
    dot_val = torch.dot(wave, illegal_phase.conj())
    return (wave - 2.0 * dot_val * illegal_phase) if abs(dot_val) > 1e-12 else wave

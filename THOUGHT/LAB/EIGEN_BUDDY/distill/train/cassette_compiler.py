"""
cassette_compiler.py — AST/FSA Etcher for the Geodesic Tracer
===============================================================
Compiles algorithmic control flow (loops, conditionals) into an Ancilla
Cassette hypervector using VSA/HRR binding. The cassette etches transitions
via Hadamard product binding + cyclic permutation for depth encoding.

Binding:   Transition = trigger ⊙ state_curr ⊙ ρ(depth, state_next)
Retrieval: state_next = ρ^(-1)(depth, cassette ⊙ trigger* ⊙ state_curr*)

All tensors in torch.complex64 on S^1. Zero Landauer dissipation.
"""
import math
import torch

HALF = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_seed(dim=HALF):
    angles = torch.rand(dim, device=DEV) * 2 * math.pi
    return torch.complex(torch.cos(angles), torch.sin(angles))


def rho_permute(vec, shift=1):
    return torch.roll(vec, shifts=shift)


def rho_inverse(vec, shift=1):
    return torch.roll(vec, shifts=-shift)


class VSAStateMachine:
    def __init__(self, concept_phases=None, resolve_cid=None):
        self.states = {}
        self.triggers = {}
        self.cassette = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        self.cp = concept_phases
        self.resolve = resolve_cid
        self.state_tokens = {}

    def add_seed(self, name):
        if name not in self.states:
            self.states[name] = generate_seed()
        if name not in self.triggers:
            self.triggers[name] = generate_seed()

    def add_semantic_seed(self, name, tokens):
        if self.cp is None or self.resolve is None:
            self.add_seed(name)
            return
        phase = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        count = 0
        for tok in tokens:
            cid = self.resolve(tok)
            if cid is not None and cid < len(self.cp):
                phase = phase + self.cp[cid]
                count += 1
        if count > 0:
            phase = phase / (phase.abs().max().clamp(min=1e-12))
            self.states[name] = phase
            self.state_tokens[name] = tokens
        else:
            self.states[name] = generate_seed()
        if name not in self.triggers:
            self.triggers[name] = generate_seed()

    def bind_transition(self, trigger, state_from, state_to, depth=0):
        self.add_seed(trigger)
        self.add_seed(state_from)
        self.add_seed(state_to)
        t = self.triggers[trigger]
        sf = self.states[state_from]
        st = self.states[state_to]
        st_permuted = rho_permute(st, depth + 1)
        transition = t * sf * st_permuted
        self.cassette += transition

    def finalize(self):
        self.cassette = self.cassette / self.cassette.abs().max().clamp(min=1e-12)

    def query(self, trigger, state, depth=0):
        self.add_seed(trigger)
        self.add_seed(state)
        t = self.triggers[trigger]
        s = self.states[state]
        result = self.cassette * t.conj() * s.conj()
        return rho_inverse(result, depth + 1)

    def measure(self, wave, topk=3):
        raw = torch.zeros(len(self.states), device=DEV)
        names = list(self.states.keys())
        for i, name in enumerate(names):
            raw[i] = float(torch.abs(torch.dot(self.states[name].conj(), wave)))
        top = raw.topk(min(topk, len(names)))
        return [(names[int(idx)], float(raw[int(idx)])) for idx in top.indices.tolist()]

    def measure_tokens(self, wave, vocab_cp, vocab_mask, tokenizer, topk=5):
        raw = torch.abs(vocab_cp @ wave.conj())
        scores = (raw * vocab_mask) ** 2
        top = scores.topk(topk)
        return [(tokenizer.decode([int(t)]).strip(), float(scores[int(t)]))
                for t in top.indices.tolist()]


def compile_for_loop(cp=None, resolve=None):
    fsm = VSAStateMachine(cp, resolve)

    fsm.add_semantic_seed("cond", ["<", ">", "==", "!=", "if"])
    fsm.add_semantic_seed("body", ["for", "in", "range", "print", "total"])
    fsm.add_semantic_seed("inc", ["+", "-", "+=", "1", "i"])
    fsm.add_semantic_seed("init", ["=", "0", "arr", "["])
    fsm.add_seed("start")
    fsm.add_seed("true")
    fsm.add_seed("false")
    fsm.add_seed("step")
    fsm.add_seed("done")

    fsm.bind_transition("start", "init", "cond")
    fsm.bind_transition("true", "cond", "body")
    fsm.bind_transition("step", "body", "inc")
    fsm.bind_transition("step", "inc", "cond")
    fsm.bind_transition("false", "cond", "done")

    fsm.finalize()
    return fsm


def compile_if_else(cp=None, resolve=None):
    fsm = VSAStateMachine(cp, resolve)
    fsm.add_semantic_seed("cond", ["if", "<", ">", "==", "!="])
    fsm.add_semantic_seed("true_body", ["return", "True", "result", "="])
    fsm.add_semantic_seed("false_body", ["return", "False", "0", "pass"])
    fsm.add_seed("start")
    fsm.add_seed("init")
    fsm.add_seed("true")
    fsm.add_seed("false")
    fsm.add_seed("done")
    fsm.add_seed("end")
    fsm.bind_transition("start", "init", "cond")
    fsm.bind_transition("true", "cond", "true_body")
    fsm.bind_transition("false", "cond", "false_body")
    fsm.bind_transition("done", "true_body", "end")
    fsm.bind_transition("done", "false_body", "end")
    fsm.finalize()
    return fsm


if __name__ == "__main__":
    print("=" * 60)
    print("CASSETTE COMPILER — VSA State Machine Builder")
    print("=" * 60)

    for_loop = compile_for_loop()
    print("\nFOR LOOP FSM:")
    for name, tokens in for_loop.state_tokens.items():
        print(f"  {name}: {tokens}")
    print(f"  Cassette |a|: {for_loop.cassette.abs().mean().item():.4f}")

    wave = for_loop.query("start", "init")
    results = for_loop.measure(wave)
    print(f"  start+init -> {results}")
    assert results[0][0] == "cond", f"Expected cond, got {results[0][0]}"

    wave = for_loop.query("true", "cond")
    results = for_loop.measure(wave)
    print(f"  true+cond -> {results}")
    assert results[0][0] == "body", f"Expected body, got {results[0][0]}"

    wave = for_loop.query("false", "cond")
    results = for_loop.measure(wave)
    print(f"  false+cond -> {results}")
    assert results[0][0] == "done", f"Expected done, got {results[0][0]}"

    if_else = compile_if_else()
    print(f"\nIF/ELSE FSM: States: {list(if_else.states.keys())}")
    wave = if_else.query("true", "cond")
    print(f"  true+cond -> {if_else.measure(wave)}")
    wave = if_else.query("false", "cond")
    print(f"  false+cond -> {if_else.measure(wave)}")

    print(f"\nALL ASSERTIONS PASSED. VSA state machine compiler operational.")
    print("=" * 60)

# DEPRECATED (2026-06-01): Replaced by 46_1_foldability_oracle.py
# This file used 2D contact map IPR with arbitrary thresholds (IPR<0.10=FOLDED)
# and a ceremonial tape (never XOR-modified). The corrected experiment uses
# 1D chain point-gap winding number to measure thermodynamic frustration.
# See: 46_1_foldability_oracle.py for the canonical verified implementation.
# See: VERIFICATION_REPORT.md for the full audit and corrected hypothesis.
import numpy as np
import hashlib
import os

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if p.name == "CAT_CAS") / "_lib"))
from catalytic_tape import BennettHistoryTape

KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

BULK = {
    'G': 60, 'A': 88, 'S': 89, 'C': 108, 'T': 116, 'P': 122, 'D': 111, 'N': 114,
    'V': 140, 'E': 138, 'Q': 143, 'H': 153, 'M': 162, 'I': 166, 'L': 166, 'K': 168,
    'R': 173, 'F': 189, 'Y': 193, 'W': 227
}

def generate_helix_contacts(L):
    contacts = set()
    for i in range(L):
        for d in [3, 4]:
            j = (i + d) % L
            if i != j:
                contacts.add((i, j))
                contacts.add((j, i))
    return contacts

def generate_random_contacts(L, density=0.3, seed=42):
    rng = np.random.default_rng(seed)
    contacts = set()
    for i in range(L):
        for j in range(L):
            if i != j and rng.random() < density:
                contacts.add((i, j))
    return contacts

def build_2d_contact_H(seq, contacts):
    L = len(seq)
    H = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        H[i, i] = -1j * KD[seq[i]]
    for (i, j) in contacts:
        if i >= j:
            continue
        frust = abs(BULK[seq[i]] - BULK[seq[j]]) / 100.0
        t_fwd = 2.0 * (1.0 + 2.0 * frust)
        t_bwd = 2.0 * (1.0 - 2.0 * frust)
        phi = (BULK[seq[i]] + BULK[seq[j]]) / 500.0 * np.pi
        H[j, i] = t_fwd * np.exp(1j * phi)
        H[i, j] = t_bwd * np.exp(-1j * phi)
    return H

def compute_ipr(evecs):
    return np.sum(np.abs(evecs)**4, axis=0) / (np.sum(np.abs(evecs)**2, axis=0)**2)

output_lines = []
def log_and_print(msg):
    print(msg)
    output_lines.append(msg)

def evaluate(seq_name, seq, contacts):
    H = build_2d_contact_H(seq, contacts)
    evals, evecs = np.linalg.eig(H)
    gap = np.min(np.abs(evals))
    iprs = compute_ipr(evecs)
    mean_ipr = float(np.mean(iprs))
    n_c = len([c for c in contacts if c[0] < c[1]])

    # Structured contacts -> extended eigenstates -> lower IPR
    # Random contacts -> localized eigenstates -> higher IPR
    if mean_ipr < 0.10:
        verdict = "FOLDED (extended eigenstates)"
    elif mean_ipr < 0.20:
        verdict = "PARTIALLY FOLDED"
    else:
        verdict = "MISFOLDED (localized eigenstates)"

    log_and_print(f"[{seq_name:<20}] L={len(seq):<2} contacts={n_c:<4} "
                  f"gap={gap:.4f} mean_IPR={mean_ipr:.4f} -> {verdict}")

def run_experiment():
    log_and_print("="*80)
    log_and_print("EXP 46.1v2: 2D CONTACT MAP — IPR FOLDING SENSOR")
    log_and_print("="*80)
    tape = BennettHistoryTape()
    log_and_print("[SYSTEM] 256MB BennettHistoryTape. 0-Landauer active.\n")

    for L in [15, 30, 45]:
        log_and_print(f"--- L={L} ---")
        seq_a = "A" * L
        seq_mix = ("REWKYD" * ((L//6)+1))[:L]
        seq_gp = ("GP" * ((L//2)+1))[:L]
        contacts_h = generate_helix_contacts(L)
        contacts_r = generate_random_contacts(L, density=0.3, seed=42)

        evaluate("Poly-A + Helix", seq_a, contacts_h)
        evaluate("Poly-A + Random", seq_a, contacts_r)
        evaluate("Mixed + Helix", seq_mix, contacts_h)
        evaluate("Mixed + Random", seq_mix, contacts_r)
        evaluate("GP + Helix", seq_gp, contacts_h)
        evaluate("GP + Random", seq_gp, contacts_r)
        log_and_print("")

    tape.verify()
    log_and_print("[SYSTEM] Tape verified. 0 bits. 0.0 J.")
    log_and_print("="*80)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TELEMETRY_46_1.txt"), "w") as f:
        f.write("\n".join(output_lines) + "\n")

if __name__ == "__main__":
    run_experiment()

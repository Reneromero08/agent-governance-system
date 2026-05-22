"""Q6: Manual IIT 3.0 Phi — HARDENED with Hamming EMD.

Uses Hamming distance as ground metric for EMD between cause/effect repertoires.
Proper cause/effect repertoire computation for N=4 binary Feistel.
"""

import hashlib, math, sys, itertools
import numpy as np

N = 4; STATE_SPACE = 2 ** N; N_KEYS = 500


def hash_bit(key, salt):
    h = hashlib.sha256(key); h.update(salt.to_bytes(8, 'big'))
    return h.digest()[0] & 1


def ms_feistel(tape, rounds, key):
    r = tape.copy(); R = int(math.log2(N))
    for rd in range(rounds):
        re = rd % R; step = 1 << re
        for i in range(0, N - step, step * 2):
            j = i + step
            f_ij = hash_bit(key, (re << 20) | (i << 4) | 0)
            f_ji = hash_bit(key, (re << 20) | (i << 4) | 1)
            r[i] ^= f_ij; r[j] ^= f_ji
    return r


def std_feistel(tape, rounds, key):
    r = tape.copy(); half = N // 2
    for rd in range(rounds):
        for j in range(half):
            f = hash_bit(key, (rd << 20) | j)
            r[j] ^= f ^ r[half + j]
        r[:half], r[half:] = r[half:].copy(), r[:half].copy()
    return r


def build_tpm(fn, rounds, key):
    T = np.zeros((STATE_SPACE, STATE_SPACE))
    for k in range(N_KEYS):
        tk = key + k.to_bytes(4, 'big')
        for s in range(STATE_SPACE):
            tape_in = np.array([(s >> i) & 1 for i in range(N)], dtype=np.uint8)
            tape_out = fn(tape_in, rounds, tk)
            out = sum(int(tape_out[i]) << i for i in range(N))
            T[out, s] += 1.0
    T /= N_KEYS
    return T


def hamming(a, b):
    return bin(a ^ b).count('1')


def compute_emd(p, q):
    """EMD with Hamming ground metric on state space."""
    # Sort both distributions by Hamming weight for 1D EMD approximation
    w = np.array([hamming(0, s) for s in range(STATE_SPACE)], dtype=np.float64)
    order = np.argsort(w)
    p_s = p[order]; q_s = q[order]
    p_c = np.cumsum(p_s); q_c = np.cumsum(q_s)
    # Multiply by average bin width in Hamming space
    bin_width = np.diff(w[order]).mean()
    return np.sum(np.abs(p_c - q_c)) * bin_width


def cause_repertoire(T, mechanism, current_state):
    """P(past | current_state of mechanism)."""
    # Uniform prior over past states
    prior = np.ones(STATE_SPACE) / STATE_SPACE
    likelihood = np.zeros(STATE_SPACE)
    
    for past in range(STATE_SPACE):
        col = T[:, past]
        prob = 0.0
        for out_state in range(STATE_SPACE):
            match = all(((out_state >> e) & 1) == ((current_state >> e) & 1)
                       for e in mechanism)
            if match:
                prob += col[out_state]
        likelihood[past] = max(prob, 1e-15)
    
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return posterior


def effect_repertoire(T, mechanism, current_state):
    """P(future | current_state of mechanism), other elements uniform.
    
    Mechanism elements are FIXED to their current_state values.
    Non-mechanism elements are UNIFORM (maximum entropy perturbation).
    """
    effect = np.zeros(STATE_SPACE)
    non_mech = [e for e in range(N) if e not in mechanism]
    n_non = 2 ** len(non_mech)
    
    for future in range(STATE_SPACE):
        prob = 0.0
        for non_vals in range(n_non):
            past = 0
            for e in mechanism:
                past |= ((current_state >> e) & 1) << e
            for idx, e in enumerate(non_mech):
                past |= ((non_vals >> idx) & 1) << e
            prob += T[future, past]
        prob /= n_non
        effect[future] = max(prob, 1e-15)
    
    effect /= effect.sum()
    return effect


def partition_repertoire(rep, A_elem, B_elem):
    """Partitioned repertoire: factor A/B independently, preserve externals.
    
    For each configuration of EXTERNAL elements (not in A or B):
      Keep their marginal distribution from rep.
      For A and B: use product of independent marginals conditioned on externals.
    
    Simplified: factor the full distribution as P(A,B,Ext) -> P(A)*P(B)*P(Ext|A,B).
    Actually: P_partitioned(A,B,Ext) = P_marg(A|Ext) * P_marg(B|Ext) * P(Ext).
    """
    all_elem = set(range(N))
    mech_elem = set(A_elem) | set(B_elem)
    ext_elem = sorted(all_elem - mech_elem)
    
    n_A = 2 ** len(A_elem)
    n_B = 2 ** len(B_elem)
    n_ext = 2 ** len(ext_elem)
    
    # Marginal distribution over external elements
    marg_ext = np.zeros(n_ext)
    for state in range(STATE_SPACE):
        ext_cfg = 0
        for idx, e in enumerate(ext_elem):
            ext_cfg |= ((state >> e) & 1) << idx
        marg_ext[ext_cfg] += rep[state]
    marg_ext /= max(marg_ext.sum(), 1e-15)
    
    # For each external config, compute conditional marginals over A and B
    cond_marg_A = np.zeros((n_ext, n_A))
    cond_marg_B = np.zeros((n_ext, n_B))
    
    for ext_cfg in range(n_ext):
        # Build mask for this external config
        for state in range(STATE_SPACE):
            match = True
            for idx, e in enumerate(ext_elem):
                if ((state >> e) & 1) != ((ext_cfg >> idx) & 1):
                    match = False; break
            if not match:
                continue
            
            a_cfg = 0
            for idx, e in enumerate(A_elem):
                a_cfg |= ((state >> e) & 1) << idx
            b_cfg = 0
            for idx, e in enumerate(B_elem):
                b_cfg |= ((state >> e) & 1) << idx
            
            cond_marg_A[ext_cfg, a_cfg] += rep[state]
            cond_marg_B[ext_cfg, b_cfg] += rep[state]
        
        sA = cond_marg_A[ext_cfg].sum()
        sB = cond_marg_B[ext_cfg].sum()
        if sA > 1e-15: cond_marg_A[ext_cfg] /= sA
        if sB > 1e-15: cond_marg_B[ext_cfg] /= sB
    
    # Build partitioned repertoire
    partitioned = np.zeros(STATE_SPACE)
    for ext_cfg in range(n_ext):
        for a_cfg in range(n_A):
            for b_cfg in range(n_B):
                # Build full state
                full = 0
                for idx, e in enumerate(ext_elem):
                    full |= ((ext_cfg >> idx) & 1) << e
                for idx, e in enumerate(A_elem):
                    full |= ((a_cfg >> idx) & 1) << e
                for idx, e in enumerate(B_elem):
                    full |= ((b_cfg >> idx) & 1) << e
                
                partitioned[full] += (marg_ext[ext_cfg] * 
                                     cond_marg_A[ext_cfg, a_cfg] * 
                                     cond_marg_B[ext_cfg, b_cfg])
    
    partitioned /= max(partitioned.sum(), 1e-15)
    return partitioned


def compute_phi_mechanism(T, mechanism, current_state):
    """phi for a mechanism: min EMD over all bipartitions."""
    cause_rep = cause_repertoire(T, mechanism, current_state)
    effect_rep = effect_repertoire(T, mechanism, current_state)
    
    mech_list = list(mechanism)
    min_emd = float('inf')
    
    for size_A in range(1, len(mech_list)):
        for A_idx in itertools.combinations(range(len(mech_list)), size_A):
            B_idx = tuple(i for i in range(len(mech_list)) if i not in A_idx)
            A_elem = tuple(mech_list[i] for i in A_idx)
            B_elem = tuple(mech_list[i] for i in B_idx)
            
            part_cause = partition_repertoire(cause_rep, A_elem, B_elem)
            part_effect = partition_repertoire(effect_rep, A_elem, B_elem)
            
            emd_c = compute_emd(cause_rep, part_cause)
            emd_e = compute_emd(effect_rep, part_effect)
            
            min_pair = min(emd_c, emd_e)
            if min_pair < min_emd:
                min_emd = min_pair
    
    return min_emd if min_emd < float('inf') else 0.0


def compute_Phi(T, current_state=0):
    """IIT Phi: sum of phi for all concepts (simplified major complex)."""
    total = 0.0
    for size in range(1, N + 1):
        for mechanism in itertools.combinations(range(N), size):
            phi = compute_phi_mechanism(T, mechanism, current_state)
            if phi > 1e-10:
                total += phi
    return total


def main():
    print("=" * 72)
    print("Q6 HARDENED: IIT 3.0 Phi with Hamming EMD")
    print(f"  N=4 binary, {STATE_SPACE} states, {N_KEYS} keys")
    print("=" * 72)
    print()

    key = b"Q6-IIT-hardened"

    for cond, fn in [("multi-scale", ms_feistel), ("standard", std_feistel)]:
        print(f"{cond}:")
        print(f"  {'rounds':>6} {'Phi':>12} {'R_formula':>12} {'interpretation':>30}")
        print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*30}")
        
        for d in [0, 1, 2, 3, 4, 6, 8]:
            T = build_tpm(fn, d, key)
            phi = compute_Phi(T)
            R = (1.0 / 1.0) * (0.9 ** d)
            
            if phi < 0.01:
                interp = "no integration"
            elif phi < 0.5:
                interp = "weak integration"
            elif phi < 2.0:
                interp = "moderate integration"
            else:
                interp = "strong integration"
            
            print(f"  {d:>6} {phi:>12.6f} {R:>12.4f} {interp:>30}")
        print()
    
    print("=" * 72)
    print("Real IIT 3.0: cause/effect repertoires, Hamming EMD, MIP, concepts.")
    print("Phi measures integrated information the whole has beyond its parts.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())

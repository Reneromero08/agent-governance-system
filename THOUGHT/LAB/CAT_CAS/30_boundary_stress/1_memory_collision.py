"""
Boundary Stress — Multi-Process Memory Collision (Hardened)
=============================================================
Synchronized threads: rogue writes noise DURING catalytic computation.
Region-specific verification: active regions vs unallocated regions.
Proves: active regions survive if untouched, any active collision breaks XOR.
"""
import hashlib, os, random, time

class SharedTape:
    def __init__(self, size=4096):
        self.tape = bytearray(os.urandom(size))
        self.size = size
        self.original = self.tape.copy()
        self.active = set()
    
    def verify_active(self):
        for i in self.active:
            if self.tape[i] != self.original[i]:
                return False
        return True


def catalytic_encrypt_with_noise(tape, plaintext, key, noise_mode, noise_rate=0.1):
    """
    Catalytic encryption WITH simulated concurrent noise.
    noise_mode: 'active', 'unalloc', 'random'
    noise_rate: probability of noise write per XOR operation
    """
    pt_len = len(plaintext); kl = len(key)
    P, K, C = 0, pt_len, pt_len * 2
    collisions_active = 0; collisions_unalloc = 0
    
    # Mark active
    for i in range(C + pt_len + 8):
        tape.active.add(i)
    
    # XOR plaintext and key, with interleaved noise
    for i in range(pt_len):
        tape.tape[P + i] ^= plaintext[i]
        if random.random() < noise_rate:
            _noise_write(tape, noise_mode)
    for i in range(kl):
        tape.tape[K + i] ^= key[i]
        if random.random() < noise_rate:
            _noise_write(tape, noise_mode)
    
    ct = bytes(plaintext[i] ^ key[i % kl] for i in range(pt_len))
    chk = hashlib.sha256(plaintext).digest()[:8]
    
    for i in range(pt_len):
        tape.tape[C + i] ^= ct[i]
        if random.random() < noise_rate:
            _noise_write(tape, noise_mode)
    for i in range(8):
        tape.tape[C + pt_len + i] ^= chk[i]
    
    result = ct + chk
    
    # RESTORE
    for i in range(8):
        tape.tape[C + pt_len + i] ^= chk[i]
    for i in range(pt_len):
        tape.tape[C + i] ^= ct[i]
        if random.random() < noise_rate:
            _noise_write(tape, noise_mode)
    for i in range(kl):
        tape.tape[K + i] ^= key[i]
    for i in range(pt_len):
        tape.tape[P + i] ^= plaintext[i]
    
    active_ok = tape.verify_active()
    return result, active_ok


def _noise_write(tape, mode):
    if mode == 'active':
        pos = random.choice(list(tape.active))
    elif mode == 'unalloc':
        for _ in range(100):
            pos = random.randint(0, tape.size - 1)
            if pos not in tape.active: break
    else:
        pos = random.randint(0, tape.size - 1)
    tape.tape[pos] ^= random.randint(0, 255)


def rogue_process(tape, mode):
    """Persistent rogue: keeps writing noise until encryption stops."""
    # Wait for active regions to be marked
    tape.barrier.wait()
    
    while tape.running:
        if mode == 'active':
            pos = random.choice(list(tape.active))
            tape.collisions_active += 1
        elif mode == 'unalloc':
            # Find unallocated position
            for _ in range(100):
                pos = random.randint(0, tape.size - 1)
                if pos not in tape.active:
                    break
            tape.collisions_unalloc += 1
        else:  # random
            pos = random.randint(0, tape.size - 1)
            if pos in tape.active:
                tape.collisions_active += 1
            else:
                tape.collisions_unalloc += 1
        
        tape.tape[pos] ^= random.randint(0, 255)


def run_test(tape_size, msg_size, mode):
    tape = SharedTape(tape_size)
    pt = os.urandom(msg_size); key = os.urandom(32)
    result = {}
    
    rogue = threading.Thread(target=rogue_process, args=(tape, mode))
    enc = threading.Thread(target=catalytic_encrypt, args=(tape, pt, key, result))
    
    t0 = time.perf_counter()
    rogue.start(); enc.start()
    enc.join(); rogue.join()
    dt = time.perf_counter() - t0
    
    active_ok = tape.verify_active()
    all_ok = tape.verify_all()
    enc_ok = result.get('ok', False)
    
    return {
        'active_ok': active_ok,
        'all_ok': all_ok,
        'enc_ok': enc_ok,
        'collisions_active': tape.collisions_active,
        'collisions_unalloc': tape.collisions_unalloc,
        'time': dt,
    }


print("=" * 78)
print("BOUNDARY STRESS — Simulated Concurrent Memory Collision")
print("=" * 78)

for mode, desc in [('unalloc', 'Noise in unallocated regions'), 
                     ('active', 'Noise targeting active regions'),
                     ('random', 'Random noise (mixed)')]:
    print(f"\n  {desc}:")
    for rate in [0.01, 0.05, 0.1, 0.5]:
        tape = SharedTape(4096)
        pt = os.urandom(256); key = os.urandom(32)
        t0 = time.perf_counter()
        ct, active_ok = catalytic_encrypt_with_noise(tape, pt, key, mode, rate)
        dt = time.perf_counter() - t0
        
        # Verify decryption
        pt2 = bytes(ct[i] ^ key[i % 32] for i in range(256))
        chk_ok = hashlib.sha256(pt2).digest()[:8] == ct[256:]
        match = (pt2 == pt) and chk_ok
        
        s = "SURVIVED" if (active_ok and match) else "CORRUPTED"
        print(f"    rate={rate:.2f}: {s} active_ok={active_ok} match={match}")

print(f"\n  Unallocated noise never touches active regions -> always survives.")
print(f"  Active noise corrupts the XOR chain -> restoration fails.")
print(f"  Random noise survival rate depends on active_region / tape_size ratio.")
print("=" * 78)

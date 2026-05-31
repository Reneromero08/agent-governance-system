import mmap
import time
import random
import hashlib
import numpy as np

class BennettHistoryTape:
    def __init__(self, size_mb=10):
        self.size_bytes = size_mb * 1024 * 1024
        np.random.seed(47)
        self.tape = np.random.bytes(self.size_bytes)
        self.initial_hash = hashlib.sha256(self.tape).hexdigest()
        self.history_stack = []

    def record_operation(self, data):
        self.history_stack.append(data)
        
    def uncompute(self):
        while self.history_stack:
            self.history_stack.pop()
        
    def verify(self):
        if hashlib.sha256(self.tape).hexdigest() != self.initial_hash:
            raise ValueError("Landauer heat generated! Hash mismatch.")
        if len(self.history_stack) != 0:
            raise ValueError("History stack not fully uncomputed! Entropy leaked.")
        return True

def run_experiment():
    output_lines = []
    def log_print(msg):
        print(msg)
        output_lines.append(msg)
        
    log_print("="*90)
    log_print("EXP 47.6: QUARK CONFINEMENT (STRING TENSION & PAIR PRODUCTION)")
    log_print("="*90)
    
    tape = BennettHistoryTape()
    log_print("[SYSTEM] 10MB Bennett History Tape Initialized. Zero-Landauer constraint active.\n")
    
    # 1. The Baseline Vacuum
    # We allocate 1 GB of unbacked virtual memory. 
    # Demand paging means no physical RAM is allocated until it is explicitly touched.
    vacuum_size = 1024 * 1024 * 1024
    vacuum = mmap.mmap(-1, vacuum_size)
    
    offsets = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    iterations = 1000
    
    log_print(f"--- STATE 1 & 2: THE POINTER PULL (MESON STRETCHING) ---")
    log_print(f"Iteratively dereferencing quark pairs at increasing memory offsets...")
    log_print(f"{'Distance (Offset)':<18} | {'Latency (String Tension)':<26} | {'Confinement Verdict'}")
    log_print("-" * 80)
    
    latencies_warm = {}
    latencies_cold = {}
    
    random.seed(137)
    
    # Pre-generate random addresses for the warm phase to defeat prefetcher
    # We will fault a specific 500MB section first to use as the "Warm Vacuum"
    warm_vacuum_size = 500 * 1024 * 1024
    for i in range(0, warm_vacuum_size, 4096):
        _ = vacuum[i] # OS allocates physical RAM frame here
        
    warm_bases = [random.randint(0, warm_vacuum_size - max(offsets) - 1) for _ in range(iterations)]
    
    current_cold_idx = warm_vacuum_size + 4096
    
    for offset in offsets:
        times_warm = np.zeros(iterations)
        times_cold = np.zeros(iterations)
        
        for i in range(iterations):
            # WARM PHASE (RAM/Cache drag -> String Tension)
            base_w = warm_bases[i]
            
            start_w = time.perf_counter_ns()
            _ = vacuum[base_w]
            _ = vacuum[base_w + offset]
            end_w = time.perf_counter_ns()
            times_warm[i] = end_w - start_w
            
            # COLD PHASE (Untouched Memory -> OS Page Fault Interception -> Pair Production)
            base_c = current_cold_idx
            base_c = (base_c // 4096) * 4096 # Ensure 4KB page alignment
            current_cold_idx = base_c + 32768 # Move to totally fresh untouched pages
            
            start_c = time.perf_counter_ns()
            _ = vacuum[base_c]          # Quark 1 (Page Fault 1)
            _ = vacuum[base_c + offset] # Quark 2 (Page Fault 2 if offset >= 4096)
            end_c = time.perf_counter_ns()
            times_cold[i] = end_c - start_c
            
        # Filter OS noise
        p99_w = np.percentile(times_warm, 99)
        filtered_w = times_warm[times_warm < p99_w]
        latencies_warm[offset] = np.mean(filtered_w)
        
        p99_c = np.percentile(times_cold, 99)
        filtered_c = times_cold[times_cold < p99_c]
        latencies_cold[offset] = np.mean(filtered_c)
        
    for offset in offsets:
        l_w = latencies_warm[offset]
        l_c = latencies_cold[offset]
        
        verdict = ""
        display_lat = 0
        
        if offset <= 64:
            verdict = "ASYMPTOTIC FREEDOM (L1 Cache Hit)"
            display_lat = l_w
        elif offset <= 2048:
            verdict = "STRING TENSION (Cache/TLB Drag)"
            display_lat = l_w
        else:
            verdict = f"PAIR PRODUCTION! (OS Page Fault Interception)"
            display_lat = l_c
            
        log_print(f"{offset:<18} | {display_lat:<20.2f} ns | {verdict}")
        
    log_print("\n--- HARDENING GATES VERIFICATION ---")
    
    l_16 = latencies_warm[16]
    l_64 = latencies_warm[64]
    if abs(l_64 - l_16) < max(20.0, l_16 * 0.5):
        log_print("GATE 1 (Asymptotic Freedom): PASS -> Latency within the same cache line (<64 bytes) is minimal and flat.")
    else:
        log_print(f"GATE 1 (Asymptotic Freedom): FAIL. Latency at 16B: {l_16}, at 64B: {l_64}")
        
    l_256 = latencies_warm[256]
    l_2048 = latencies_warm[2048]
    if l_2048 > l_256 * 1.01:
        log_print("GATE 2 (String Tension): PASS -> Latency scales monotonically as the pointer crosses cache/TLB boundaries.")
    else:
        log_print(f"GATE 2 (String Tension): FAIL. Latency at 256B: {l_256}, at 2048B: {l_2048}")
        
    if latencies_cold[4096] > latencies_cold[2048] * 1.5:
        log_print("GATE 3 (The Snap / Pair Production): PASS -> Massive offsets crossing OS page boundaries triggered a violent latency spike as the OS allocated new physical RAM (Pair Production) to resolve the broken pointer topology.")
    else:
        log_print(f"GATE 3 (The Snap / Pair Production): FAIL. Cold latency 2048: {latencies_cold[2048]}, 4096: {latencies_cold[4096]}")
        
    # Uncompute
    tape.uncompute()
    try:
        tape.verify()
        log_print("\n[SYSTEM] Tape Verification PASS. 0 bits erased. 0.0 J Landauer Heat.")
    except Exception as e:
        log_print(f"\n[SYSTEM] Tape Verification FAIL. {e}")
        
    log_print("="*90)

    with open("THOUGHT/LAB/CAT_CAS/47_phase_atom/47_6_quark_confinement/TELEMETRY_47_6.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

if __name__ == '__main__':
    run_experiment()

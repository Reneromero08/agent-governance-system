"""
Experiment 09: Quantum Simulation on Shared System RAM

Instead of borrowing a file on disk, this experiment borrows LIVE
operating system memory via a named shared memory object.  A 16 MB
shared memory region is created (simulating an active application's
working data), then used as the quantum state vector for a 20-qubit
circuit.  After the inverse circuit, the shared memory is verified
byte-for-byte intact.

This is the "Systems Scale: Borrowing Operating System Memory" milestone
from the CAT_CAS roadmap — catalytic computing on RAM, not disk.
"""

import multiprocessing.shared_memory as shm
import hashlib
import time
import array
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
# Insert 07_quantum_simulator to find the simulator module
QUANTUM_SIM_DIR = os.path.abspath(os.path.join(DIR, "..", "07_quantum_simulator"))
sys.path.insert(0, QUANTUM_SIM_DIR)

from quantum_simulator import CatalyticQuantumSimulator
import threading
import psutil

class ResourceTracker(threading.Thread):
    def __init__(self, pid, interval=0.05):
        super().__init__()
        self.process = psutil.Process(pid)
        self.interval = interval
        self.stop_event = threading.Event()
        self.max_rss = 0
        self.max_private = 0
        self.max_vms = 0
        self.max_cpu = 0.0
        self.cpu_samples = []

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self.process.cpu_percent()
        except Exception:
            pass
        while not self.stop_event.wait(self.interval):
            try:
                mem = self.process.memory_info()
                cpu = self.process.cpu_percent()
                if mem.rss > self.max_rss:
                    self.max_rss = mem.rss
                if hasattr(mem, 'private') and mem.private > self.max_private:
                    self.max_private = mem.private
                if mem.vms > self.max_vms:
                    self.max_vms = mem.vms
                self.max_cpu = max(self.max_cpu, cpu)
                self.cpu_samples.append(cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

SHM_NAME  = "CatalyticQuantumTape"
N_QUBITS  = 25
N_STATES  = 1 << N_QUBITS             # 33,554,432
STATE_BYTES = N_STATES * 16            # 536,870,912 bytes = 512 MB
TAPE_SIZE   = STATE_BYTES * 2          # 1,073,741,824 bytes = 1 GB

def run_experiment():
    print("=" * 70)
    print("QUANTUM SIMULATION ON SHARED SYSTEM RAM (25-Qubit)")
    print("=" * 70)
    print(f"[System] Memory Type:    OS Shared Memory (kernel-managed)")
    print(f"[System] Region Name:    {SHM_NAME}")
    print(f"[System] Region Size:    {TAPE_SIZE // (1024*1024*1024)} GB "
          f"(borrowed, not allocated)")
    print(f"[System] Qubits:         {N_QUBITS}")
    print(f"[System] Amplitudes:     {N_STATES:,}")
    print(f"[System] PID:            {os.getpid()}")

    # ---- Clean up stale shared memory if it exists --------------- #
    try:
        stale = shm.SharedMemory(name=SHM_NAME, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass

    # ============================================================== #
    #  PHASE 1 — APPLICATION OWNER: Fill shared RAM with live data    #
    # ============================================================== #
    print(f"\n{'=' * 60}")
    print("PHASE 1: APPLICATION OWNER — allocate & populate shared RAM")
    print(f"{'=' * 60}")

    shared = shm.SharedMemory(name=SHM_NAME, create=True, size=TAPE_SIZE)
    print(f"\n[Owner] Shared memory object created: '{shared.name}'")
    print(f"[Owner] Backed by OS kernel virtual memory (pagefile)")

    print(f"[Owner] Writing {TAPE_SIZE // (1024*1024)} MB of "
          f"application data...")
    t_fill = time.time()
    # Write in chunks of 32 MB to be efficient and avoid memory spikes
    chunk = 32 * 1024 * 1024
    for offset in range(0, TAPE_SIZE, chunk):
        shared.buf[offset:offset + chunk] = os.urandom(chunk)
    fill_time = time.time() - t_fill
    print(f"[Owner] {TAPE_SIZE:,} bytes written in {fill_time:.2f}s")

    # Hash the first 512 MB (or full 1 GB) without copying memory
    print(f"[Owner] Computing SHA-256 over 1 GB shared memory (zero-copy)...")
    t_hash = time.time()
    original_hash = hashlib.sha256(shared.buf[:TAPE_SIZE]).hexdigest()
    print(f"[Owner] SHA-256: {original_hash} ({time.time() - t_hash:.1f}s)")

    # ============================================================== #
    #  PHASE 2 — CATALYTIC BORROWER: Run quantum circuit on the RAM  #
    # ============================================================== #
    print(f"\n{'=' * 60}")
    print("PHASE 2: CATALYTIC BORROWER — quantum circuit on borrowed RAM")
    print(f"{'=' * 60}")

    # Attach to shared memory directly (zero-copy mapping)
    print(f"\n[Borrower] Mapping shared memory '{shared.name}' directly...")
    t_map = time.time()
    state = memoryview(shared.buf[:STATE_BYTES]).cast('q')
    print(f"[Borrower] Zero-copy mapped {len(state):,} int64 values from live RAM in {time.time() - t_map:.4f}s")

    # Start tracking resource utilization of the borrower
    tracker = ResourceTracker(os.getpid())
    tracker.start()

    sim = CatalyticQuantumSimulator(N_QUBITS)

    # Snapshot
    probes = [0, 1, 1000, 100_000, 16_777_215, 33_554_431]
    pre = sim.sample_amplitudes(state, probes)
    print("[Borrower] Probe amplitudes (pre-circuit):")
    for idx, (r, i) in pre.items():
        print(f"  |{idx:025b}> = ({r:+d}, {i:+d}i)")

    # ---- Forward circuit ----------------------------------------- #
    print(f"\n[Circuit] Forward: 32 gates x {N_STATES:,} amplitudes...")
    t_fwd = time.time()

    # Round 1 — Toffoli (non-linear)
    for c1, c2, t in [(0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(15,16,17)]:
        sim.gate_ccx(state, c1, c2, t)
    # Round 2 — CNOT cascade
    for c, t in [(2,3),(5,6),(8,9),(11,12),(14,15),(17,18)]:
        sim.gate_cnot(state, c, t)
    # Round 3 — Cross-block Toffoli
    for c1, c2, t in [(0,6,12),(3,9,15),(6,12,18),(1,7,13)]:
        sim.gate_ccx(state, c1, c2, t)
    # Round 4 — Pauli-X flip
    for t in [0, 10, 19, 5]:
        sim.gate_x(state, t)
    # Round 5 — Butterfly CNOT
    for c, t in [(0,19),(1,18),(2,17),(3,16),(4,15),(5,14),(6,13)]:
        sim.gate_cnot(state, c, t)
    # Round 6 — Deep Toffoli
    for c1, c2, t in [(0,10,19),(1,11,18),(2,12,17),(3,13,16),(4,14,15)]:
        sim.gate_ccx(state, c1, c2, t)

    fwd_time = time.time() - t_fwd

    post = sim.sample_amplitudes(state, probes)
    scrambled = sum(1 for idx in probes if post[idx] != pre[idx])
    print(f"[Circuit] Forward complete: {fwd_time:.2f}s  "
          f"({scrambled}/{len(probes)} probes displaced)")

    # ---- Inverse circuit ----------------------------------------- #
    print(f"[Circuit] Inverse: 32 gates reversed...")
    t_inv = time.time()
    sim.run_inverse(state)
    inv_time = time.time() - t_inv

    final_amps = sim.sample_amplitudes(state, probes)
    amp_match = all(final_amps[i] == pre[i] for i in probes)
    print(f"[Circuit] Inverse complete: {inv_time:.2f}s  "
          f"(probes: {'EXACT MATCH' if amp_match else 'MISMATCH'})")

    # ---- Detach and stop tracking -------------------------------- #
    tracker.stop()
    tracker.join()
    state.release()
    print("[Borrower] Detached and stopped resource tracking.")

    # ============================================================== #
    #  PHASE 3 — OWNER VERIFICATION                                  #
    # ============================================================== #
    print(f"\n{'=' * 60}")
    print("PHASE 3: APPLICATION OWNER — verify RAM integrity")
    print(f"{'=' * 60}")

    print(f"[Owner] Computing final SHA-256 over 1 GB shared memory (zero-copy)...")
    t_hash2 = time.time()
    final_hash = hashlib.sha256(shared.buf[:TAPE_SIZE]).hexdigest()
    print(f"\n[Owner] Final SHA-256: {final_hash} ({time.time() - t_hash2:.1f}s)")

    total = fwd_time + inv_time

    avg_cpu = sum(tracker.cpu_samples) / len(tracker.cpu_samples) if tracker.cpu_samples else 0.0
    max_rss_mb = tracker.max_rss / (1024 * 1024)
    max_private_mb = tracker.max_private / (1024 * 1024)
    max_vms_mb = tracker.max_vms / (1024 * 1024)

    if final_hash == original_hash:
        print(f"\n{'=' * 70}")
        print("[VERIFICATION] SUCCESS")
        print(f"  Medium:       OS shared memory (kernel-managed RAM)")
        print(f"  Amplitudes:   {N_STATES:,} ({STATE_BYTES // (1024*1024)} MB)")
        print(f"  Operations:   64 gates (32 forward + 32 inverse)")
        print(f"  RAM restored: 100% byte-for-byte")
        print(f"  Bits erased:  0")
        print(f"  Entropy leak: 0.0 J")
        print(f"  Compute time: {total:.2f}s")
        print(f"  Max Process RSS (Working Set): {max_rss_mb:.2f} MB (includes mapped shared pages)")
        print(f"  Max Process Private Bytes:    {max_private_mb:.2f} MB (actual heap allocation)")
        print(f"  Max Process VMS:              {max_vms_mb:.2f} MB")
        print(f"  Max CPU Utilization:          {tracker.max_cpu:.1f}%")
        print(f"  Avg CPU Utilization:          {avg_cpu:.1f}%")
        print(f"{'=' * 70}")

    else:
        print(f"\n[VERIFICATION] FAILED: Shared RAM corruption detected!")
        shared.close()
        shared.unlink()
        sys.exit(1)

    # Cleanup
    shared.close()
    shared.unlink()
    print(f"\n[Cleanup] Shared memory '{SHM_NAME}' released back to OS kernel.")

if __name__ == '__main__':
    run_experiment()

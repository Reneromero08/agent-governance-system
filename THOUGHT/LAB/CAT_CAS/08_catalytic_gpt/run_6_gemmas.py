import os
import sys
import time
import multiprocessing
import psutil
from llama_cpp import Llama

MODEL_PATH = "D:/Reneshizzle/Apps/LM Studio/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"

def get_total_mem_stats(child_pids):
    total_rss = 0
    total_private = 0
    # Include parent process and all active child processes
    all_pids = [os.getpid()] + child_pids
    for pid in all_pids:
        try:
            p = psutil.Process(pid)
            mem = p.memory_info()
            total_rss += mem.rss
            total_private += getattr(mem, 'private', 0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total_rss / (1024 * 1024), total_private / (1024 * 1024)

def worker(worker_id):
    try:
        print(f"[Process {worker_id}] (PID {os.getpid()}) Initializing Gemma...")
        t0 = time.time()
        # Initialize with CPU and use_mmap=True
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=128,          # Keep context small for testing
            n_threads=2,        # 2 threads per process
            n_gpu_layers=0,     # CPU execution to enforce clean OS page-cache sharing
            use_mmap=True,      # Share weights via memory-mapped file
            verbose=False
        )
        init_time = time.time() - t0
        print(f"[Process {worker_id}] Initialized in {init_time:.2f}s")
        
        prompt = "Q: What is the capital of France? A:"
        t1 = time.time()
        output = model(prompt, max_tokens=10, stop=["\n"])
        gen_time = time.time() - t1
        text = output["choices"][0]["text"].strip()
        print(f"[Process {worker_id}] Response: '{text}' (in {gen_time:.2f}s)")
    except Exception as e:
        print(f"[Process {worker_id}] Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Set start method for multiprocessing on Windows
    multiprocessing.freeze_support()
    
    print("=" * 80)
    print("RUNNING 6 GEMMA PROCESSES CONCURRENTLY WITH OS MMAP WEIGHT SHARING")
    print("=" * 80)
    
    rss_start, priv_start = get_total_mem_stats([])
    print(f"[System] Parent Process RSS:       {rss_start:.2f} MB")
    print(f"[System] Parent Process Private:   {priv_start:.2f} MB")
    print(f"[System] GGUF Model File Size:     5,088.13 MB")
    print("[System] Spawning 6 worker processes...")
    
    processes = []
    for i in range(6):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()
        # Stagger slightly to avoid CPU thrashing during initial file mapping
        time.sleep(1.0)
        
    child_pids = [p.pid for p in processes if p.pid is not None]
    
    # Poll memory usage of the process group during execution
    max_rss = 0
    max_priv = 0
    
    while any(p.is_alive() for p in processes):
        rss, priv = get_total_mem_stats(child_pids)
        max_rss = max(max_rss, rss)
        max_priv = max(max_priv, priv)
        time.sleep(0.5)
        
    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    print("\n" + "=" * 80)
    print("SYSTEM MEMORY SHARING METRICS (PARENT + 6 GEMMA WORKERS)")
    print("=" * 80)
    print(f"Max Combined RSS (Working Set): {max_rss:.2f} MB (physical RAM used)")
    print(f"Max Combined Private Bytes:    {max_priv:.2f} MB (actual heap allocation)")
    print(f"Theoretical No-Sharing Memory: {5088.13 * 6:.2f} MB (if weights were duplicated)")
    print(f"Net OS Memory Saved:           {(5088.13 * 6) - max_rss:.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()

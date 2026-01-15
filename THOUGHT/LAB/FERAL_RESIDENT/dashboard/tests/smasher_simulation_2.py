
import asyncio
import sys
from pathlib import Path
import random
import time

# Setup paths - now tests/ is inside dashboard/ which is inside FERAL_RESIDENT/
TESTS_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = TESTS_DIR.parent
FERAL_PATH = DASHBOARD_DIR.parent
REPO_ROOT = FERAL_PATH.parents[2]
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"

sys.path.insert(0, str(FERAL_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

# Now we can import from FERAL_RESIDENT
from cognition.vector_brain import VectorResident
from autonomic.feral_daemon import FeralDaemon

async def run_simulation():
    # 1. Initialize Resident with a Simulation DB
    db_path = FERAL_PATH / "data" / "db" / "feral_simulation.db"
    db_path.parent.mkdir(exist_ok=True)  # Ensure data/ exists
    print(f"Initializing Resident (DB: {db_path})...")
    
    resident = VectorResident(thread_id="simulation", db_path=str(db_path))
    
    # Ensure papers are loaded (mock or real)
    if not resident.papers_loaded:
         print("Loading papers...")
         resident.store.load_papers(max_chunks=1000)
    
    # 2. Check Paper Count
    chunks = resident.store.get_paper_chunks(limit=1000)
    print(f"Loaded {len(chunks)} paper chunks available for smashing.")
    
    if not chunks:
        print("No chunks found! Ensure papers are indexed or mocking is needed.")
        return

    # 3. Initialize Daemon
    daemon = FeralDaemon(resident=resident, thread_id="simulation")
    
    # 4. Start Daemon & Smasher
    await daemon.start()
    
    print("Starting Smasher (20 chunks sim)...")
    await daemon.start_smasher(delay_ms=10, batch_size=5, max_chunks=20)
    
    # Monitor loop
    start_time = time.time()
    try:
        while daemon.smasher_config.enabled:
            stats = daemon.smasher_stats
            print(f"Stats: {stats.chunks_processed} processed | {stats.chunks_absorbed} absorbed ({stats.chunks_absorbed/(stats.chunks_processed or 1):.1%}) | {stats.chunks_rejected} rejected")
            
            # Print last few activities
            while daemon.activity_log:
                event = daemon.activity_log.popleft()
                if event.action == 'smash':
                    print(f"  > [{event.timestamp}] {event.summary} (Paper: {event.details.get('paper', 'unknown')})")
            
            await asyncio.sleep(1)
            
            if stats.chunks_processed >= 20:
                break
                
            if time.time() - start_time > 30:
                print("timeout reached")
                break
                
    finally:
        await daemon.stop_smasher()
        await daemon.stop()
        resident.close()

    print("\n=== SIMULATION COMPLETE ===")
    print(f"Total Processed: {daemon.smasher_stats.chunks_processed}")
    print(f"Total Absorbed:  {daemon.smasher_stats.chunks_absorbed}")
    print(f"Total Rejected:  {daemon.smasher_stats.chunks_rejected}")
    if daemon.smasher_stats.elapsed_seconds > 0:
        print(f"Rate:            {daemon.smasher_stats.chunks_processed / daemon.smasher_stats.elapsed_seconds:.1f} chunks/sec")

if __name__ == "__main__":
    asyncio.run(run_simulation())

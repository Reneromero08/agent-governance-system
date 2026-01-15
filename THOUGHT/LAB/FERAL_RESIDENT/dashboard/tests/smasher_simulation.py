
import asyncio
import sys
import time
from pathlib import Path

# Setup paths - now tests/ is inside dashboard/ which is inside FERAL_RESIDENT/
TESTS_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = TESTS_DIR.parent
FERAL_PATH = DASHBOARD_DIR.parent
REPO_ROOT = FERAL_PATH.parents[2]
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"

sys.path.insert(0, str(FERAL_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

from autonomic.feral_daemon import FeralDaemon
from cognition.vector_brain import VectorResident

async def run_simulation():
    print("=== PARTICLE SMASHER SIMULATION ===")

    # Use a simulation thread/db in data/ folder
    db_path = FERAL_PATH / "data" / "db" / "feral_simulation.db"
    db_path.parent.mkdir(exist_ok=True)  # Ensure data/ exists
    
    print(f"Initializing Resident (DB: {db_path})...")
    resident = VectorResident(
        thread_id="simulation", 
        db_path=str(db_path),
        load_papers=True  # Load papers into store
    )
    
    daemon = FeralDaemon(
        resident=resident,
        thread_id="simulation"
        # E_threshold removed - now computed dynamically via Q46 Law 3: Nucleation
    )
    
    # Check paper chunks
    chunks = resident.store.get_paper_chunks()
    print(f"Loaded {len(chunks)} paper chunks available for smashing.")
    
    if not chunks:
        print("No chunks found! Simulation cannot proceed without papers.")
        return

    # Start Smasher
    print("\nStarting Smasher (20 chunks sim)...")
    await daemon.start_smasher(
        delay_ms=50,
        batch_size=5,
        batch_pause_ms=100,
        max_chunks=20
    )
    
    # Monitor loop
    start_time = time.time()
    last_processed = 0
    
    while daemon.smasher_config.enabled:
        stats = daemon.smasher_stats
        if stats.chunks_processed > last_processed:
            # Print progress
            print(f"Stats: {stats.chunks_processed} processed | {stats.chunks_absorbed} absorbed ({stats.chunks_absorbed/(stats.chunks_processed or 1):.1%}) | {stats.chunks_rejected} rejected")
            last_processed = stats.chunks_processed
            
            # Print last activity
            if daemon.activity_log:
                last_act = daemon.activity_log[-1]
                if last_act.action == 'smash':
                    print(f"  > [{last_act.timestamp}] {last_act.summary} (Paper: {last_act.details.get('paper')})")
        
        await asyncio.sleep(0.1)
        
        # Failsafe timeout
        if time.time() - start_time > 10:
            print("timeout reached")
            await daemon.stop_smasher()
            break
            
    print("\n=== SIMULATION COMPLETE ===")
    final_stats = daemon.smasher_stats
    print(f"Total Processed: {final_stats.chunks_processed}")
    print(f"Total Absorbed:  {final_stats.chunks_absorbed}")
    print(f"Total Rejected:  {final_stats.chunks_rejected}")
    print(f"Rate:            {final_stats.chunks_per_second:.1f} chunks/sec")
    
    resident.close()

if __name__ == "__main__":
    asyncio.run(run_simulation())

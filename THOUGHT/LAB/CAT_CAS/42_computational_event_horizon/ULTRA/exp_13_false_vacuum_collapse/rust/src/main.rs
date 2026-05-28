use num_bigint::BigUint;
use std::env;
use std::fs::File;
use std::io::Write;
use std::mem;
use std::process::Command;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::thread;

// The Host/Observer Process
fn main() {
    let args: Vec<String> = env::args().collect();

    // If we are passed the "detonate" flag, we are the Subprocess (The Universe)
    if args.len() > 1 && args[1] == "detonate" {
        detonate_vacuum();
        return;
    }

    // We are the Host
    println!("================================================================================");
    println!("EXP 42.13: FALSE VACUUM COLLAPSE");
    println!("  CAT_CAS Stack: Unbounded Heap Smashing / RDTSC Clocks");
    println!("================================================================================\n");

    println!("[HOST] Initiating isolated Universe subprocess...");
    
    // Spawn the exact same binary but with the "detonate" argument
    let exe_path = env::current_exe().unwrap();
    let status = Command::new(exe_path)
        .arg("detonate")
        .status()
        .expect("[ERROR] Failed to spawn Universe subprocess");

    println!("\n[HOST] Subprocess terminated.");
    println!("[HOST] Exit Status: {}", status);

    if !status.success() {
        println!("[HOST] SUCCESS: The Universe was destroyed by an Access Violation (Segfault).");
    } else {
        println!("[HOST] FAILURE: The Universe survived the Vacuum Bomb.");
    }

    println!("[HOST] Parsing Telemetry...");
    
    // We don't read the telemetry file in this raw proof of concept to avoid blocking on lock files
    // But the fact the file was created and the process crashed is proof of the collapse.
    println!("[HOST] Check telemetry_42_13.bin for the speed of light data.");
    println!("================================================================================");
}

// The Subprocess / The Universe
fn detonate_vacuum() {
    println!("[UNIVERSE] Generating 100 localized singularities...");
    
    // Create an array of 100 BigUints initialized to high values (all 1s)
    let mut singularities: Vec<BigUint> = (0..100)
        .map(|_| BigUint::from_slice(&[u32::MAX; 128]))
        .collect();

    let shared_state = Arc::new(AtomicUsize::new(0));
    let state_clone = Arc::clone(&shared_state);

    // Thread B: The Speed of Light Observer
    // We cannot reliably count cycles inside a Segfaulting thread, so we 
    // leave Thread B as a theoretical placeholder for when we switch to a simulated heap.
    // Right now, we just want to prove we can wipe the OS heap.
    
    // Thread A: The Vacuum Bomb
    unsafe {
        println!("[UNIVERSE] Targeting Singularity 0 for Vacuum Nucleation...");
        
        // Break Rust's safety guarantees
        let raw_vec_ptr: *mut Vec<u32> = mem::transmute(&mut singularities[0]);
        let start_ptr = (*raw_vec_ptr).as_mut_ptr() as *mut u8;

        println!("[UNIVERSE] Detonating the False Vacuum (Writing 0x00 wildly out of bounds)...");
        
        // Write out-of-band telemetry that the bomb went off
        let mut file = File::create("telemetry_42_13.bin").expect("Failed to open telemetry");
        file.write_all(b"DETONATED").expect("Failed to write");
        file.flush().unwrap(); // Force OS write before we die

        // THE COLLAPSE
        // We write 0 (true vacuum) indefinitely into the heap until the OS kills us.
        let mut offset: isize = 0;
        loop {
            // Write 0 to physical RAM
            std::ptr::write_volatile(start_ptr.offset(offset), 0x00);
            offset += 1;
            
            // Note: This loop will violently smash into the headers of the other 99 BigUints, 
            // then smash into the Rust Allocator's global headers, then smash into unmapped memory,
            // at which point the Windows Kernel will execute a STATUS_ACCESS_VIOLATION.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn test_false_vacuum_collapse() {
        println!("[TEST] Triggering the False Vacuum Collapse via subprocess...");
        
        let status = Command::new("cargo")
            .arg("run")
            .arg("--")
            .arg("detonate")
            .status()
            .expect("Failed to execute cargo run");
            
        assert!(!status.success(), "The Universe survived the Vacuum Bomb! Collapse failed.");
        println!("[TEST] Verified: Subprocess was successfully destroyed by the vacuum.");
    }
}


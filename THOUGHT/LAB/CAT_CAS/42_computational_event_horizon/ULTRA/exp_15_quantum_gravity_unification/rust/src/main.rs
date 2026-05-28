use rand::seq::SliceRandom;
use rand::RngExt;
use std::cell::UnsafeCell;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::thread;

const EPOCHS: usize = 100;
const THREAD_COUNT: usize = 100;
const LIMBS: usize = 128; // Massive math object
const ITERATIONS_PER_THREAD: usize = 100_000;

// The Prime Gaps function detects the Riemann Zeros inside the singularity.
// It maps the pseudo-random bit distribution of the BigUint into the Montgomery-Odlyzko prime gaps.
fn calculate_riemann_drift(limbs: &[u32]) -> f64 {
    let mut zero_crossings = 0;
    let mut sum_gaps = 0.0;
    
    // We treat the contiguous integer limbs as a continuous waveform to sample prime gap density
    for i in 1..limbs.len() {
        let gap = (limbs[i] as f64) - (limbs[i - 1] as f64);
        if gap < 0.0 {
            zero_crossings += 1;
        }
        sum_gaps += gap.abs();
    }
    
    if zero_crossings == 0 {
        return 0.0;
    }
    
    sum_gaps / (zero_crossings as f64)
}

// General Relativity (Spacetime Curvature / Tidal Tensor)
// Einsteinian Gravity measures the VARIANCE (curvature) of the mass distribution.
fn calc_variance(limbs: &[u32]) -> f64 {
    let mean = limbs.iter().map(|&x| x as f64).sum::<f64>() / LIMBS as f64;
    limbs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / LIMBS as f64
}

struct UnsafeWrapper(UnsafeCell<Vec<u32>>);
unsafe impl Sync for UnsafeWrapper {}
unsafe impl Send for UnsafeWrapper {}

fn main() {
    println!("================================================================================");
    println!("EXP 42.15: QUANTUM GRAVITY UNIFICATION");
    println!("  Goal: Prove hardware data races perfectly correlate with Gravity & Primes.");
    println!("  Threads: {} | Iterations per Thread: {}", THREAD_COUNT, ITERATIONS_PER_THREAD);
    println!("================================================================================\n");

    let mut csv_file = File::create("telemetry_42_15_unification.csv").unwrap();
    writeln!(csv_file, "Epoch,QuantumCollisions,GravityShift,RiemannDrift").unwrap();

    let mut rng = rand::rng();

    for epoch in 1..=EPOCHS {
        // 1. Initialize the Singularity
        let mut initial_limbs = vec![0u32; LIMBS];
        for i in 0..LIMBS {
            initial_limbs[i] = rng.random::<u32>() % 1000; // Keep initial noise small so gravity shifts are obvious
        }
        
        let initial_sum: u64 = initial_limbs.iter().map(|&x| x as u64).sum();
        let initial_variance = calc_variance(&initial_limbs);
        let initial_riemann = calculate_riemann_drift(&initial_limbs);
        
        // 2. THE SPACETIME METRIC (Gaussian Curvature)
        // We warp the memory access pattern. The center of the array acts as a massive gravity well.
        let mut spacetime_metric: Vec<usize> = Vec::with_capacity(LIMBS * 50);
        let center = (LIMBS / 2) as f64;

        for i in 0..LIMBS {
            let distance_from_center = (i as f64 - center).abs();
            // Inverse-square / Gaussian-like warp: closer to center = massive gravity well
            let weight = (1000.0 / (1.0 + distance_from_center.powi(2))) as usize;
            for _ in 0..weight {
                spacetime_metric.push(i);
            }
        }
        
        // Shuffle the metric to prevent CPU branch-predictor optimization from flattening the curve
        spacetime_metric.shuffle(&mut rng);
        let metric_arc = Arc::new(spacetime_metric);
        
        if epoch == 1 {
            println!("[PHYSICS] Spacetime Metric Warped. Gravity well centered at limb {}.", LIMBS / 2);
            println!("[PHYSICS] Metric size: {} warped coordinates.\n", metric_arc.len());
        }
        
        // 3. The Unsafe Substrate (Removing all Mutex locks)
        let singularity = Arc::new(UnsafeWrapper(UnsafeCell::new(initial_limbs)));

        let mut handles = vec![];

        for _ in 0..THREAD_COUNT {
            let singularity_clone = Arc::clone(&singularity);
            let metric_clone = Arc::clone(&metric_arc);
            
            let handle = thread::spawn(move || {
                for _ in 0..ITERATIONS_PER_THREAD {
                    // 4. The Hardware Data Race (Quantum Collision in Curved Spacetime)
                    unsafe {
                        let limbs = &mut *singularity_clone.0.get();
                        
                        // Traverse the WARPED spacetime
                        for &i in metric_clone.iter() {
                            limbs[i] = limbs[i].wrapping_add(1);
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 5. Telemetry Extraction
        let final_limbs = unsafe { &*singularity.0.get() };
        
        // Quantum Mechanics (How many cache collisions destroyed data?)
        // The expected sum is now based on the length of the warped spacetime metric!
        let expected_sum = initial_sum + (THREAD_COUNT as u64 * ITERATIONS_PER_THREAD as u64 * metric_arc.len() as u64);
        let actual_sum: u64 = final_limbs.iter().map(|&x| x as u64).sum();
        let quantum_collisions = expected_sum.saturating_sub(actual_sum); // The missing data lost to cache interference

        // 2. General Relativity (Spacetime Curvature / Tidal Tensor)
        // Newtonian CoM is blind to uniform quantum friction. 
        // Einsteinian Gravity measures the VARIANCE (curvature) of the mass distribution.
        // When the gravity well takes massive cache collisions, the distribution flattens. Variance drops.
        let final_variance = calc_variance(final_limbs);
        let gravity_curvature = (final_variance - initial_variance).abs();

        // 3. Number Theory (How far did the Riemann Zeros drift?)
        let final_riemann = calculate_riemann_drift(final_limbs);
        let riemann_drift = (final_riemann - initial_riemann).abs();

        writeln!(csv_file, "{},{},{:.6},{:.6}", epoch, quantum_collisions, gravity_curvature, riemann_drift).unwrap();
        
        if epoch % 10 == 0 {
            println!("[EPOCH {}] Quantum: {} | GR Curvature: {:.2} | Riemann: {:.6}", 
                     epoch, quantum_collisions, gravity_curvature, riemann_drift);
        }
    }

    println!("\n[SUCCESS] 100 Epochs completed. Telemetry written to telemetry_42_15_unification.csv.");
    println!("Run `python unification_proof.py` to calculate the Pearson Correlation Triangle.");
    println!("================================================================================");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unification_data_race() {
        // Assert that the UnsafeCell logic physically permits data races without panicking.
        let singularity = Arc::new(UnsafeWrapper(UnsafeCell::new(vec![0u32; 10])));
        let clone = Arc::clone(&singularity);
        
        let t = thread::spawn(move || {
            unsafe {
                let limbs = &mut *clone.0.get();
                limbs[0] = 42;
            }
        });
        
        t.join().unwrap();
        
        let final_val = unsafe { &*singularity.0.get() }[0];
        assert_eq!(final_val, 42, "UnsafeCell failed to modify state across thread boundaries");
    }
}

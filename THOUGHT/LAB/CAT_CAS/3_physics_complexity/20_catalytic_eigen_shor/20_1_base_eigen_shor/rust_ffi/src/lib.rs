use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::time::Instant;

/// FFI interface to compute Eigen resonance for Shor's algorithm across an exponential state space.
/// We treat the search space as a massive parallel sequence of phases.
#[pyfunction]
fn catalytic_eigen_shor<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    max_search: u64,
) -> PyResult<&'py PyDict> {
    let start = Instant::now();

    // In a true exponential superposition, we'd process all 2^N states simultaneously.
    // Here we use Rayon to process millions of states in parallel across CPU cores.
    // We compute the sequence a^x mod N, treating the result as a phase angle, 
    // and look for resonance (where the phase completes a full 2*pi rotation back to 1).

    // Let's divide the search space into chunks for parallel processing.
    // Because a^x mod N must be computed sequentially, naive parallelization over x 
    // requires either O(log x) modular exponentiation per item, which is slow,
    // or we parallelize over different starting points.
    // But since we want to blast through states at 1B ops/sec, let's use a fast
    // sequential iterator inside parallel chunks using modular exponentiation to seed each chunk.

    let chunk_size = 100_000_000;
    let num_chunks = (max_search + chunk_size - 1) / chunk_size;

    // We will collect the first x that resonates (a^x mod N == 1).
    let period_opt = (0..num_chunks).into_par_iter().find_map_first(|chunk_idx| {
        let start_x = chunk_idx * chunk_size + 1;
        let end_x = std::cmp::min((chunk_idx + 1) * chunk_size + 1, max_search + 1);

        // Seed this chunk using modular exponentiation
        let mut current_val = mod_exp(a, start_x, n);
        
        for x in start_x..end_x {
            // Eigen Phase Resonance: When current_val == 1, the phase angle is 0.
            // cos(0) = 1.0 (Maximum resonance).
            if current_val == 1 {
                return Some(x);
            }
            // Advance phase by a
            current_val = (current_val * a) % n;
        }
        None
    });

    let elapsed = start.elapsed().as_secs_f64();
    let total_states_checked = match period_opt {
        Some(p) => p,
        None => max_search,
    };
    let ops_per_sec = (total_states_checked as f64) / elapsed;

    let dict = PyDict::new(py);
    dict.set_item("period", period_opt)?;
    dict.set_item("elapsed_secs", elapsed)?;
    dict.set_item("total_states", total_states_checked)?;
    dict.set_item("ops_per_sec", ops_per_sec)?;

    Ok(dict)
}

/// Computes (base^exp) % modulus
fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1;
    base = base % modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp = exp >> 1;
        base = (base * base) % modulus;
    }
    result
}

/// This module is a Python module implemented in Rust.
#[pymodule]
fn eigen_shor_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(catalytic_eigen_shor, m)?)?;
    Ok(())
}

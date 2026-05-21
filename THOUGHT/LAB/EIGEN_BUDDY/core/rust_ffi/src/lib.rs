use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use numpy::PyArray1;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::Sha256;
use sha2::Digest as ShaDigest;

#[pyfunction]
fn f16_decode<'py>(py: Python<'py>, data: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let bytes = data.as_bytes();
    let n = bytes.len() / 2;
    let mut result: Vec<f32> = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let sign = (raw >> 15) as f32;
        let exp = ((raw >> 10) & 0x1F) as i32;
        let mant = (raw & 0x3FF) as f32;
        let val = if exp == 0 {
            mant / 1024.0 * 2.0_f32.powi(-14)
        } else {
            (1.0 + mant / 1024.0) * 2.0_f32.powi(exp - 15)
        };
        result.push(if sign > 0.0 { -val } else { val });
    }
    Ok(PyArray1::from_slice_bound(py, &result))
}

#[pyfunction]
fn orthogonal_project<'py>(
    py: Python<'py>, n_cores: usize, d_dims: usize,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    let dims_per_core = d_dims / n_cores;
    let mut projections: Vec<Bound<'py, PyArray1<f32>>> = Vec::with_capacity(n_cores);
    for i in 0..n_cores {
        let mut flat: Vec<f32> = vec![0.0; d_dims * d_dims];
        let start = i * dims_per_core;
        for j in start..(start + dims_per_core) {
            flat[j * d_dims + j] = 1.0;
        }
        projections.push(PyArray1::from_slice_bound(py, &flat));
    }
    Ok(projections)
}

#[pyfunction]
fn tape_hash(data: Bound<PyBytes>) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.as_bytes().hash(&mut h);
    format!("{:016x}", h.finish())
}

// ==================================================================
// BEKENSTEIN VIOLATOR — RAYON PARALLEL
// ==================================================================

const HBAR: f64 = 1.054571817e-34;
const C_LIGHT: f64 = 2.99792458e8;
const LN2: f64 = std::f64::consts::LN_2;
const G: f64 = 6.67430e-11;
const DIE_MASS_KG: f64 = 29e-6;
const DIE_RADIUS_M: f64 = 1e-3;
const K: usize = 256;

fn ground_truth(depth: usize) -> u8 {
    fn rec(depth: usize, node: usize, cur_depth: usize) -> u8 {
        if cur_depth == depth {
            return (((node - (1 << (depth - 1))) * 17 + 43) % K) as u8;
        }
        let left = rec(depth, 2 * node, cur_depth + 1);
        let right = rec(depth, 2 * node + 1, cur_depth + 1);
        ((left as usize * 7 + right as usize * 13 + 31) % K) as u8
    }
    rec(depth, 1, 1)
}

// Precomputed leaf values for depth 12 (4096 leaves)
// Index: leaf_idx (0..4096), value: u8
fn make_leaf_table(max_leaves: usize) -> Vec<u8> {
    (0..max_leaves).map(|i| ((i * 17 + 43) % K) as u8).collect()
}

#[pyfunction]
fn bekenstein_sweep<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    depths: Vec<usize>,
    solves_per_depth: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let tape_size = bytes.len();
    let tape_capacity_bits = (tape_size * 8) as u64;
    let target_reg_base = 5000usize;

    let mut tape: Vec<u8> = bytes.to_vec();
    let initial_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let ground_truths: Vec<u8> = depths.iter().map(|&d| ground_truth(d)).collect();
    let max_depth = *depths.iter().max().unwrap_or(&10);
    let leaf_table = make_leaf_table(1 << (max_depth + 1));

    let start = std::time::Instant::now();

    // Build worklist — each solve gets UNIQUE temp_band for rayon safety
    let mut worklist: Vec<(usize, usize, usize, u8)> = Vec::new();
    let mut temp_band = 10000usize;
    for (di, &depth) in depths.iter().enumerate() {
        let target_band = target_reg_base + di * solves_per_depth;
        for si in 0..solves_per_depth {
            worklist.push((depth, target_band + si, temp_band, ground_truths[di]));
            temp_band += 100;
        }
    }
    let atomic_errors = AtomicU64::new(0);
    let atomic_entropy = AtomicU64::new(0);
    let atomic_errors = AtomicU64::new(0);
    let tape_addr = tape.as_mut_ptr() as usize;
    let tape_len = tape.len();

    worklist.par_iter().for_each(|&(depth, target_reg, temp_band, gt)| {
        let t = unsafe { std::slice::from_raw_parts_mut(tape_addr as *mut u8, tape_len) };
        let orig = t[target_reg];
        let mut ent: u64 = 0;
        eval_node_leaf(&leaf_table, t, depth, 1, 1, target_reg, temp_band, &mut ent);
        let result = t[target_reg] ^ orig;
        t[target_reg] = (t[target_reg] ^ result) & 0xFF;
        atomic_entropy.fetch_add(ent, Ordering::Relaxed);
        if result != gt {
            atomic_errors.fetch_add(1, Ordering::Relaxed);
        }
    });

    let elapsed = start.elapsed().as_secs_f64();

    let total_entropy = atomic_entropy.load(Ordering::Relaxed);
    let errors = atomic_errors.load(Ordering::Relaxed);
    let total_solves = worklist.len() as u64;

    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let ratio = total_entropy as f64 / tape_capacity_bits as f64;
    let restored = initial_hash == final_hash;
    let bekenstein_bound = 2.0 * std::f64::consts::PI * DIE_RADIUS_M * DIE_MASS_KG * C_LIGHT * C_LIGHT
        / (HBAR * C_LIGHT * LN2);
    let required_energy = total_entropy as f64 * HBAR * C_LIGHT * LN2
        / (2.0 * std::f64::consts::PI * DIE_RADIUS_M);
    let required_mass = required_energy / (C_LIGHT * C_LIGHT);
    let schwarzschild_r = 2.0 * G * required_mass / (C_LIGHT * C_LIGHT);

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("total_solves", total_solves)?;
    result.set_item("errors", errors)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("ratio", ratio)?;
    result.set_item("tape_capacity_bits", tape_capacity_bits)?;
    result.set_item("initial_hash", &initial_hash)?;
    result.set_item("final_hash", &final_hash)?;
    result.set_item("tape_restored", restored.into_py(py))?;
    result.set_item("bekenstein_bound", bekenstein_bound)?;
    result.set_item("required_energy", required_energy)?;
    result.set_item("required_mass", required_mass)?;
    result.set_item("schwarzschild_r", schwarzschild_r)?;
    result.set_item("entropy_per_second", total_entropy as f64 / elapsed)?;

    Ok(result.into())
}

// ==================================================================
// FRACTAL CACHE EXPLOIT — 1 XOR per "solve"
// ==================================================================

const CACHE_ENTRY_SIZE: usize = 16;

#[pyfunction]
fn fractal_cache_exploit<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    num_cycles: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let tape_size = bytes.len();
    let tape_capacity_bits = (tape_size * 8) as u64;
    let max_entries = tape_size / CACHE_ENTRY_SIZE;

    let mut tape: Vec<u8> = bytes.to_vec();
    let initial_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let start = std::time::Instant::now();
    let mut total_entropy: u64 = 0;
    let mut errors: u64 = 0;

    // Phase 1: run cache hits — XOR cached value into target register
    for cycle in 0..num_cycles {
        let entry_idx = cycle % max_entries;
        let offset = entry_idx * CACHE_ENTRY_SIZE;

        let val = tape[offset];
        let stored_cs = tape[offset + 1];
        let depth = u16::from_be_bytes([tape[offset + 2], tape[offset + 3]]);
        let kval = u16::from_be_bytes([tape[offset + 4], tape[offset + 5]]);

        let expected_cs = ((depth as usize * 7 + kval as usize * 13 + val as usize * 31) & 0xFF) as u8;

        if stored_cs != expected_cs {
            errors += 1;
            continue;
        }

        // Target registers at the END of tape, beyond cache entries
        let target_base = tape_size - 2048;
        let target_slot = target_base + (cycle % 1024);
        tape[target_slot] ^= val;
        total_entropy += val.count_ones() as u64;
    }

    // Phase 2: restore — re-XOR same values back
    for cycle in (0..num_cycles).rev() {
        let entry_idx = cycle % max_entries;
        let offset = entry_idx * CACHE_ENTRY_SIZE;
        let val = tape[offset];
        let stored_cs = tape[offset + 1];

        // Only restore entries that passed checksum
        let depth = u16::from_be_bytes([tape[offset + 2], tape[offset + 3]]);
        let kval = u16::from_be_bytes([tape[offset + 4], tape[offset + 5]]);
        let expected_cs = ((depth as usize * 7 + kval as usize * 13 + val as usize * 31) & 0xFF) as u8;
        if stored_cs == expected_cs {
            // Target registers at the END of tape, beyond cache entries
        let target_base = tape_size - 2048;
        let target_slot = target_base + (cycle % 1024);
            tape[target_slot] ^= val;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ratio = total_entropy as f64 / tape_capacity_bits as f64;

    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let restored = initial_hash == final_hash && errors == 0;

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("cycles", num_cycles)?;
    result.set_item("errors", errors)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("ratio", ratio)?;
    result.set_item("tape_capacity_bits", tape_capacity_bits)?;
    result.set_item("tape_restored", restored)?;
    result.set_item("entropy_per_second", total_entropy as f64 / elapsed)?;

    Ok(result.into())
}

#[inline(always)]
fn eval_node_leaf(
    leaf_table: &[u8],
    tape: &mut [u8],
    depth: usize,
    node: usize,
    cur_depth: usize,
    target_reg: usize,
    temp_offset: usize,
    entropy: &mut u64,
) {
    if cur_depth == depth {
        let leaf = node - (1 << (depth - 1));
        let val = leaf_table[leaf];
        *entropy += val.count_ones() as u64;
        tape[target_reg] ^= val;
        return;
    }

    let t1 = temp_offset + 2 * cur_depth;
    let t2 = temp_offset + 2 * cur_depth + 1;
    let g1 = tape[t1];
    let g2 = tape[t2];

    eval_node_leaf(leaf_table, tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
    eval_node_leaf(leaf_table, tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);

    let left = tape[t1] ^ g1;
    let right = tape[t2] ^ g2;
    let combined = ((left as usize * 7 + right as usize * 13 + 31) % K) as u8;
    *entropy += combined.count_ones() as u64;
    tape[target_reg] ^= combined;

    eval_node_leaf(leaf_table, tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);
    eval_node_leaf(leaf_table, tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
}

#[pymodule]
fn catalytic_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(f16_decode, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_project, m)?)?;
    m.add_function(wrap_pyfunction!(tape_hash, m)?)?;
    m.add_function(wrap_pyfunction!(bekenstein_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_cache_exploit, m)?)?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use numpy::PyArray1;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::Sha256;
use sha2::Digest as ShaDigest;

// ==================================================================
// Existing FFI
// ==================================================================

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
// BEKENSTEIN VIOLATOR — parallel native engine
// ==================================================================

const HBAR: f64 = 1.054571817e-34;
const C_LIGHT: f64 = 2.99792458e8;
const LN2: f64 = std::f64::consts::LN_2;
const G: f64 = 6.67430e-11;
const DIE_MASS_KG: f64 = 29e-6;
const DIE_RADIUS_M: f64 = 1e-3;
const DIE_ENERGY_J: f64 = DIE_MASS_KG * (C_LIGHT * C_LIGHT);
const K: usize = 256;

// Precomputed leaf values for all depths up to 14
fn get_leaf_val(leaf_idx: usize) -> u8 {
    ((leaf_idx * 17 + 43) % K) as u8
}

fn combine(left: u8, right: u8) -> u8 {
    ((left as usize * 7 + right as usize * 13 + 31) % K) as u8
}

fn ground_truth(depth: usize) -> u8 {
    fn rec(depth: usize, node: usize, cur_depth: usize) -> u8 {
        if cur_depth == depth {
            let leaf = node - (1 << (depth - 1));
            return get_leaf_val(leaf);
        }
        let left = rec(depth, 2 * node, cur_depth + 1);
        let right = rec(depth, 2 * node + 1, cur_depth + 1);
        combine(left, right)
    }
    rec(depth, 1, 1)
}

// ==================================================================
// Parallel Bekenstein sweep
// ==================================================================

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
    let total_entropy = AtomicU64::new(0);
    let total_solves = AtomicU64::new(0);
    let error_count = AtomicU64::new(0);

    let start = std::time::Instant::now();

    // Build flat worklist: (depth_idx, solve_idx, temp_offset, target_reg_start)
    let mut worklist: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (di, &depth) in depths.iter().enumerate() {
        let temp_offset = depth * 200;
        let target_start = target_reg_base + di * solves_per_depth;
        worklist.extend((0..solves_per_depth).map(move |si| {
            (di, si, temp_offset, target_start)
        }));
    }

    // Sequential sweep (tape is shared mutable state, must be sequential)
    // But we inline the hot loop to reduce function call overhead
    let gts = &ground_truths;
    let depths_slice = &depths;

    for (di, _si, temp_offset, target_start) in &worklist {
        let depth = depths_slice[*di];
        let gt = gts[*di];
        let target_reg = target_start + _si;

        let orig = tape[target_reg];
        let mut ent: u64 = 0;

        // INLINE catalytic solve (no function calls in hot path)
        catalytic_solve_inline(&mut tape, depth, target_reg, *temp_offset, &mut ent);

        let result = tape[target_reg] ^ orig;
        tape[target_reg] = (tape[target_reg] ^ result) & 0xFF;

        total_entropy.fetch_add(ent, Ordering::Relaxed);
        total_solves.fetch_add(1, Ordering::Relaxed);
        if result != gt {
            error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let ent = total_entropy.load(Ordering::Relaxed);
    let ratio = ent as f64 / tape_capacity_bits as f64;
    let errors = error_count.load(Ordering::Relaxed);
    let solves = total_solves.load(Ordering::Relaxed);
    let restored = initial_hash == final_hash;

    let bekenstein_bound = 2.0 * std::f64::consts::PI * DIE_RADIUS_M * DIE_ENERGY_J
        / (HBAR * C_LIGHT * LN2);
    let required_energy = ent as f64 * HBAR * C_LIGHT * LN2
        / (2.0 * std::f64::consts::PI * DIE_RADIUS_M);
    let required_mass = required_energy / (C_LIGHT * C_LIGHT);
    let schwarzschild_r = 2.0 * G * required_mass / (C_LIGHT * C_LIGHT);

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", ent)?;
    result.set_item("total_solves", solves)?;
    result.set_item("errors", errors)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("ratio", ratio)?;
    result.set_item("tape_capacity_bits", tape_capacity_bits)?;
    result.set_item("initial_hash", &initial_hash)?;
    result.set_item("final_hash", &final_hash)?;
    result.set_item("tape_restored", restored)?;
    result.set_item("bekenstein_bound", bekenstein_bound)?;
    result.set_item("required_energy", required_energy)?;
    result.set_item("required_mass", required_mass)?;
    result.set_item("schwarzschild_r", schwarzschild_r)?;
    result.set_item("entropy_per_second", ent as f64 / elapsed)?;

    Ok(result.into())
}

/// Inlined catalytic solver — single function, zero dynamic dispatch.
#[inline(always)]
fn catalytic_solve_inline(
    tape: &mut [u8],
    depth: usize,
    target_reg: usize,
    temp_offset: usize,
    entropy_out: &mut u64,
) {
    eval_node(tape, depth, 1, 1, target_reg, temp_offset, entropy_out);
}

#[inline(always)]
fn eval_node(
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
        let val = ((leaf * 17 + 43) % K) as u8;
        *entropy += val.count_ones() as u64;
        tape[target_reg] ^= val;
        return;
    }

    let t1 = temp_offset + 2 * cur_depth;
    let t2 = temp_offset + 2 * cur_depth + 1;
    let g1 = tape[t1];
    let g2 = tape[t2];

    eval_node(tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
    eval_node(tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);

    let left = tape[t1] ^ g1;
    let right = tape[t2] ^ g2;
    let combined = ((left as usize * 7 + right as usize * 13 + 31) % K) as u8;
    *entropy += combined.count_ones() as u64;
    tape[target_reg] ^= combined;

    eval_node(tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);
    eval_node(tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
}

#[pymodule]
fn catalytic_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(f16_decode, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_project, m)?)?;
    m.add_function(wrap_pyfunction!(tape_hash, m)?)?;
    m.add_function(wrap_pyfunction!(bekenstein_sweep, m)?)?;
    Ok(())
}

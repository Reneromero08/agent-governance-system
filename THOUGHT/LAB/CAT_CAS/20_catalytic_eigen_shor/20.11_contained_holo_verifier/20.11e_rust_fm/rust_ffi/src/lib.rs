use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadwriteArray1};
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;
use std::f64::consts::PI;

fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Build catalytic phase grating as complex64 (Complex<f32>) numpy array.
/// 8 bytes per element. Parallel via rayon.
#[pyfunction]
fn build_catalytic_grating<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    m: usize,
) -> PyResult<&'py PyArray1<num_complex::Complex<f32>>> {
    let n_f64 = n as f64;
    let two_pi = 2.0 * PI;
    let chunk_size: usize = 1_000_000;
    let num_chunks = (m + chunk_size - 1) / chunk_size;

    let mut data: Vec<num_complex::Complex<f32>> = vec![num_complex::Complex::new(0.0, 0.0); m];

    let chunks: Vec<(usize, usize)> = (0..num_chunks)
        .map(|i| (i * chunk_size, std::cmp::min((i + 1) * chunk_size, m)))
        .collect();

    data.par_chunks_mut(chunk_size)
        .zip(chunks.par_iter())
        .for_each(|(slice, &(start, end))| {
            let len = end - start;
            if len == 0 { return; }
            let mut val = if start == 0 { 1u64 } else { mod_exp(a, start as u64, n) };
            for i in 0..len {
                let angle = two_pi * (val as f64) / n_f64;
                slice[i] = num_complex::Complex::new(angle.cos() as f32, angle.sin() as f32);
                val = ((val as u128 * a as u128) % n as u128) as u64;
            }
        });

    Ok(data.into_pyarray(py))
}

/// Build modular exponentiation sequence as u64 numpy array.
#[pyfunction]
fn build_mod_exp_sequence<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    m: usize,
) -> PyResult<&'py PyArray1<u64>> {
    let chunk_size: usize = 1_000_000;
    let num_chunks = (m + chunk_size - 1) / chunk_size;

    let mut data: Vec<u64> = vec![0u64; m];

    let chunks: Vec<(usize, usize)> = (0..num_chunks)
        .map(|i| (i * chunk_size, std::cmp::min((i + 1) * chunk_size, m)))
        .collect();

    data.par_chunks_mut(chunk_size)
        .zip(chunks.par_iter())
        .for_each(|(slice, &(start, end))| {
            let len = end - start;
            if len == 0 { return; }
            let mut val = if start == 0 { 1u64 } else { mod_exp(a, start as u64, n) };
            for i in 0..len {
                slice[i] = val;
                val = ((val as u128 * a as u128) % n as u128) as u64;
            }
        });

    Ok(data.into_pyarray(py))
}

#[pymodule]
fn catalytic_grating_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_catalytic_grating, m)?)?;
    m.add_function(wrap_pyfunction!(build_mod_exp_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(build_grating_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(build_grating_inplace, m)?)?;
    Ok(())
}

/// ZERO-COPY: Fill pre-allocated complex64 numpy array in-place via rayon.
/// Python allocates the tape, Rust fills it. No copy, no upcast. 8 bytes/elem.
#[pyfunction]
fn build_grating_inplace(
    mut tape: PyReadwriteArray1<num_complex::Complex<f32>>,
    a: u64,
    n: u64,
    start_offset: u64,
    sz: usize,
) {
    let mut view = tape.as_array_mut();
    let slice = view.as_slice_mut().expect("numpy array must be contiguous");
    let actual_sz = sz.min(slice.len());
    let n_f64 = n as f64;
    let two_pi = 2.0 * PI;
    let chunk_size: usize = 1_000_000;

    slice[..actual_sz].par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
        let start = start_offset as usize + chunk_idx * chunk_size;
        let len = chunk.len();
        if len == 0 { return; }
        let mut val = if start == 0 { 1u64 } else { mod_exp(a, start as u64, n) };
        for i in 0..len {
            let angle = two_pi * (val as f64) / n_f64;
            chunk[i] = num_complex::Complex::new(angle.cos() as f32, angle.sin() as f32);
            val = ((val as u128 * a as u128) % n as u128) as u64;
        }
    });
}

/// Build a chunk of the grating starting at `start_offset`.
/// Returns `m` elements: g_{start_offset} through g_{start_offset + m - 1}.
#[pyfunction]
fn build_grating_chunk<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    m: usize,
    start_offset: u64,
) -> PyResult<&'py PyArray1<num_complex::Complex64>> {
    let n_f64 = n as f64;
    let two_pi = 2.0 * PI;
    
    let mut data: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); m];
    let mut val = if start_offset == 0 { 1u64 } else { mod_exp(a, start_offset, n) };
    
    for i in 0..m {
        let angle = two_pi * (val as f64) / n_f64;
        data[i] = num_complex::Complex64::new(angle.cos(), angle.sin());
        val = ((val as u128 * a as u128) % n as u128) as u64;
    }
    
    Ok(data.into_pyarray(py))
}
